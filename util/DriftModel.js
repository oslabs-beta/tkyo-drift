import fs from 'fs';
import path from 'path';
import * as float16 from '@petamoriken/float16';
import { pipeline } from '@xenova/transformers';
import {
  OUTPUT_DIR,
  ROLLING_MAX_SIZE,
  TRAINING_MAX_SIZE,
  MODEL_CACHE,
} from '../tkyoDrift.js';

export class DriftModel {
  constructor(modelType, modelName, ioType, baselineType, depth = 0) {
    this.baselineType = baselineType;
    this.modelType = modelType;
    this.modelName = modelName;
    this.ioType = ioType;
    this.depth = depth;
    this.maxSize =
      baselineType === 'rolling' ? ROLLING_MAX_SIZE : TRAINING_MAX_SIZE;
    this.filePath = null;
    this.embedding = null;
    this.byteOffset = null;
    this.dimensions = null;
    this.vectorArray = null;
    this.baselineArray = null;
    this.embeddingModel = null;
  }

  // * Function to set the file path
  setFilePath() {
    // Construct the file path for this model
    const filepath = path.join(
      OUTPUT_DIR,
      `${this.modelType}.${this.ioType}.${this.baselineType}.bin`
    );

    // Check to see if a file exists at that path, if yes, use it
    if (fs.existsSync(filepath)) {
      this.filePath = filepath;
    } else {
      // If not, set it to use the rolling path instead
      this.filePath = path.join(
        OUTPUT_DIR,
        `${this.modelType}.${this.ioType}.rolling.bin`
      );
    }
  }

  // * Function to load the embedding model
  async loadModel() {
    // Don't reload a model if it's loaded.
    if (this.embeddingModel) return;

    // Check the global cache to see if the model was already downloaded
    if (MODEL_CACHE[this.modelName]) {
      this.embeddingModel = await MODEL_CACHE[this.modelName];
      return;
    }

    // Load the model using xenova transformer and the model ID
    this.embeddingModel = await pipeline('feature-extraction', this.modelName);
    MODEL_CACHE[this.modelName] = this.embeddingModel;
  }

  // * Function to make an embedding from an input/output pair
  async makeEmbedding(text) {
    // Invoke the load model if it hasn't been done yet
    await this.loadModel();

    // Get the embedding for the input, save to object
    const result = await this.embeddingModel(text, {
      pooling: 'mean',
      normalize: this.baselineType !== 'concept',
    });

    // Save embedding to the object
    this.embedding = result.data;

    // Save dimensions to object (the actual vector dim is at position 1)
    this.dimensions = this.embedding.length;
    //TODO: This may not exist from the return

    // save byte offset to object
    this.byteOffset = this.embedding.byteOffset;
    //TODO: This may not exist in the return
  }

  // * Function to Save Data to file path
  async saveToBin() {
    // Skip if training — this method is only for rolling baseline
    if (this.baselineType === 'training') return;

    // Convert the Float32Array to a buffer and append it to file
    const buffer = Buffer.from(this.embedding.buffer);
    await fs.promises.appendFile(this.filePath, buffer);
  }

  // * Function to read the contents of the Bins
  async readVectorsFromBin() {
    // Load the raw binary blob
    const stream = fs.createReadStream(this.filePath, {
      highWaterMark: this.dimensions * 4,
    });

    const vectorList = [];

    // Convert the blob into a Float32 Array
    for await (const chunk of stream) {
      // Guard against partial chunks
      if (chunk.length !== this.dimensions * 4) continue;

      // convert binary blob to float32
      const float32Array = new Float32Array(
        chunk.buffer,
        chunk.byteOffset,
        this.dimensions
      );
      vectorList.push(float32Array);
    }

    // Determine if we have less vectors than the rolling max size
    const totalVectors = vectorList.length;
    const vectorCount = Math.min(this.maxSize, totalVectors);

    // Calculate how many vectors we need to pull out of the float array
    const startIndex =
      this.baselineType === 'training'
        ? 0
        : Math.max(totalVectors - vectorCount - 1, 0);

    // Set vector array to an array of arrays equal to the size of vector count
    this.vectorArray = new Array(vectorCount);

    // For each dim length, push the numbers into a vector array
    for (let i = 0; i < vectorCount; i++) {
      // Calculate the start and end positions need to pull out of from the float array
      const vector = vectorList[startIndex + i];

      // Assign the vector from the float array to the vector array
      this.vectorArray[i] = vector;
    }
  }

  // * Function to get baseline value from vectorArray
  getBaseline() {
    // TODO: This is K of All Vectors, and we should not use that K value
    // Set the baseline array to the proper dimensions
    this.baselineArray = new Float32Array(this.dimensions);

    // Set each value in the baseline array equal to the mean of the vector array
    for (let i = 0; i < this.dimensions; i++) {
      this.baselineArray[i] =
        this.vectorArray.reduce(
          (accumulator, currentValue) => accumulator + currentValue[i],
          0
        ) / this.vectorArray.length;
    }
  }

  // * Function to get cosine similarity between baseline and embedding
  getCosineSimilarity() {
    // Calculate the dot product of the A and B arrays
    let dotProduct = 0;
    for (let i = 0; i < this.dimensions; i++) {
      dotProduct += this.embedding[i] * this.baselineArray[i];
    }

    // Calculate the magnitude of A
    const magnitudeA = Math.sqrt(
      this.embedding.reduce((sum, a) => sum + a * a, 0)
    );

    // Calculate the magnitude of B
    const magnitudeB = Math.sqrt(
      this.baselineArray.reduce((sum, b) => sum + b * b, 0)
    );

    // return the cosine similarity between A and B, clamping the results to prevent rounding errors
    return Math.min(1, dotProduct / (magnitudeA * magnitudeB));
  }
}
