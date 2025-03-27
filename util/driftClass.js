import fs from 'fs';
import zlib from 'zlib'
import path from 'path';
import { pipeline } from '@xenova/transformers';
import {
  OUTPUT_DIR,
  ROLLING_MAX_SIZE,
  TRAINING_MAX_SIZE,
  MODEL_CACHE,
} from '../tkyoDrift.js';

export class DriftModel {
  constructor(modelType, modelName, ioType, baselineType) {
    this.baselineType = baselineType;
    this.modelType = modelType;
    this.modelName = modelName;
    this.ioType = ioType;
    this.maxSize =
      baselineType === 'rolling' ? ROLLING_MAX_SIZE : TRAINING_MAX_SIZE;
    this.filePath = null;
    this.embedding = null;
    this.byteOffset = null;
    this.dimensions = null;
    this.vectorArray = null;
    this.embeddingModel = null;
    this.baselineArray = null;
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
      // TODO: With normalize set to true, we will never get results below 0
      normalize: true,
    });

    // Save embedding to the object
    this.embedding = result.data;

    // Save dimensions to object (the actual vector dim is at position 1)
    this.dimensions = result.dims[1];
    //TODO: This may not exist from the return

    // save byte offset to object
    this.byteOffset = this.embedding.byteOffset;
    //TODO: This may not exist in the return
  }

  // * Function to Save Data to file path
  async saveToBin() {
    // Check is the model's baseline is training, and abort if true
    if (this.baselineType === 'training') return;

    // Turn the embedding into a blob and define it's offset/dimensions
    const buffer = Buffer.from(
      this.embedding.buffer,
      this.byteOffset,
      this.dimensions * 4
    );

    // TODO: This is still 10k vectors per 2gbs, and we cant change that with less precision.
    // ? We can, however, compress and decompress files after write and before read. (tis a speed hit tho)
    // Append the binary blob to the file path
    fs.promises.appendFile(this.filePath, buffer);
  }

  // TODO: We should check to see if a library for this exists. Its probably faster
  // * Function to read the contents of the Bins
  // async readVectorsFromBin() {
  //   // Load the raw binary blob
  //   const binContents = await fs.promises.readFile(this.filePath);

  //   // Convert the blob into a Float32 Array
  //   const floatArray = new Float32Array(
  //     binContents.buffer,
  //     binContents.byteOffset,
  //     binContents.length / 4
  //   );

  //   // Determine if we have less vectors than the rolling max size
  //   const totalVectors = Math.floor(floatArray.length / this.dimensions);
  //   const vectorCount = Math.min(this.maxSize, totalVectors);

  //   // Calculate how many vectors we need to pull out of the float array
  //   const startIndex =
  //     this.baselineType === 'training'
  //       ? 0
  //       : Math.max(totalVectors - vectorCount - 1, 0);

  //   // Set vector array to an array of arrays equal to the size of vector count
  //   // TODO: This needs to be a float 32 array
  //   this.vectorArray = new Array(vectorCount);

  //   // For each dim length, push the numbers into a vector array
  //   for (let i = 0; i < vectorCount; i++) {
  //     // Calculate the start and end positions need to pull out of from the float array
  //     const offset = (startIndex + i) * this.dimensions;

  //     // Assign the vector from the float array to the vector array
  //     this.vectorArray[i] = floatArray.slice(offset, offset + this.dimensions);
  //   }
  // }

  // !fs read file has an upper limit on 2gb files, which is 10k of the big vectors. This version fixes that
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
      
      const floatArray = new Float32Array(
        chunk.buffer,
        chunk.byteOffset,
        this.dimensions
      );
      vectorList.push(floatArray);
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

  // TODO: This is K of All Vectors, and we should not use that K value
  // * Function to get baseline value from vectorArray
  getBaseline() {
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

    // return the cosine similarity between A and B
    return dotProduct / (magnitudeA * magnitudeB);
  }
}
