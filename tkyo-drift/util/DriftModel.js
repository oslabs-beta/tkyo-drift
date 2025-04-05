import fs from 'fs';
import path from 'path';
import { pipeline } from '@xenova/transformers';
import { spawn } from 'child_process';
import {
  OUTPUT_DIR,
  ROLLING_MAX_SIZE,
  TRAINING_MAX_SIZE,
  MODEL_CACHE,
} from '../tkyoDrift.js';
import { error } from 'console';

export class DriftModel {
  constructor(modelType, modelName, ioType, baselineType, depth = 0) {
    this.baselineType = baselineType;
    this.modelType = modelType;
    this.modelName = modelName;
    this.ioType = ioType;
    this.depth = depth;
    this.maxSize = // TODO: What happens when someone loads 50k training files, are we still limiting the max size to N?
      baselineType === 'rolling' ? ROLLING_MAX_SIZE : TRAINING_MAX_SIZE;
    this.filePath = null;
    this.embedding = null;
    this.byteOffset = null;
    this.dimensions = null;
    this.vectorArray = null;
    this.baselineArray = null;
    this.embeddingModel = null;
  }

  //TODO Work on global error handling
  //TODO Address read/write operation order

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

    let normalizeType = true;
    if (this.modelType === 'concept') {
      normalizeType = false;
    }
    // Get the embedding for the input, save to object
    const result = await this.embeddingModel(text, {
      pooling: 'mean',
      normalize: normalizeType,
    });
    // console.log(result.data)
    // Save embedding to the object
    this.embedding = result.data;

    // Save dimensions to object (the actual vector dim is at position 1)
    this.dimensions = this.embedding.length;
    //TODO: This may not exist from the return

    // save byte offset to object
    this.byteOffset = this.embedding.byteOffset;
    //TODO: This may not exist in the return

    // if we throw an error here, it should halt the rest of the code
    //our check will throw the error if we do not have a match on embedding.length and dims. Given that those are the same, we're being efficient by checking that the two values exist and that they're equal. For some reason, checking if this.byteOffset exists is returning false (e.g. if(this.byteOffset)), despite consistently console logging as "0". Given the todo above, I think we can leave byteOffset out.
    // console.log(this.embedding, this.dimensions, this.byteOffset)
    if (!((this.embedding.length === this.dimensions) && this.embedding)) {
      // console.log(this.byteOffset)
      throw error('Error making embeddings');
    }
  }

  // * Function to Save Data to file path
  async saveToBin() {
    // Skip if training — this method is only for rolling baseline
    if (this.baselineType === 'training') return;

    // adding in a try/catch block because this doesn't update anything in our constructor, so we'll need to use this. I don't know if this catch block will ever trigger.
    try {
      // Create a Float32Array from the embedding
      const float32Array = new Float32Array(this.embedding);

      // Convert to Node buffer and write to disk
      const buffer = Buffer.from(float32Array.buffer);
      await fs.promises.appendFile(this.filePath, buffer);
    } catch (error) {
      throw error('error saving to binary file', error);
    }
  }

  // * Function to read the contents of the Bins
  async readFromBin() {
    console.time(this.filePath);
    // Load the raw binary blob (4 bytes per value, saved as float32)
    const stream = fs.createReadStream(this.filePath, {
      highWaterMark: this.dimensions * 4,
    });

    // Make a placeholder storage array
    const vectorList = [];

    // Iterate through each chunk from the data stream
    for await (const chunk of stream) {
      // Guard against partial chunks
      if (chunk.length !== this.dimensions * 4) continue;
      // Interpret the chunk directly as Float32Array
      const float32Array = new Float32Array(
        chunk.buffer,
        chunk.byteOffset,
        this.dimensions
      );
      // Push to the storage array
      vectorList.push(float32Array);
    }
    // console.log(this.filePath,vectorList[0])

    // Determine if we have less vectors than the rolling max size
    const totalVectors = vectorList.length;
    const vectorCount = Math.min(this.maxSize, totalVectors);

    // console.log(this.filePath,this.dimensions,totalVectors)
    // Calculate the start index based on rolling or training window
    const startIndex =
      this.baselineType === 'training'
        ? 0
        : Math.max(totalVectors - vectorCount - 1, 0);

    // Set vector array to an array of arrays equal to the size of vector count
    this.vectorArray = new Array(vectorCount);

    // Populate the final vector array with the most recent N vectors
    for (let i = 0; i < vectorCount; i++) {
      // Calculate the start and end positions need to pull out of from the float array
      const vector = vectorList[startIndex + i];

      // Assign the vector from the float array to the vector array
      this.vectorArray[i] = vector;
    }
    console.timeEnd(this.filePath);
    // initialize a length checker to be the training max size for training and the rolling max size for not trainings. We are verifying that the vector array length for each model is equal to the appropriate size (currently 10 for rolling and 10000 for training), and also checking to see if there aren't values listed in totalVectors (totalVectors counts the number of vectors in the file, so if it is falsy we have a problem)
    const lengthCheck =
      this.baselineType === 'training' ? TRAINING_MAX_SIZE : ROLLING_MAX_SIZE;
    if (this.vectorArray.length !== lengthCheck || !totalVectors) {
      throw error('error reading from binary file');
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
    // this checks two things. First that our baseline array length is equal to dims (should always be true), and second that the baseline array at index 0 is between 1 and -1, inclusively. Currently commented out because our data is returning some NaNs. I'm leaving it at this point right now, but rather than checking at [0], we should probably include it in the for loop so we check every value, not just one. That would be more robust.
    if (
      this.baselineArray.length !== this.dimensions
      //
      // || !(this.baselineArray[0]<=1 || this.baselineArray>=-1)
    ) {
      throw error('Error getting baseline');
    }
    // console.log(this.baselineArray.length, this.dimensions)
  }
  // TODO: add error handling for getCosineSimilarity and getEuclideanDistance
  // * Function to get cosine similarity between baseline and embedding
  getCosineSimilarity() {
    // Calculate the dot product of the A and B arrays
    let dotProduct = 0;
    for (let i = 0; i < this.dimensions; i++) {
      dotProduct += this.embedding[i] * this.baselineArray[i];
      // commenting this out because I am having some trouble identifying whether or not dotProduct is an actual number. NaN is still classified as number, and we can't successfully check if dotProduct is strictly equal to NaN to throw an error at that point.
      // if (dotProduct === NaN){
      //   console.log("NaN")
      // }
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

  getEuclideanDistance() {
    // Calculate the distance between the embedding and baselineArray
    return Math.sqrt(
      this.embedding.reduce(
        (sum, a, i) => sum + (a - this.baselineArray[i]) ** 2,
        0
      )
    );
  }
}
