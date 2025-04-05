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
    if (!(this.embedding.length === this.dimensions && this.embedding)) {
      // console.log(this.byteOffset)
      throw error('Error making embeddings');
    }
  }

  // * Function to Save Data to file path
  async saveToBin() {
    // Skip if training — this method is only for rolling baseline
    if (this.baselineType === 'training') return;

    try {
      // Make a new float 32 array out of the embedding
      const float32Array = new Float32Array(this.embedding);

      // Buffer the vector
      const embeddingBuffer = Buffer.from(float32Array.buffer);

      // Check if the file exists
      const fileExists = fs.existsSync(this.filePath);

      // If the file doesn't exist, add the vector, and write a new header
      if (!fileExists) {
        // Allocate space for the header shit
        const headerBuffer = Buffer.alloc(8);

        // Total vectors is 1 on first write
        headerBuffer.writeUInt32LE(1, 0);

        // Update the header with vector dimensions
        headerBuffer.writeUInt32LE(this.dimensions, 4);

        // Concatenate the header data with the vector data
        const fullBuffer = Buffer.concat([headerBuffer, embeddingBuffer]);

        // Write the header to the file
        await fs.promises.writeFile(this.filePath, fullBuffer);

        // If the file does exist, append the vector, and update the existing header
      } else {
        // Append new vector
        await fs.promises.appendFile(this.filePath, embeddingBuffer);

        // Recalculate new vector count
        const stats = await fs.promises.stat(this.filePath);
        const vectorsInBinCount = Math.floor(
          (stats.size - 8) / (this.dimensions * 4)
        );

        // Update header: numVectors
        const headerBuffer = Buffer.alloc(4);
        headerBuffer.writeUInt32LE(vectorsInBinCount, 0);

        const fileHandle = await fs.promises.open(this.filePath, 'r+');
        // Write just the first 4 bytes, skip the 2nd 4 since the dimensions don't change.
        await fileHandle.write(headerBuffer, 0, 4, 0);
        await fileHandle.close();
      }
    } catch (error) {
      throw new Error(`Error saving to binary file: ${error.message}`);
    }
  }

  // * Function to read the contents of the Bins, Build an HNSW, and then get the
  async readFromBin() {
    // ! This is for testing, remove later
    // console.log("Embedding sent to Python (first 5):", this.embedding.slice(0, 5));

    return new Promise((resolve, reject) => {
      const pyProg = spawn('python3', [
        './util/pythonHNSW.py',
        this.ioType,
        this.modelType,
        JSON.stringify(Array.from(this.embedding)),
        this.baselineType,
      ]);

      let result = '';
      let error = '';

      // This function is for accepting to data from python
      // Data is the binary form of the result from python
      pyProg.stdout.on('data', (data) => {
        // Result is the stringified version of the result from python
        result += data.toString();
      });

      // This function is for error handling
      // Data is the binary from of the error from python
      pyProg.stderr.on('data', (data) => {
        console.error('[PYTHON STDERR]', data.toString()); // ! This is for testing, remove later
        // Error is the stringified version of the error from python
        error += data.toString();
      });

      pyProg.on('close', (code) => {
        if (code !== 0) {
          reject(new Error(`Python process failed: ${error}`));
          return;
        }

        // TODO: distances here is a set of euclidean distances, but from what to what?
        // ? If this is the euclidean dist from the HNSW's nearest neighbor to the input vector
        // ? then should we pass this directly to the log maker function?
        // Destructure from result after parsing the result, changing it from a string to an object
        const { centroids, distances } = JSON.parse(result);

        // TODO: vectorArray was named before we were using HNSW, so presumably it needs a rename
        // Assign the output of the centroids to vectorArray
        this.vectorArray = centroids;

        resolve({ centroids, distances });
      });
    });
  }

  // * Function to get baseline value from vectorArray
  getBaseline() {
    // If readFromBin returns a single vector, skip the math and return out.
    if (this.vectorArray.length === 1) {
      this.baselineArray = new Float32Array(this.vectorArray[0]);
      return;
    }

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

    // Sanity check: Make sure the baseline is valid
    const valid = this.baselineArray.every(
      (val) => typeof val === 'number' && !Number.isNaN(val)
    );

    if (!valid || this.baselineArray.length !== this.dimensions) {
      throw error(
        'Error getting baseline: invalid values or dimension mismatch'
      );
    }
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

  // * Function to calculate the euclidean distance from the baseline
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
