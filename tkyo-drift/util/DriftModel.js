import fs from 'fs';
import path from 'path';
import { pipeline } from '@xenova/transformers';
import { spawn } from 'child_process';
import { OUTPUT_DIR, MODEL_CACHE } from '../tkyoDrift.js';
import { error } from 'console';

export class DriftModel {
  constructor(modelType, modelName, ioType, baselineType, depth = 0) {
    this.baselineType = baselineType;
    this.modelType = modelType;
    this.modelName = modelName;
    this.ioType = ioType;
    this.depth = depth;
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
    try {
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
    } catch (error) {
      throw Error(`Error in setFilePath: ${error.message}`);
    }
  }

  // * Function to load the embedding model
  async loadModel() {
    try {
      // Don't reload a model if it's loaded.
      if (this.embeddingModel) return;

      // Check the global cache to see if the model was already downloaded
      if (MODEL_CACHE[this.modelName]) {
        this.embeddingModel = await MODEL_CACHE[this.modelName];
        return;
      }

      // Load the model using xenova transformer and the model ID
      this.embeddingModel = await pipeline(
        'feature-extraction',
        this.modelName
      );
      MODEL_CACHE[this.modelName] = this.embeddingModel;
    } catch (error) {
      throw Error(`Error in loadModel: ${error.message}`);
    }
  }

  // * Function to make an embedding from an input/output pair
  async makeEmbedding(text) {
    try {
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

      // save byte offset to object
      this.byteOffset = this.embedding.byteOffset;

      // If we throw an error here, it should halt the rest of the code
      if (!(this.embedding.length === this.dimensions)) {
        throw error('Dimension Mismatch');
      }

      if (!this.byteOffset === result.data.byteOffset){
        throw error('ByteOffset mismatch');
      }
      
    } catch (error) {
      throw Error(`Error in makeEmbedding: ${error.message}`);
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

      // Validate embedding dimensions BEFORE writing
      if (float32Array.length !== this.dimensions) {
        throw new Error(
          `Dimension mismatch: embedding has ${float32Array.length} values, expected ${this.dimensions}`
        );
      }

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
        // Validate the file header matches this.dimensions BEFORE writing
        const fd = await fs.promises.open(this.filePath, 'r');
        const headerBuffer = Buffer.alloc(8);
        await fd.read(headerBuffer, 0, 8, 0);
        await fd.close();

        const fileVectorDims = headerBuffer.readUInt32LE(4);
        if (fileVectorDims !== this.dimensions) {
          throw new Error(
            `File dimension mismatch: file expects ${fileVectorDims}, embedding has ${this.dimensions}`
          );
        }

        // Append new vector
        await fs.promises.appendFile(this.filePath, embeddingBuffer);

        // Recalculate new vector count
        const stats = await fs.promises.stat(this.filePath);
        const vectorsInBinCount = Math.floor(
          (stats.size - 8) / (this.dimensions * 4)
        );

        // Update header: numVectors
        const fullHeaderBuffer = Buffer.alloc(8);
        fullHeaderBuffer.writeUInt32LE(vectorsInBinCount, 0);
        fullHeaderBuffer.writeUInt32LE(this.dimensions, 4);

        const fileHandle = await fs.promises.open(this.filePath, 'r+');
        // Rewrite the full header 8 bytes to prevent bin corruption
        // ! Once upon a time, we only updated the first 4 bytes. It broke everything.
        await fileHandle.write(headerBuffer, 0, 8, 0);
        await fileHandle.close();
      }
    } catch (error) {
      throw Error(`Error in saveToBin: ${error.message}`);
    }
  }

  // * Function to read the contents of the Bins, Build an HNSW, and then get the
  async readFromBin() {
    try {
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

          // Assign the output of the centroids to vectorArray
          this.vectorArray = centroids;

          resolve({ centroids, distances });
        });
      });
    } catch (error) {
      throw Error(`Error in readFromBin: ${error.message}`);
    }
  }

  // * Function to get baseline value from vectorArray
  getBaseline() {
    try {
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
    } catch (error) {
      throw Error(`Error in getBaseline: ${error.message}`);
    }
  }

  // * Function to get cosine similarity between baseline and embedding
  getCosineSimilarity() {
    try {
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
    } catch (error) {
      throw Error(`Error in getCosineSimilarity: ${error.message}`);
    }
  }

  // TODO: We are already calculating the EUD when making the HNSW, so we should use that if we have it, and if not, calculate it
  // * Function to calculate the euclidean distance from the baseline
  getEuclideanDistance() {
    try {
      // Calculate the distance between the embedding and baselineArray
      return Math.sqrt(
        this.embedding.reduce(
          (sum, a, i) => sum + (a - this.baselineArray[i]) ** 2,
          0
        )
      );
    } catch (error) {
      throw Error(`Error in getEuclideanDistance: ${error.message}`);
    }
  }
}
