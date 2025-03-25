import fs from 'fs';
import path from 'path';
import {
  OUTPUT_DIR,
  ROLLING_MAX_SIZE,
  TRAINING_MAX_SIZE,
} from '../tkyoDrift.js';

export default function readVectorsFromBin(ioType, object) {
  // Create an empty return object
  const resultsObject = {};

  // Iterate through each drift type to read file contents
  for (const driftType in object) {
    // Destructure the object to make the code more better
    const { dims } = object[driftType];

    // console.log(
    //   `Attempting to open ${ioType} ${driftType} Embeddings to ${OUTPUT_DIR}: `,
    //   {
    //     modelOutput: modelOutput.length,
    //     dims,
    //     byteOffset,
    //   }
    // );

    // Construct the file path for each Rolling File Type
    const rollingPath = path.join(
      OUTPUT_DIR,
      `${driftType}_${ioType}.rolling.bin`
    );

    // Load the raw binary blob
    const rollingBinContents = fs.readFileSync(rollingPath);

    // Convert the blob into a Float32 Array
    const rollingFloatArray = new Float32Array(
      rollingBinContents.buffer,
      rollingBinContents.byteOffset,
      rollingBinContents.length / 4
    );

    // Create an empty Array to house the vectors
    const rollingVectorArray = [];

    // Determine if we have less vectors than the rolling max size
    const rollingTotalVectors = Math.floor(rollingFloatArray.length / dims);
    const rollingVectorCount = Math.min(ROLLING_MAX_SIZE, rollingTotalVectors);
    const rollingStartIndex = Math.max((rollingTotalVectors - rollingVectorCount - 1), 0);

    // TODO: We set the rolling to 1, and then set the for loop to exclude the most recent entry
    // and we still got data back. The expected output should have been an empty vector, since 
    // there was only 1 vector to return and we explicitly excluded it.

    // For each dim length, push the numbers into a vector array
    for (
      let i = rollingStartIndex * dims;
      i < rollingFloatArray.length - dims;
      i += dims
    ) {
      rollingVectorArray.push(rollingFloatArray.slice(i, i + dims));
    }

    // Make object key name & save rolling data to it
    const rollingKeyName = `${driftType}${ioType}Rolling`;
    resultsObject[rollingKeyName] = rollingVectorArray;

    // Construct the file path for each Training File Type
    const trainingPath = path.join(
      OUTPUT_DIR,
      `${driftType}_${ioType}.training.bin`
    );

    // Create an empty Array to house the vectors
    const trainingVectorArray = [];

    // Check if the training data file exists
    if (fs.existsSync(trainingPath)) {
      // Load the raw binary blob
      const trainingBinContents = fs.readFileSync(trainingPath);

      // Convert the blob into a Float32 Array
      const trainingFloatArray = new Float32Array(
        trainingBinContents.buffer,
        trainingBinContents.byteOffset,
        trainingBinContents.length / 4
      );

      // Push the training data into the training array
      for (let i = 0; i < trainingFloatArray.length; i += dims) {
        trainingVectorArray.push(trainingFloatArray.slice(i, i + dims));
      }
    } else {
      // If there is no training data, push ALL rolling data into the training array
      for (let i = 0; i < rollingFloatArray.length - dims; i += dims) {
        trainingVectorArray.push(rollingFloatArray.slice(i, i + dims));
      }
    }

    // Add the training data to the results object
    const trainingKeyName = `${driftType}${ioType}Training`;
    resultsObject[trainingKeyName] = trainingVectorArray;
  }

  // Return the results
  return resultsObject;
}
