// EXPECTED INPUTS/OUTPUT
// This file is meant to take in an embedding object and save the data to the data folder
// ? The input for this function is meant to be an object: {type:[vector]} (there can be more than 1 type)
// ? There is no output for this function.
// If this step is failing, it is likely because the package lacks write access.
import fs from 'fs';
import path from 'path';
import { OUTPUT_DIR } from '../tkyoDrift.js';

// * Function to save embeddings
export default function saveEmbeddings(ioType, object) {
  // Iterate through each drift type
  for (const driftType in object) {
    // Destructure the object to make the code more better
    const { modelOutput, dims, byteOffset } = object[driftType];

    console.log(
      `Attempting to save ${ioType} ${driftType} Embeddings to ${OUTPUT_DIR}: `,
      {
        modelOutput: modelOutput.length,
        dims,
        byteOffset,
      }
    );

    // * Save Data to Rolling files
    // Construct the file path to the rolling data files
    const rollingPath = path.join(
      OUTPUT_DIR,
      `${driftType}_${ioType}.rolling.bin`
    );

    // Write the new vector to the end of the rolling file
    console.log('👇 Attempting to Write to Rolling Bin File');
    const rollingBuffer = Buffer.from(modelOutput.buffer, byteOffset, dims * 4);
    fs.appendFileSync(rollingPath, rollingBuffer);
  }
}
