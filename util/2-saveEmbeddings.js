// EXPECTED INPUTS/OUTPUT
// This file is meant to take in an embedding object and save the data to the data folder
// ? The input for this function is meant to be an object: {type:vector} (there can be more than 1 type)
// ? There is no output for this function.
// If this step is failing, it is likely because the package lacks write access.
import fs from 'fs';
import path from 'path';
import {
  OUTPUT_DIR,
  TRAINING_MAX_SIZE,
  ROLLING_MAX_SIZE,
} from '../tkyoDrift.js';

// * Function to save embeddings
export default function saveEmbeddings({ ioType, embeddings }) {
  // Check if directory exists
  if (!fs.existsSync(OUTPUT_DIR)) {
    fs.mkdirSync(OUTPUT_DIR, { recursive: true });
  }

  // Iterate through each drift type
  for (const [driftType, vector] of Object.entries(embeddings)) {
    // Dynamically calculate the vector size (in case we swap embedding models)
    const vectorSize = vector.length * 4; // float32 = 4 bytes

    // * Save Data to Rolling files
    // Construct the file path to the rolling data files //TODO: Make this rolling/training specific
    const rollingPath = path.join(OUTPUT_DIR, `${driftType}_${ioType}.rolling.bin`);

    // Check how many vectors are currently in the rolling file
    const rollingSize = fs.existsSync(rollingPath)
      ? fs.statSync(rollingPath).size
      : 0;
    const rollingCount = rollingSize / vectorSize;

    // If the rolling file is full, trim the oldest vector (shift the file left)
    if (rollingCount >= ROLLING_MAX_SIZE) {
      const fileBuffer = fs.readFileSync(rollingPath);
      const trimmed = fileBuffer.subarray(vectorSize); // remove first vector
      fs.writeFileSync(rollingPath, trimmed);
    }

    // Write the new vector to the end of the rolling file
    const rollingFloatArray = new Float32Array(vector);
    const rollingBuffer = Buffer.from(rollingFloatArray.buffer);
    fs.appendFileSync(rollingPath, rollingBuffer);

    // * Save Data to Training files if they are unlocked
    // Construct the file path for the training data files
    const trainingPath = path.join(
      OUTPUT_DIR,
      `${driftType}_${ioType}.training.bin`
    );
    const lockPath = path.join(
      OUTPUT_DIR,
      `${driftType}_${ioType}.training.lock`
    );

    // Find out if the training data is locked
    const isLocked = fs.existsSync(lockPath);

    // If it's not locked
    if (!isLocked) {
      const trainingSize = fs.existsSync(trainingPath)
        ? fs.statSync(trainingPath).size
        : 0;
      const trainingCount = trainingSize / vectorSize;

      // If the count is less than the max size, add it
      if (trainingCount < TRAINING_MAX_SIZE) {
        const trainingFloatArray = new Float32Array(vector);
        const trainingBuffer = Buffer.from(trainingFloatArray.buffer);
        fs.appendFileSync(trainingPath, trainingBuffer);

        // If the max size is reached, lock the file
        if (trainingCount + 1 >= TRAINING_MAX_SIZE) {
          fs.writeFileSync(lockPath, `locked at ${new Date().toISOString()}`);
          console.log(
            `🔒 ${driftType}_${ioType} training locked at ${TRAINING_MAX_SIZE} entries.`
          );
        }
      }
    }
  }
}
