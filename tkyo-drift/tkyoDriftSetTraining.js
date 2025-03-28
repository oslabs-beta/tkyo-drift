import fs from 'fs';
import path from 'path';
import { pipeline } from '@xenova/transformers';
import { setFloat16 } from '@petamoriken/float16';
import { MODELS, OUTPUT_DIR } from './tkyoDrift.js';

export default async function tkyoDriftSetTrainings(dataArray) {
  const inputTexts = dataArray.map((d) => d.input);
  const outputTexts = dataArray.map((d) => d.output);

  // ------------- << Load All Models >> -------------
  // * Load each transformer pipeline once for all drift types
  const loadedModels = await Promise.all(
    Object.entries(MODELS).map(([name, modelID]) =>
      pipeline('feature-extraction', modelID).then((model) => [name, model])
    )
  );
  const models = Object.fromEntries(loadedModels);

  // ------------- << Prepare Output Directory >> -------------
  // * Ensure the data directory exists before writing files
  if (!fs.existsSync(OUTPUT_DIR)) {
    fs.mkdirSync(OUTPUT_DIR, { recursive: true });
  }

  // ------------- << Clear Old Training Files >> -------------
  // * Wipe out previous training bins to avoid contamination
  for (const driftType of Object.keys(models)) {
    for (const ioType of ['input', 'output']) {
      const filePath = path.join(
        OUTPUT_DIR,
        `${driftType}.${ioType}.training.bin`
      );

      if (fs.existsSync(filePath)) {
        fs.unlinkSync(filePath);
        console.log(`🧹 Cleared old file: ${filePath}`);
      }
    }
  }

  // ------------- << Helper: Embed & Write Vectors >> -------------
  // * Run batching to reduce memory usage and avoid timeouts
  const embedAndSave = async (ioType, texts, chunkSize = 100) => {
    // For each model, embed and write the entire set of vectors
    for (const [driftType, model] of Object.entries(models)) {
      // Construct the destination path and open a write stream
      const trainingPath = path.join(
        OUTPUT_DIR,
        `${driftType}.${ioType}.training.bin`
      );
      const writeStream = fs.createWriteStream(trainingPath, { flags: 'a' });

      let totalWritten = 0;

      // Chunks the dataset into smaller groups (100 at a time)
      for (let i = 0; i < texts.length; i += chunkSize) {
        const chunk = texts.slice(i, i + chunkSize);

        // Convert each text into an embedding buffer
        for (const text of chunk) {
          const result = await model(text, {
            pooling: 'mean',
            normalize: true,
          });

          // Make a float 16 buffer from the input data
          const buffer = new ArrayBuffer(result.data.length * 2);
          const view = new DataView(buffer);
          for (let i = 0; i < result.data.length; i++) {
            setFloat16(view, i * 2, result.data[i]);
          }

          // Write the binary buffer to the file stream
          writeStream.write(Buffer.from(buffer));
          totalWritten++;
        }

        // Log progress
        console.log(
          `📦 [${totalWritten}/${texts.length}] ${driftType} ${ioType}`
        );
      }

      // Close the write stream after processing all vectors
      writeStream.end();
    }
  };

  // ------------- << Run Training Embeddings >> -------------
  await embedAndSave('input', inputTexts);
  await embedAndSave('output', outputTexts);
  console.log('✅ Training embeddings saved successfully.');

  // ------------- << Create Lock Files >> -------------
  // * Write lock sidecars to signal completed baseline setup
//   for (const driftType of Object.keys(models)) {
//     for (const ioType of ['input', 'output']) {
//       const lockFile = path.join(
//         OUTPUT_DIR,
//         `${driftType}.${ioType}.training.lock`
//       );
//       fs.writeFileSync(lockFile, '');
//       console.log(`🔐 Lock file created: ${lockFile}`);
//     }
//   }
}


// Adjust this path to be relative to where you're running the script
const datasetPath = path.resolve('../aiModel/dataset.json');

// Read and parse the dataset
const rawData = fs.readFileSync(datasetPath, 'utf-8');
const dataArray = JSON.parse(rawData);

// Run training ingestion
tkyoDriftSetTrainings(dataArray);
