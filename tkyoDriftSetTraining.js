import fs from 'fs';
import path from 'path';
import { pipeline } from '@xenova/transformers';
import { MODELS, OUTPUT_DIR } from './tkyoDrift.js';

export default async function tkyoDriftSetTrainings(dataArray) {
  const inputTexts = dataArray.map((d) => d.input);
  const outputTexts = dataArray.map((d) => d.output);

  // Load all models once
  const loadedModels = await Promise.all(
    Object.entries(MODELS).map(([name, modelID]) =>
      pipeline('feature-extraction', modelID).then((model) => [name, model])
    )
  );
  const models = Object.fromEntries(loadedModels);

  // Ensure output dir exists
  if (!fs.existsSync(OUTPUT_DIR)) {
    fs.mkdirSync(OUTPUT_DIR, { recursive: true });
  }

  // Clear existing training files before writing new ones
  for (const driftType of Object.keys(models)) {
    for (const ioType of ['input', 'output']) {
      const filePath = path.join(
        OUTPUT_DIR,
        `${driftType}_${ioType}.training.bin`
      );

      if (fs.existsSync(filePath)) {
        fs.unlinkSync(filePath);
        console.log(`🧹 Cleared old file: ${filePath}`);
      }
    }
  }

  // Helper to embed and save a batch
  const embedAndSave = async (ioType, texts, chunkSize = 100) => {
    for (let i = 0; i < texts.length; i += chunkSize) {
      const chunk = texts.slice(i, i + chunkSize);

      for (const [index, text] of chunk.entries()) {
        const absoluteIndex = i + index;
        for (const [driftType, model] of Object.entries(models)) {
          const result = await model(text, {
            pooling: 'mean',
            normalize: true,
          });

          const buffer = Buffer.from(
            result.data.buffer,
            result.data.byteOffset,
            result.data.length * 4
          );

          const trainingPath = path.join(
            OUTPUT_DIR,
            `${driftType}_${ioType}.training.bin`
          );

          fs.appendFileSync(trainingPath, buffer);
          console.log(
            `📦 [${absoluteIndex + 1}/${
              texts.length
            }] Appended ${driftType} ${ioType} embedding to ${trainingPath}`
          );
        }
      }

      // Optional: force garbage collection if enabled with --expose-gc
      if (global.gc) {
        global.gc();
      }
    }
  };

  // Run for both inputs and outputs
  await embedAndSave('input', inputTexts);
  await embedAndSave('output', outputTexts);

  console.log('✅ Training embeddings saved successfully.');

  // Create .lock sidecar files
  for (const driftType of Object.keys(models)) {
    for (const ioType of ['input', 'output']) {
      const lockFile = path.join(
        OUTPUT_DIR,
        `${driftType}_${ioType}.training.lock`
      );
      fs.writeFileSync(lockFile, '');
      console.log(`🔐 Lock file created: ${lockFile}`);
    }
  }
}
