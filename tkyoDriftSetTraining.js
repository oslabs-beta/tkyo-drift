// TODO: This whole file needs to be vetted, much of it was done at 2am with hefty gpt help.
// * It is the BATCH version of makeEmbeddings AND saveEmbeddings,
// * which should only ever be called by when a full training data set is provided

import fs from 'fs';
import path from 'path';
import readline from 'readline';
import { fileURLToPath } from 'url';
import { pipeline as transformersPipeline } from '@xenova/transformers';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

const CONCEPT_MODEL = 'Xenova/all-MiniLM-L6-v2';
const SEMANTIC_MODEL = 'Xenova/distilroberta-base';

// Helper: mean pooling across token vectors
function mean(vectors) {
  const dim = vectors[0].length;
  const sum = new Array(dim).fill(0);

  for (const vec of vectors) {
    for (let i = 0; i < dim; i++) {
      sum[i] += vec[i];
    }
  }

  return sum.map(v => v / vectors.length);
}

export default async function buildTrainingEmbeddings(jsonlPath, concurrency = 4) {
  const ioTypes = ['input', 'output'];
  const dataDir = path.join(__dirname, 'data');
  const lockPaths = {};
  const streams = {};

  // Load models
  const conceptEmbedder = await transformersPipeline('feature-extraction', CONCEPT_MODEL);
  const semanticEmbedder = await transformersPipeline('feature-extraction', SEMANTIC_MODEL);

  // Setup file streams and clear existing files
  for (const ioType of ioTypes) {
    const filePath = path.join(dataDir, `training_${ioType}_embeddings.jsonl`);
    const lockPath = filePath + '.lock';
    lockPaths[ioType] = lockPath;

    if (fs.existsSync(filePath)) fs.rmSync(filePath);
    if (fs.existsSync(lockPath)) fs.rmSync(lockPath);
    fs.mkdirSync(path.dirname(filePath), { recursive: true });

    streams[ioType] = fs.createWriteStream(filePath, { flags: 'a' });
  }

  // Stream dataset line by line
  const fileStream = fs.createReadStream(jsonlPath);
  const rl = readline.createInterface({ input: fileStream });

  let queue = [];
  let active = 0;
  let processed = 0;

  const enqueue = async (item) => {
    while (active >= concurrency) {
      await new Promise(r => setTimeout(r, 10)); // wait until a worker is free
    }

    active++;
    processItem(item).then(() => {
      active--;
      processed++;
      if (processed % 100 === 0) {
        console.log(`✅ Processed ${processed}`);
      }
    });
  };

  const processItem = async (item) => {
    try {
      for (const ioType of ioTypes) {
        const text = item[ioType];
        if (!text) continue;
  
        const conceptTensor = await conceptEmbedder(text);
        const semanticTensor = await semanticEmbedder(text);
  
        const conceptPooled = mean((await conceptTensor.tolist())[0]);
        const semanticPooled = mean((await semanticTensor.tolist())[0]);
  
        const embeddingObj = {
          concept: conceptPooled,
          semantic: semanticPooled
        };
  
        streams[ioType].write(JSON.stringify(embeddingObj) + '\n');
      }
    } catch (err) {
      console.warn(`❌ Failed to process item: ${err.message}`);
    }
  };

  for await (const line of rl) {
    if (line.trim()) {
      try {
        const item = JSON.parse(line);
        enqueue(item);
      } catch (err) {
        console.warn('⚠️ Skipping invalid line:', err.message);
      }
    }
  }

  // Wait for remaining tasks
  while (active > 0) {
    await new Promise(r => setTimeout(r, 100));
  }

  // Finalize
  for (const ioType of ioTypes) {
    streams[ioType].end();
    fs.writeFileSync(lockPaths[ioType], 'LOCKED\n');
  }

  console.log('🎉 All training embeddings built and locked.');
}

