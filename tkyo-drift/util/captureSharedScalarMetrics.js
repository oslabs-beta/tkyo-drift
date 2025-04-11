import fs from 'fs/promises';
import path from 'path';
import { OUTPUT_DIR } from '../tkyoDrift.js';

// Calculates the shared scalar values for a given input/output pair
export default async function captureSharedScalarMetrics(
  input,
  output,
) {
  const timestamp = new Date().toISOString();

  const sharedMetrics = {
    input: computeMetrics(input),
    output: computeMetrics(output),
  };

  // Write each metric to its respective file
  await Promise.all(
    Object.entries(sharedMetrics).flatMap(([ioType, metricSet]) => {
      return Object.entries(metricSet).map(([metric, value]) => {
        const filePath = path.join(
          OUTPUT_DIR,
          'scalars',
          `${ioType}.${metric}.rolling.scalar.jsonl`
        );
        const line =
          JSON.stringify({ timestamp, metrics: { [metric]: value } }) + '\n';
        return fs.appendFile(filePath, line);
      });
    })
  );
}

// Internal helper to calculate scalar metrics for a given string
function computeMetrics(text) {
  const metrics = {};

  // * Get Raw Input Length
  metrics.characterLength = text.length;

  // * Get Character Entropy
  const counts = {};
  for (const char of text) counts[char] = (counts[char] || 0) + 1;
  metrics.characterEntropy = -Object.values(counts).reduce((sum, count) => {
    const p = count / text.length;
    return sum + p * Math.log2(p);
  }, 0);

  // * Get Average Word Length
  const words = text.split(/\s+/);
  metrics.avgWordLength =
    words.length > 0
      ? words.reduce((sum, word) => sum + word.length, 0) / words.length
      : 0;

  // * Get Punctuation Density
  metrics.punctuationDensity =
    (text.match(/[.,!?;:]/g)?.length || 0) / text.length;

  // * Get Uppercase Ratio
  metrics.uppercaseRatio = (text.match(/[A-Z]/g)?.length || 0) / text.length;

  return metrics;
}
