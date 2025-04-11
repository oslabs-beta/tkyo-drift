import fs from 'fs';
import readline from 'readline';
import path from 'path';

// * Function to read scalar metrics from the scalar jsonl files and group them by metric name
export async function loadScalarMetrics(
  metricNames,
  ioType,
  baselineType,
  // ! Note that most scalar metrics do not give a shit what model they come from, and only L2 Norm and Token Length do
  modelType = null
) {
  const metrics = {}; // this will hold the final merged metric data

  for (const metric of metricNames) {
    // Construct the full path to the JSONL file for this metric
    const filePath = modelType
      ? path.join(
          'data/scalars',
          // ? If the scalar metric is model specific, this will catch it (when this function gets invoked with a model value)
          `${ioType}.${metric}.${modelType}.${baselineType}.scalar.jsonl`
        )
      : path.join(
          'data/scalars',
          // ? Otherwise, the scalar metric will come from a model agnostic file
          `${ioType}.${metric}.${baselineType}.scalar.jsonl`
        );

    // Skip if the file doesn't exist (can happen in partially populated environments)
    if (!fs.existsSync(filePath)) continue;

    // Create a line reader for the .jsonl file
    const fileStream = fs.createReadStream(filePath);
    const rl = readline.createInterface({
      input: fileStream,
      crlfDelay: Infinity,
    });

    // Read each line from the file
    for await (const line of rl) {
      try {
        // Parse the JSON line (each line is a complete JSON object)
        const entry = JSON.parse(line);

        // Extract the scalar value from the "metrics" block
        const val = entry.metrics?.[metric];

        // Make sure it's a number before storing it
        if (typeof val === 'number') {
          if (!metrics[metric]) metrics[metric] = [];
          metrics[metric].push(val);
        }
      } catch (err) {
        console.warn(`Could not parse line in ${filePath}:`, err.message);
      }
    }
  }

  return metrics; // returns a dictionary of arrays keyed by metric name
}
