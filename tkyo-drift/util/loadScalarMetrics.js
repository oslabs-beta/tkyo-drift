import fs from 'fs';
import readline from 'readline';

// * Function to read scalar metrics from the scalar jsonl files and group them by metric name
export async function loadScalarMetrics(filePath) {
  // Initialize an object to hold the grouped metrics 
  const metrics = {};

  // Create a file stream and readline interface to process the file line by line
  const fileStream = fs.createReadStream(filePath, { encoding: 'utf-8' });
  const rl = readline.createInterface({ input: fileStream });

  // Loop through each line of the file
  for await (const line of rl) {
    // Skip empty lines
    if (!line.trim()) continue;

    try {
      // Parse the line into a JSON object
      const entry = JSON.parse(line);
      const values = entry.metrics;

      // Group each metric value by its key (e.g., norm, textLength)
      for (const [key, value] of Object.entries(values)) {
        if (!metrics[key]) {
          metrics[key] = [];
        }
        metrics[key].push(value);
      }
    } catch (err) {
      // Warn and skip if the line is malformed
      console.warn(`⚠️ Failed to parse line: ${line}`);
    }
  }

  // Return the grouped metrics
  return metrics;
}