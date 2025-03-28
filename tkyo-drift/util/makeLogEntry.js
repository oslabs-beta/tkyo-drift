import fs from 'fs';
import path from 'path';
import { OUTPUT_DIR } from '../tkyoDrift.js';

export default function makeLogEntry(id, similarityObject, depth) {
  // Construct the destination to the log in the data folder
  const logPath = path.join(OUTPUT_DIR, 'drift_log.csv');

  // Create a timestamp
  const timestamp = new Date().toISOString();

   // Group similarity scores by ioType
  const grouped = {
    input: {},
    output: {},
  };

  // Dynamically unpack all similarity values
  for (const [key, value] of Object.entries(similarityObject)) {
    const [modelType, ioType, baselineType] = key.split('.');
    if (!grouped[ioType][modelType]) {
      grouped[ioType][modelType] = {};
    }
    grouped[ioType][modelType][baselineType] = value;
  }

  // Build dynamic headers based on keys present
  const modelTypes = Object.keys(
    Object.assign({}, grouped.input, grouped.output)
  );
  
  // or dynamically detect if needed
  const baselineTypes = ['rolling', 'training']; 

  const headerCols = ['ID', 'DEPTH', 'TIMESTAMP', 'I/O TYPE'];
  for (const model of modelTypes) {
    for (const baseline of baselineTypes) {
      headerCols.push(`${model.toUpperCase()} ${baseline.toUpperCase()} COS`);
    }
  }
  const headers = headerCols.join(',') + '\n';

  // Helper function to build CSV rows for input/output
  function buildRow(ioType){
    const row = [id, depth, timestamp, ioType];
    for (const model of modelTypes) {
      for (const baseline of baselineTypes) {
        const val = grouped[ioType]?.[model]?.[baseline] ?? '';
        row.push(val);
      }
    }
    return row.join(',') + '\n';
  };

  // Build the CSV rows
  const inputRow = buildRow('input');
  const outputRow = buildRow('output');

  // Check if file exists
  const fileExists = fs.existsSync(logPath);

  // Write to file
  if (!fileExists) {
    fs.writeFileSync(logPath, headers + inputRow + outputRow);
  } else {
    fs.appendFileSync(logPath, inputRow + outputRow);
  }
}