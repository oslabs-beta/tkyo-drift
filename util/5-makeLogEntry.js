import fs from 'fs';
import path from 'path';
import { fileURLToPath } from 'url';

// TODO: I got lazy and had GPT write this, it feels like there is a mistake here? We are rebuilding __dirname
// Reconstruct __dirname in ES6
const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

// Construct the destination to the log in the data folder
const DEFAULT_LOG_FILE = path.join(__dirname, '..', 'data', 'drift_log.csv');

export default function makeLogEntry(
  { id, ioType, rollingCos, trainingCos, idealCos },
  logFilePath = DEFAULT_LOG_FILE
) {
  const timestamp = new Date().toISOString(); 
  const headers = 'ID,TIMESTAMP,I/O TYPE,ROLLING COS,TRAINING COS,IDEAL COS\n';
  const row = `${id},${timestamp},${ioType},${rollingCos},${trainingCos},${idealCos}\n`;

  const fileExists = fs.existsSync(logFilePath);

  if (!fileExists) {
    fs.writeFileSync(logFilePath, headers + row);
  } else {
    fs.appendFileSync(logFilePath, row);
  }
}
