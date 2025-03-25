import fs from 'fs';
import path from 'path';
import { OUTPUT_DIR } from '../tkyoDrift.js';

// TODO: I got lazy and had GPT write this, it feels like there is a mistake here? We are rebuilding __dirname


export default function makeLogEntry(
  id,
  inputSimilarityObject,
  outputSimilarityObject
) {
  // Construct the destination to the log in the data folder
  const logPath = path.join(OUTPUT_DIR, 'drift_log.csv');

  const {
    semanticinputRolling,
    semanticinputTraining,
    conceptinputRolling,
    conceptinputTraining,
  } = inputSimilarityObject;

  const {
    semanticoutputRolling,
    semanticoutputTraining,
    conceptoutputRolling,
    conceptoutputTraining,
  } = outputSimilarityObject;

  const timestamp = new Date().toISOString();
  const headers = `ID,TIMESTAMP,I/O TYPE,SEMANTIC ROLLING COS,SEMANTIC TRAINING COS,CONCEPT ROLLING COS,CONCEPT TRAINING COS\n`;
  const inputRow = `${id},${timestamp},input,${semanticinputRolling},${semanticinputTraining},${conceptinputRolling},${conceptinputTraining}\n`;
  const outputRow = `${id},${timestamp},output,${semanticoutputRolling},${semanticoutputTraining},${conceptoutputRolling},${conceptoutputTraining}\n`;

  const fileExists = fs.existsSync(logPath);

  if (!fileExists) {
    fs.writeFileSync(logPath, headers + inputRow + outputRow);
  } else {
    fs.appendFileSync(logPath, inputRow + outputRow);
  }
}

// {
//   id: sharedID,
//   ioType: 'input',
//   rollingCos: inputSimilarity.rollingCos,
//   trainingCos: inputSimilarity.trainingCos,
// idealCos: inputSimilarity.idealCos, //? Stretch Goal
// }
