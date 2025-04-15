import { spawn } from 'child_process';

const tkyoDriftSetTraining = async (dataSetPath, ioType, ioTypeName) => {
  try {
    return new Promise((resolve, reject) => {
      const pyProg = spawn('python3', [
        '-u',
        './tkyoDriftSetTraining.py',
        dataSetPath,
        ioType,
        ioTypeName,
      ]);

      let result = '';
      let error = '';

      // This function is for accepting to data from python
      // Data is the binary form of the result from python
      pyProg.stdout.on('data', (data) => {
        const chunk = data.toString();
        result += chunk;

        // Print each chunk as it comes in
        process.stdout.write(chunk); // or console.log(chunk) if you prefer line-based output
      });

      // This function is for error handling
      // Data is the binary from of the error from python
      pyProg.stderr.on('data', (data) => {
        // Error is the stringified version of the error from python
        error += data.toString();
      });

      pyProg.on('close', (code) => {
        if (code !== 0) {
          console.error('Python stderr:', error);
          reject(new Error(`Failed to embed training data`));
          return;
        }
        resolve(result);
      });
    });
  } catch (error) {
    throw new Error(
      `Error in readFromBin for the ${this.modelType} ${this.ioType} ${this.baselineType} model: ${error.message}`
    );
  }
};

// TODO Remove hardcoded path, input name, & output name
const dataSetPath = './data';
// First call: embed the "problem" column as "problem"
// await tkyoDriftSetTraining(dataSetPath, 'problem', 'problem');

// // Second call: embed the "solution" column as "solution"
// await tkyoDriftSetTraining(dataSetPath, 'solution', 'solution');

await tkyoDriftSetTraining('./data', 'domain', 'domain');
await tkyoDriftSetTraining('./data', 'domain_description', 'domain_description');
await tkyoDriftSetTraining('./data', 'sql_complexity', 'sql_complexity');
await tkyoDriftSetTraining('./data', 'sql_complexity_description', 'sql_complexity_description');
await tkyoDriftSetTraining('./data', 'sql_task_type', 'sql_task_type');
await tkyoDriftSetTraining('./data', 'sql_task_type_description', 'sql_task_type_description');
await tkyoDriftSetTraining('./data', 'sql_context', 'sql_context');
await tkyoDriftSetTraining('./data', 'sql_explanation', 'sql_explanation');

await tkyoDriftSetTraining('./data', 'sql', 'sql');
await tkyoDriftSetTraining('./data', 'sql_prompt', 'sql_prompt');