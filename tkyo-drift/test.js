import { spawn } from 'child_process';

const getHNSW = async () => {
  return new Promise((resolve, reject) => {
    const query = Array.from({ length: 384 }, () => Math.random() * 2 - 1);
    const pyProg = spawn('python3', [
      './pythonHNSW.py',
      // this.ioType,
      // this.modelType,
      // this.embedding,
      'input',
      'lexical',
      JSON.stringify(query),
    ]);

    let result = '';
    let error = '';

    //This function is for accepting to data from python
    //Data is the binary form of the result from python
    pyProg.stdout.on('data', (data) => {
      //result is the stringified version of the result from python
      result += data.toString();
    });

    //This function is for error handling
    //Data is the binary from of the error from python
    pyProg.stderr.on('data', (data) => {
      //error is the stringified version of the error from python
      error += data.toString();
    });

    pyProg.on('close', (code) => {
      if (code !== 0) {
        reject(new Error(`Python process failed: ${error}`));
        return;
      }
      try {
        //destructures from result after parsing the result, changing it from a string to an object
        const { labels, distances } = JSON.parse(result);
        console.log(distances);
        resolve({ labels, distances });
      } catch (e) {
        reject(new Error(`Failed to parse Python output: ${e.message}`));
      }
    });
  });
};

getHNSW();
