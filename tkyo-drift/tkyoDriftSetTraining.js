/*...:::::..................................::::...............::;::.....:...........................:;;;;++++xxxxxxxxxxxXXXXXXXXXXXXXX       
....::::::..................................::::::::....::::....:;::....::.....................:....:x;::::.:+xxxxxxxxXXXXXXXXXXXXXXXXX       
....:::::::...............................:::::::::..:::;;;;::;;xxxxx+;;;;::....................;xXXXXXx;;;;:;.:.+xxxXXXXXXXXXXXXXXXXXX       
....:::::::::.......................::::..:::::;;;;+++xxxxXXXXXXX$$$$$$$XX$Xx++;;;;:..........::+XXXXXXXXX$$X+.......xXXXXXXXXXXXXXXXXX       
....:::::::.....................::::::::::;;+++++xxxxxxXXXXXXXXXX$$$$$$$$$$$$$$XXxx;::......:.+X$XxxxXXX$XXXXXXXX;..+XXXXXXXXXXXXXXXXXX       
...:::::::..................::::::::::::;;+++xxxxxXXXXXXXXXXXXXXXXX$$$$$$$$$$$$$$$$X+;:::::;::Xx+xxxx+xxxxXX$XX+xXXXXXXXXXXXXXXXXXXXXXX       
...:::::...........::::::::::;;;;:;;;;+++++xxxxxxXXXXXX$XXXXXXXXXXXXXXX$$$$$$$$$$$$$$$Xxxx;;...++X$$xxxx+xxxX$$++XXXXXXXXXXXXXXXXXXXXXX       
:..:::............::::::::;;;;;;;;++++++xxxxxxXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX$$$$$X$$$$$$$x:x$x+x+XXXXXxX$$X++X;;+XXXX$$XXXXXXXXXXXXXXX$       
:.::::::..........:::::::::;;;++++++xxxxxXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX$XXX$$$$$$$+.X$x+x+xXXXXXXXXXXX$+;+XX$$$$XXXXXXXXXXXXXXX$$       
:.::::::::::.....::::::::::;;;;++xxxxxXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX$$$$$;.$X:...;XXXXXXXxxX+Xx;:$$$$$$$$XXXXX$$XXXXXX$$$       
.:::::::::::....::::..::::::;;;;++xxxxXXXXXXXXXXXXXXXXXXxxxxXXXXXXXXXXXXXXXXXXXXXXXXx.;.$+x+xx;+:.;+xX:xX+$&;x$&$$$$$$X$$X$$$$XXXXX$$$$       
::::::::::::........::::::;;;;++++xxxXXXXXXXXXXXXXxXXXXxxxXXXXXXXXXXXXXXXXXXXXXXXXXX$+:+X$$$Xx;;xxxx+;:..$&++&&&&$$$$$$$$$$X$XXXXX$$$$$       
::::::::::::........::::;;;;++++xxxxxXXXXXXXXXXXXXXXxxXxxxxxxXXXXXXXXXXXXXXXXXXXXXXx..X$$$$$$$$X$$Xx+;;X+&x:&&&&&$$$$$XX$$$XX$XX$$$$$$X       
:::::::::::::::::::::;;;;;+++xxxxxxXXXXXXXXXXXXXXxxXXXXxXxXxXxXXXXXXXXXXXXXXXXXXXXX++:X&&$$$$&&&&&$$$&$X$x;&&&&&&$$$$XX$X$$$$X$$$$$$$X$       
:::::::::::::::::;;;;;;++++xxxxXXXXXXXXXXXXXXXXXXXXXxxXxxXxXXXXXXXXXXXXXXXXXXXXXX+x;.++;+$&&$$$$$&&&&&$XX;::x&&&&&&&&$X$$$$$$$$$$$$$X$$       
::::::::::::;;;;;;;++++++++xxxXXXXXXXXXXXXXXXXXXXXXXXxXXXXXXXXXXXXXXXXXXXXXXXXXXX++.;;+  ..:x$&&&&$$&&$;:X&&&&&&$$$&$$$$$$$$$$$$$$$X$$$       
::::::::::::;;;;;;;;+++++++xxxXXXXXXXXXXXXXXXXXXXXXXXxXXxXXxxXXXXXXXXXXXXXXXXXXx+:.:++. ..    ....:X++.:+&&&&&&$$$$XX$$$$$X$$X$$$$$XX$$       
:::::::;;;;;;;;;;;;+++++++xxXXXXXX$$XXXXXXXXXXXxXXXxxxXXxXXXXXXXXXXXXXXXXXXXXXX++..;::;;:.  ...  ;++:.;X&&&&&&$$$$X$$$$$$$$$$$$$$$$$$$$       
::;;;:;;;;;;;;;;;;;+++++xxxxXXXXXXX$XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXxxx$+;+..:....+:..x;+.:xX&&&&&&$$$$$X$$$$$$$$$$$$$$$$$$$$       
:;;;;;;;;;;++++++++++++xxxXXXXXXXXX$XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX$$x+;++:..:....::::++:.+XX&&&&&&$$$$$X$$$$$$$$$$$$$$$$$$$$$
;;;;;;;;;;++++++++++++xxxxXXXXXXXXX$X$XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX$$$XXX;.+;:.::::;;;+..;xX&&&&&&$$$$$$$$$$$$$$$$$$$$$$$$$$$$       
;;;;;;;+++++++++xxxxxxxxxxxXXXXXXX$X$XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX$&$$XXXX+:++;..:+XX$$XX&&&&&&$$$$$X$$$$$$$X$$$$$$$$$$$$$$$       
;;;++++++++;;+++xxx++++++xxxxXXXXXX$XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX$&&&$$$XXXXXxxxx;;:X$&&&&&&&&&&$$$$$$$$$$$$$$$$$$$$$$$$$$      
;;;+++++++;;;;;++++++;;;+xxxXXXXXXXX$XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX$&&&&&&&&$$$$X$X$X$&&&&&&&&$$$$XX$$$$$$$$$$$$$$$$$$$$$$*/
import fs from 'fs';
import path from 'path';
import duckdb from 'duckdb';
import { pipeline } from '@xenova/transformers';
import { MODELS, OUTPUT_DIR } from './tkyoDrift.js';

// ------------- << Main Training Function >> -------------
export default async function tkyoDriftSetTrainings(dataArray) {
  const inputTexts = dataArray.map((d) => d.problem);
  const outputTexts = dataArray.map((d) => d.solution);

  // ------------- << Load All Models >> -------------
  // * Load each transformer pipeline once for all drift types
  const loadedModels = await Promise.all(
    Object.entries(MODELS).map(([name, modelID]) =>
      pipeline('feature-extraction', modelID).then((model) => [name, model])
    )
  );
  const models = Object.fromEntries(loadedModels);

  // ------------- << Prepare Output Directory >> -------------
  // * Ensure the data directory exists before writing files
  if (!fs.existsSync(OUTPUT_DIR)) {
    fs.mkdirSync(OUTPUT_DIR, { recursive: true });
  }

  // ------------- << Clear Old Training Files >> -------------
  // * Wipe out previous training bins to avoid contamination
  for (const driftType of Object.keys(models)) {
    for (const ioType of ['input', 'output']) {
      const filePath = path.join(
        OUTPUT_DIR,
        `${driftType}.${ioType}.training.bin`
      );

      if (fs.existsSync(filePath)) {
        fs.unlinkSync(filePath);
        console.log(`🧹 Cleared old file: ${filePath}`);
      }
    }
  }

  // ------------- << Helper: Embed & Write Vectors >> -------------
  // * Run batching to reduce memory usage and avoid timeouts
  const embedAndSave = async (ioType, texts, chunkSize = 100) => {
    // For each model, embed and write the entire set of vectors
    for (const [driftType, model] of Object.entries(models)) {
      // Construct the destination path and open a write stream
      const trainingPath = path.join(
        OUTPUT_DIR,
        `${driftType}.${ioType}.training.bin`
      );
      const writeStream = fs.createWriteStream(trainingPath, { flags: 'a' });

      let totalWritten = 0;

      // Chunks the dataset into smaller groups (100 at a time)
      for (let i = 0; i < texts.length; i += chunkSize) {
        const chunk = texts.slice(i, i + chunkSize);

        // Convert each text into an embedding buffer
        for (const text of chunk) {
          const result = await model(text, {
            pooling: 'mean',
            normalize: true,
          });

          // Create a Float32Array from the embedding data
          const float32Array = new Float32Array(result.data);

          // Write the buffer directly to the file stream
          writeStream.write(Buffer.from(float32Array.buffer));

          totalWritten++;
        }

        // Log progress
        console.log(
          `📦 [${totalWritten}/${texts.length}] ${driftType} ${ioType}`
        );
      }

      // Close the write stream after processing all vectors
      writeStream.end();
    }
  };

  // ------------- << Run Training Embeddings >> -------------
  await embedAndSave('input', inputTexts);
  await embedAndSave('output', outputTexts);
  console.log('✅ Training embeddings saved successfully.');
}

// ------------- << Flexible File Loader: JSON or Parquet >> -------------
async function loadDataset(filePath) {
  if (filePath.endsWith('.json')) {
    const rawData = fs.readFileSync(filePath, 'utf-8');
    return JSON.parse(rawData);
  }

  if (filePath.endsWith('.parquet')) {
    const db = new duckdb.Database(':memory:');
    const connection = db.connect();

    return new Promise((resolve, reject) => {
      connection.all(
        `SELECT * FROM read_parquet('${filePath}')`,
        (err, rows) => {
          if (err) reject(err);
          else resolve(rows);
        }
      );
    });
  }

  throw new Error(`Unsupported file format: ${filePath}`);
}

// ------------- << Load Dataset & Kickoff Training >> -------------
const datasetPath = process.argv[2] || path.resolve('../smallTraining.parquet');

loadDataset(datasetPath)
  .then((dataArray) => tkyoDriftSetTrainings(dataArray))
  .catch((err) => console.error('❌ Failed to load dataset:', err));
