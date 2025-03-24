import path from 'path';
import { fileURLToPath } from 'url';
import buildTrainingEmbeddings from '../tkyoDriftSetTraining.js';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

const jsonlPath = path.join(__dirname, '../aiModel/smollthoughts_data/dataset.jsonl');

buildTrainingEmbeddings(jsonlPath)
  .catch(err => console.error('❌ Error building embeddings:', err));
