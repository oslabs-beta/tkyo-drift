import fs from 'fs';
import path from 'path';
import tkyoDriftSetTrainings from '../tkyoDriftSetTraining.js';

// Adjust this path to be relative to where you're running the script
const datasetPath = path.resolve('./aiModel/smollthoughts_data/dataset.json');

// Read and parse the dataset
const rawData = fs.readFileSync(datasetPath, 'utf-8');
const dataArray = JSON.parse(rawData);

// Run training ingestion
tkyoDriftSetTrainings(dataArray);
