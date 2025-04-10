import fs from 'fs';
import path from 'path';
import chalk from 'chalk';
import Table from 'cli-table3';
import { MODELS, IO_TYPES, OUTPUT_DIR } from '../tkyoDrift.js';
import { loadScalarMetrics } from './loadScalarMetrics.js';
import { compareScalarDistributions } from './compareScalarDistributions.js';

// Create an object to hold the drift results, grouped by each metric name
const driftByMetric = {};

// Loop through every input/output type
for (const ioType of IO_TYPES) {
  // Loop through every model type (semantic, lexical, concept, etc.)
  for (const [modelType] of Object.entries(MODELS)) {
    // Build the base file name for scalar logs
    const baseName = `${modelType}.${ioType}`;
    const trainingPath = path.join(OUTPUT_DIR, 'scalars', `${baseName}.training.scalar.jsonl`);
    const rollingPath = path.join(OUTPUT_DIR, 'scalars', `${baseName}.rolling.scalar.jsonl`);

    // Skip this model/io combination if either file doesn't exist
    if (!fs.existsSync(trainingPath) || !fs.existsSync(rollingPath)) continue;

    // Load scalar values from both training and rolling baseline
    const training = await loadScalarMetrics(trainingPath);
    const rolling = await loadScalarMetrics(rollingPath);

    // Compare the distributions for each metric to compute deltas
    const drift = compareScalarDistributions(training, rolling);

    // For every metric (e.g., norm, entropy, etc.), store the deltas in driftByMetric
    for (const [metric, values] of Object.entries(drift)) {
      if (!driftByMetric[metric]) driftByMetric[metric] = [];
      driftByMetric[metric].push({
        ioType,
        modelType,
        meanDelta: values.meanDelta,
        stdDelta: values.stdDelta,
      });
    }
  }
}

// Loop over each unique scalar metric (e.g., norm, entropy, etc.)
for (const [metric, rows] of Object.entries(driftByMetric)) {
  // Create a new table for this metric
  const table = new Table({
    head: [
      chalk.bold.white('I/O Type'),
      chalk.bold.white('Drift Type'),
      chalk.bold.white(`Mean Delta`),
      chalk.bold.white(`Std Delta`),
    ],
  });

  // Add one row per model/io combo to the table
  for (const row of rows) {
    table.push([
      row.ioType.toUpperCase(),
      row.modelType.toUpperCase(),
      formatDelta(row.meanDelta),
      formatDelta(row.stdDelta),
    ]);
  }

  // Generate a fancy boxed title for this section
  const title = `'${metric}' drift in Rolling vs. Training`;
  const pad = 12;
  const width = title.length + pad;
  const top = '╔' + '═'.repeat(width) + '╗';
  const middle = `║${' '.repeat(pad / 2)}${title}${' '.repeat(pad / 2)}║`;
  const bottom = '╚' + '═'.repeat(width) + '╝';

  // Print the section header and table
  console.log(chalk.cyanBright(`\n${top}\n${middle}\n${bottom}`));
  console.log(table.toString());
}

// Takes a delta value and adds color coding depending on severity
function formatDelta(val) {
  if (typeof val !== 'number') return chalk.gray('n/a');
  const formatted = val.toFixed(2);
  if (Math.abs(val) < 0.1) return chalk.green(formatted);
  if (Math.abs(val) < 0.5) return chalk.yellow(formatted);
  return chalk.red(formatted);
}

// Count how many unique samples we are comparing (normalized)
let trainingCount = 0;
let rollingCount = 0;
let trainingCombos = 0;
let rollingCombos = 0;

// Count total lines across all scalar files, and how many file combos exist
for (const ioType of IO_TYPES) {
  for (const [modelType] of Object.entries(MODELS)) {
    const baseName = `${modelType}.${ioType}`;
    const trainingPath = path.join(OUTPUT_DIR, 'scalars', `${baseName}.training.scalar.jsonl`);
    const rollingPath = path.join(OUTPUT_DIR, 'scalars', `${baseName}.rolling.scalar.jsonl`);

    if (fs.existsSync(trainingPath)) {
      const lines = fs.readFileSync(trainingPath, 'utf-8').split('\n').filter(Boolean);
      trainingCount += lines.length;
      trainingCombos++;
    }

    if (fs.existsSync(rollingPath)) {
      const lines = fs.readFileSync(rollingPath, 'utf-8').split('\n').filter(Boolean);
      rollingCount += lines.length;
      rollingCombos++;
    }
  }
}

// Normalize the totals by dividing by number of model/io combinations
const adjustedTrainingCount = trainingCombos ? Math.floor(trainingCount / trainingCombos) : 0;
const adjustedRollingCount = rollingCombos ? Math.floor(rollingCount / rollingCombos) : 0;
const footer = `Samples — Training: ${adjustedTrainingCount.toLocaleString()}   |   Rolling: ${adjustedRollingCount.toLocaleString()}`;

// Display total sample counts for transparency
console.log(chalk.gray(`\n${footer}\n`));
