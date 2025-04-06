import fs from 'fs';
import path from 'path';
import chalk from 'chalk';
import Table from 'cli-table3';
import { MODELS, IO_TYPES, OUTPUT_DIR } from '../tkyoDrift.js';
import { loadScalarMetrics } from './loadScalarMetrics.js';
import { compareScalarDistributions } from './compareScalarDistributions.js';

// TODO: This is 1 step short of actually doing PSI math, which should probably be it's own util function 
// Container to hold all metrics for each model/io combo
const driftByMetric = {};

for (const ioType of IO_TYPES) {
  for (const [modelType] of Object.entries(MODELS)) {
    const baseName = `${modelType}.${ioType}`;
    const trainingPath = path.join(OUTPUT_DIR, 'scalars', `${baseName}.training.scalar.jsonl`);
    const rollingPath = path.join(OUTPUT_DIR, 'scalars', `${baseName}.rolling.scalar.jsonl`);

    // Skip if either file doesn't exist
    if (!fs.existsSync(trainingPath) || !fs.existsSync(rollingPath)) continue;

    const training = await loadScalarMetrics(trainingPath);
    const rolling = await loadScalarMetrics(rollingPath);
    const drift = compareScalarDistributions(training, rolling);

    // Push each metric's comparison results into grouped object
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

//  Render Each Metric Table
for (const [metric, rows] of Object.entries(driftByMetric)) {
  // Build CLI-style table for each metric
  const table = new Table({
    head: [
      chalk.bold.white('I/O Type'),
      chalk.bold.white('Drift Type'),
      chalk.bold.white(`Mean Delta`),
      chalk.bold.white(`Std Delta`),
    ],
  });

  for (const row of rows) {
    table.push([
      row.ioType.toUpperCase(),
      row.modelType.toUpperCase(),
      formatDelta(row.meanDelta),
      formatDelta(row.stdDelta),
    ]);
  }

  // CLI-style title block for each metric section
  const title = `'${metric}' drift in Rolling vs. Training`;
  const pad = 12;
  const width = title.length + pad;
  const top = '╔' + '═'.repeat(width) + '╗';
  const middle = `║${' '.repeat(pad / 2)}${title}${' '.repeat(pad / 2)}║`;
  const bottom = '╚' + '═'.repeat(width) + '╝';

  console.log(chalk.cyanBright(`\n${top}\n${middle}\n${bottom}`));
  console.log(table.toString());
}

//  Format Delta with fancy colors
function formatDelta(val) {
  if (typeof val !== 'number') return chalk.gray('n/a');
  const formatted = val.toFixed(2);
  if (Math.abs(val) < 0.1) return chalk.green(formatted);
  if (Math.abs(val) < 0.5) return chalk.yellow(formatted);
  return chalk.red(formatted);
}

//  this whole section below is to calculate how many training vs rolling we are comparing
let trainingCount = 0;
let rollingCount = 0;

for (const ioType of IO_TYPES) {
  for (const [modelType] of Object.entries(MODELS)) {
    const baseName = `${modelType}.${ioType}`;
    const trainingPath = path.join(OUTPUT_DIR, 'scalars', `${baseName}.training.scalar.jsonl`);
    const rollingPath = path.join(OUTPUT_DIR, 'scalars', `${baseName}.rolling.scalar.jsonl`);

    if (fs.existsSync(trainingPath)) {
      const lines = fs.readFileSync(trainingPath, 'utf-8').split('\n').filter(Boolean);
      trainingCount += lines.length;
    }

    if (fs.existsSync(rollingPath)) {
      const lines = fs.readFileSync(rollingPath, 'utf-8').split('\n').filter(Boolean);
      rollingCount += lines.length;
    }
  }
}

const modelCombos = Object.keys(MODELS).length *2; // times 2 because input and output
const adjustedTrainingCount = Math.floor(trainingCount / modelCombos);
const adjustedRollingCount = Math.floor(rollingCount / modelCombos);
const footer = `Samples — Training: ${adjustedTrainingCount.toLocaleString()}   |   Rolling: ${adjustedRollingCount.toLocaleString()}`;

console.log(chalk.gray(`\n${footer}\n`));
