import fs from 'fs';
import path from 'path';
import chalk from 'chalk';
import Table from 'cli-table3';
import { compareScalarDistributions } from './compareScalarDistributions.js';
import { loadScalarMetrics } from './loadScalarMetrics.js';

// Define the path to where scalar .jsonl files are stored
const SCALAR_DIR = path.join('data', 'scalars');
console.log(chalk.gray(`\n📂 Scanning scalar directory: ${SCALAR_DIR}\n`));

// Load all filenames in the scalar directory
const files = fs.readdirSync(SCALAR_DIR);

// Regex pattern to extract metadata from filenames:
// Format: ioType.metric.[modelType?].baseline.scalar.jsonl
const scalarFileRegex = /^([a-z]+)\.([a-zA-Z]+)(?:\.([a-zA-Z]+))?\.(training|rolling)\.scalar\.jsonl$/;

// Container to group scalar file pairs (training + rolling) by metric/io/model
const matchedPairs = new Map();

// Step 1: Group files into training/rolling pairs by ioType + metric + modelType
for (const file of files) {
  const match = file.match(scalarFileRegex);

  if (!match) {
    console.warn(chalk.yellow(`⚠️ Skipping unrecognized file: ${file}`));
    continue;
  }

  const [_, ioType, metric, modelTypeRaw, baselineType] = match;
  const modelType = modelTypeRaw || 'shared'; // Shared metrics have no modelType
  const key = `${ioType}.${modelType}`; // Group by I/O and model type

  // Create group key if it doesn't exist
  if (!matchedPairs.has(key)) matchedPairs.set(key, {});

  // Inside that group, nest by metric
  if (!matchedPairs.get(key)[metric]) matchedPairs.get(key)[metric] = {};

  // Store file metadata
  matchedPairs.get(key)[metric][baselineType] = {
    file,
    metric,
    ioType,
    modelType,
  };
}

// Step 2: Print a single banner at the top of the CLI
const banner = `SCALAR METRIC DRIFT: ROLLING vs TRAINING`;
const pad = 12;
const width = banner.length + pad;
const top = '╔' + '═'.repeat(width) + '╗';
const middle = `║${' '.repeat(pad / 2)}${banner}${' '.repeat(pad / 2)}║`;
const bottom = '╚' + '═'.repeat(width) + '╝';
console.log(chalk.cyanBright(`\n${top}\n${middle}\n${bottom}\n`));

// Step 3: Loop through each (I/O + modelType) group
for (const [groupKey, metricsObj] of matchedPairs.entries()) {
  const [ioType, modelType] = groupKey.split('.');

  // Initialize the CLI table with fixed headers
  const table = new Table({
    head: [
      chalk.bold.white('Metric'),
      chalk.bold.white('Train μ'),    // Mean of training data
      chalk.bold.white('Roll μ'),     // Mean of rolling data
      chalk.bold.white('Δ Mean'),     // Difference in means
      chalk.bold.white('Train σ'),    // Standard deviation of training data
      chalk.bold.white('Roll σ'),     // Standard deviation of rolling data
      chalk.bold.white('Δ Std'),      // Difference in std deviation
      chalk.bold.white('PSI'),        // PSI Value (low is good)
    ],
  });

  // Step 4: For each metric in this group, calculate drift values
  for (const [metric, pair] of Object.entries(metricsObj)) {
    if (!pair.training || !pair.rolling) {
      console.log(chalk.dim(`⏭ Skipping incomplete pair: ${metric} (Do you have training data?)`));
      continue;
    }

    // Load values from each file
    const training = await loadScalarMetrics([metric], ioType, 'training', modelType === 'shared' ? null : modelType);
    const rolling = await loadScalarMetrics([metric], ioType, 'rolling', modelType === 'shared' ? null : modelType);

    // Compare statistical distributions (mean/std)
    const drift = compareScalarDistributions(training, rolling);
    if (!drift[metric]) {
      console.log(chalk.dim(`No data returned for ${metric}, skipping.`));
      continue;
    }

    // Push the computed values to the table
    table.push([
      metric,
      formatDelta(drift[metric].trainMean),
      formatDelta(drift[metric].rollMean),
      formatDelta(drift[metric].meanDelta),
      formatDelta(drift[metric].trainStd),
      formatDelta(drift[metric].rollStd),
      formatDelta(drift[metric].stdDelta),
      formatPSI(drift[metric].psi),
    ]);
  }

  // Only render tables that have valid data
  if (table.length > 0) {
    const sectionLabel = `→ ${ioType.toUpperCase()} • ${modelType.toUpperCase()} SCALAR METRIC VALUES`;
    console.log(chalk.bold.white(`\n${sectionLabel}`));
    console.log(table.toString());
  }
}

// Helper to color code delta values by severity
function formatDelta(val) {
  if (typeof val !== 'number') return chalk.gray('n/a');
  const formatted = val.toFixed(2);
  if (Math.abs(val) < 0.1) return chalk.green(formatted);  // Safe
  if (Math.abs(val) < 0.5) return chalk.yellow(formatted); // Caution
  return chalk.red(formatted);                             // Drifted
}

// Helper to color code PSI values by severity
function formatPSI(val) {
  if (typeof val !== 'number') return chalk.gray('n/a');
  const formatted = val.toFixed(3);
  if (val < 0.1) return chalk.green(formatted);    // No significant change
  if (val < 0.25) return chalk.yellow(formatted);  // Moderate change
  return chalk.red(formatted);                     // Major drift
}

// -------------<< SAMPLE COUNT FOOTER >>-------------

// Initialize per-direction counters for each baseline
let trainingCount = { input: 0, output: 0 };
let rollingCount = { input: 0, output: 0 };

// Loop through both I/O types (input/output)
for (const ioType of ['input', 'output']) {
  // Define a quick check: shared files will always follow this pattern
  const isShared = (f) =>
    f.startsWith(`${ioType}.`) &&
    f.endsWith('.scalar.jsonl') &&
    f.split('.').length === 5; // shared = io.metric.baseline.scalar.jsonl

  // Find *any one* shared scalar file for each baseline (we only need one to count lines)
  const trainingFile = files.find(
    (f) => isShared(f) && f.includes('.training.')
  );
  const rollingFile = files.find(
    (f) => isShared(f) && f.includes('.rolling.')
  );

  // Read the number of lines for the training baseline
  if (trainingFile) {
    const lines = fs
      .readFileSync(path.join(SCALAR_DIR, trainingFile), 'utf-8')
      .split('\n')
      .filter(Boolean); // Filter out empty final line
    trainingCount[ioType] = lines.length;
  }

  // Read the number of lines for the rolling baseline
  if (rollingFile) {
    const lines = fs
      .readFileSync(path.join(SCALAR_DIR, rollingFile), 'utf-8')
      .split('\n')
      .filter(Boolean);
    rollingCount[ioType] = lines.length;
  }
}

// Sum both directions (input + output) for final total count
const totalTrain = trainingCount.input + trainingCount.output;
const totalRoll = rollingCount.input + rollingCount.output;

// Display results as a final CLI footer line
console.log(
  chalk.gray(
    `\nSamples — Training: ${totalTrain.toLocaleString()}   |   Rolling: ${totalRoll.toLocaleString()}\n`
  )
);
