import fs from 'fs';
import path from 'path';
import chalk from 'chalk';
import Table from 'cli-table3';
import { MODELS, IO_TYPES, BASELINE_TYPES, OUTPUT_DIR } from '../tkyoDrift.js';

// Constants & CLI Args;
const logPath = path.join(OUTPUT_DIR, 'drift_log.csv');
const args = process.argv.slice(2);
const days = isNaN(parseInt(args[0])) ? 1 : parseInt(args[0]);
const driftThreshold = 0.9;
const startTime = Date.now() - days * 86400000;

// Guard: Check File Exists
if (!fs.existsSync(logPath)) {
  console.error(`No log file not found at: ${logPath}`);
  process.exit(1);
}

// Parse CSV
const [headerLine, ...dataLines] = fs
  .readFileSync(logPath, 'utf-8')
  .trim()
  .split('\n');
const headers = headerLine.split(',');
const rows = dataLines.map((line) => line.split(','));

// Identify cosine similarity columns dynamically
const COS_COLUMNS = headers.filter((h) => h.endsWith('COS'));

// Filter rows by timestamp
const filteredRows = rows.filter((row) => {
  const timestamp = new Date(row[2]).getTime(); // index 2 = TIMESTAMP
  return timestamp >= startTime;
});

// Aggregation Maps
const columnSums = {};
const rowCounts = {};
// let totalViolations = 0;

// Sum up values for averaging
for (const row of filteredRows) {
  const ioType = row[3];

  for (const col of COS_COLUMNS) {
    const key = `${ioType}|${col}`;
    const index = headers.indexOf(col);
    const val = parseFloat(row[index]);

    if (!isNaN(val)) {
      columnSums[key] = (columnSums[key] || 0) + val;
      rowCounts[key] = (rowCounts[key] || 0) + 1;
    }
  }
}

// Helpers
const colorizeSimilarity = (val) => {
  if (val >= 0.9) return chalk.green(val.toFixed(2));
  if (val >= 0.77) return chalk.yellow(val.toFixed(2));
  return chalk.red(val.toFixed(2));
};

// Table Setup
const table = new Table({
  head: [
    chalk.bold.white('I/O Type'),
    chalk.bold.white('Drift Type'),
    chalk.bold.white('Baseline'),
    chalk.bold.white('Avg COS'),
    chalk.bold.white('Violation Count'),
  ],
});

// Build Table Rows
for (const ioType of IO_TYPES) {
  for (const [modelType] of Object.entries(MODELS)) {
    for (const baselineType of BASELINE_TYPES) {
      const columnHeader = `${modelType.toUpperCase()} ${baselineType.toUpperCase()} COS`;
      const colIndex = headers.indexOf(columnHeader);
      if (colIndex === -1) continue;

      const key = `${ioType}|${columnHeader}`;
      const sum = columnSums[key];
      const count = rowCounts[key];

      if (!sum || !count) continue;

      const avg = sum / count;

      // Count violations for this group
      let groupViolations = 0;
      let groupTotal = 0;

      for (const row of filteredRows) {
        if (row[3] !== ioType) continue;

        const val = parseFloat(row[colIndex]);
        if (!isNaN(val)) {
          groupTotal++;
          if (val < driftThreshold) groupViolations++;
        }
      }

      const percentValue =
        groupTotal > 0 ? Math.round((groupViolations / groupTotal) * 100) : 0;

      let coloredCount;
      if (percentValue <= 5) {
        coloredCount = chalk.green(`${groupViolations} (${percentValue}%)`);
      } else if (percentValue <= 10) {
        coloredCount = chalk.yellow(`${groupViolations} (${percentValue}%)`);
      } else {
        coloredCount = chalk.red(`${groupViolations} (${percentValue}%)`);
      }

      table.push([
        ioType.toUpperCase(),
        modelType.toUpperCase(),
        baselineType.toUpperCase(),
        colorizeSimilarity(avg),
        coloredCount,
      ]);
    }
  }
}

const titleText = `TKYO DRIFT ANALYTICS FOR PAST ${days} DAY(S)`;
const padding = 24;
const contentWidth = titleText.length + padding;
const top = '╔' + '═'.repeat(contentWidth) + '╗';
const middle = `║${' '.repeat(padding / 2)}${titleText}${' '.repeat(
  padding / 2
)}║`;
const bottom = '╚' + '═'.repeat(contentWidth) + '╝';

console.log(chalk.redBright(`${top}\n${middle}\n${bottom}`));
console.log(table.toString());
// console.log(
//   `${totalViolations} total drift events below ${driftThreshold} similarity in ${days}\n`
// );
