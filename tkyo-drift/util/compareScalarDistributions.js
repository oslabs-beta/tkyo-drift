// * Function that compares the scalar distributions between rolling and training 
export function compareScalarDistributions(trainingMetrics, rollingMetrics) {
  const result = {};

  // Get the keys of metrics that exist in both data sets
  const sharedMetrics = Object.keys(trainingMetrics).filter((key) =>
    rollingMetrics.hasOwnProperty(key)
  );

  // Iterate through each shared metric key 
  for (const key of sharedMetrics) {
    const train = trainingMetrics[key];
    const roll = rollingMetrics[key];

    // Skip if there is no data for that key
    if (!train.length || !roll.length) continue;

    // Get the mean and standard deviation from the training data
    const trainMean = mean(train);
    const trainStd = stddev(train);

    // Get the mean and standard deviation from the rolling data
    const rollMean = mean(roll);
    const rollStd = stddev(roll);

    result[key] = {
      trainMean,
      rollMean,
      meanDelta: rollMean - trainMean,
      trainStd,
      rollStd,
      stdDelta: rollStd - trainStd,
    };
  }

  return result;
}

// Helper: Mean
function mean(arr) {
  return arr.reduce((sum, val) => sum + val, 0) / arr.length;
}

// Helper: Standard Deviation
function stddev(arr) {
  const avg = mean(arr);
  const variance = arr.reduce((sum, val) => sum + (val - avg) ** 2, 0) / arr.length;
  return Math.sqrt(variance);
}
