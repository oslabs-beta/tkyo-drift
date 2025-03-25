// * Calculate the Cosine Similarity between the baseline value and and an input
export default function getSimilarity(embeddingObject, baselineObject) {
  // console.log(embeddingObject)
  // console.log(baselineObject)
  const outputObject = {}
  for (const driftType in embeddingObject) {
    // Destructure the object to make the code more better
    const { modelOutput } = embeddingObject[driftType];
    for (const keys in baselineObject) {
      if (driftType[0] === keys[0]) {
      const a = modelOutput
        const b = baselineObject[keys]
        const arrayOfComparisonVectors = [a, b];
        const reducedArray = [];
        for (let i = 0; i < arrayOfComparisonVectors[0].length; i++) {
          reducedArray[i] = arrayOfComparisonVectors.reduce(
            (accumulator, currentValue) => accumulator * currentValue[i],
            1
          );
        }
        const dotProduct = reducedArray.reduce(
          (accumulator, currentValue) => accumulator + currentValue,
          0
        );
        const magnitudeA = Math.sqrt(a.reduce((sum, a) => sum + a * a, 0));
        const magnitudeB = Math.sqrt(b.reduce((sum, b) => sum + b * b, 0));

        outputObject[keys] = dotProduct / (magnitudeA * magnitudeB);
      }
    }
  }
  console.log(outputObject)
  return outputObject
}
//   }
//   // Iterate through the object to do the stuff every time for all the things
//   for (const keys in )

//   const arrayOfComparisonVectors = [a, b];
//   const reducedArray = [];
//   for (let i = 0; i < arrayOfComparisonVectors[0].length; i++) {
//     reducedArray[i] = arrayOfComparisonVectors.reduce(
//       (accumulator, currentValue) => accumulator * currentValue[i],
//       1
//     );
//   }
//   const dotProduct = reducedArray.reduce(
//     (accumulator, currentValue) => accumulator + currentValue,
//     0
//   );
//   const magnitudeA = Math.sqrt(a.reduce((sum, a) => sum + a * a, 0));
//   const magnitudeB = Math.sqrt(b.reduce((sum, b) => sum + b * b, 0));
//   console.log(magnitudeA);
//   console.log(magnitudeB);
//   return dotProduct / (magnitudeA * magnitudeB);
// }
