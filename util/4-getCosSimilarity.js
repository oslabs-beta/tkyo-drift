// * Calculate the Cosine Similarity between the baseline value and and an input
export default function getSimilarity(a, b) {
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
  console.log(magnitudeA);
  console.log(magnitudeB);
  return dotProduct / (magnitudeA * magnitudeB);
}
