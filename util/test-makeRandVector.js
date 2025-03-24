// TODO: Remove this shit. This should not be part of the final project.
// * Make test vectors to demonstrate how to get drift values
export function makeVector(num) {
  const arrayOfVectors = [];
  for (let i = 0; i < num; i++) {
    let counter = 0;
    const vector = [];
    while (counter < 10) {
      vector.push(Math.random() * 2 - 1);
      counter++;
    }
    arrayOfVectors.push(vector);
  }

  return arrayOfVectors;
}
