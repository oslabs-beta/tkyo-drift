// * Calculate the Reference Model's Input/Output Baseline Value
export default function getBaseline(files, data) {
  // Make a results object to hold the data
  const resultsObject = {};

  // Iterate through each file to get the vectors
  for (const keys in files) {

    // Get the array of arrays of vectors from each file
    const vectors = files[keys];

    // Initialize and empty array to house the Mean Value of the input arrays
    const result = [];

    // Iterate through each array, and get the average for the set
    for (let i = 0; i < vectors[0].length; i++) {
      result[i] =
        vectors.reduce(
          (accumulator, currentValue) => accumulator + currentValue[i],
          0
        ) / vectors.length;
    }

    // Push the result array into the results object
    resultsObject[keys] = result;
  }

  // Return the results
  return resultsObject;
}

// * After spending a few hours with GPT and reading, here is some pseudocode for an HNSW NNS

/*
?So how does this work?

*Think of it like this:
You already have a large set of embedded vectors (e.g., from your training data).
You build an HNSW index from these vectors — which lets you efficiently search for the most similar ones to any new query vector.
Now you get a new input/output vector that you want to compare.
You ask the HNSW index: "Hey, what are the N closest vectors to this one?"
It returns the k most similar vectors — based on cosine similarity — without you having to manually calculate anything.

*At this point, you now:
Have a small subset of vectors from the original data.
Can compute their average (centroid) to produce a contextual baseline vector.
Then you compute the cosine similarity between the new input vector and that baseline vector.

!Recap of the Flow:
New input/output vector arrives
You query the HNSW index: "Give me the k-nearest neighbors to this vector"
HNSW internally uses cosine similarity (or other metric) to find those neighbors
You average those neighbors → get a baseline vector
You compare the input/output vector to this baseline using cosine similarity
If similarity is low → possible semantic drift
*/

// *Define constants:
// Set the dimension size to match your embedding model (e.g., 384)
// ? We can get this by loading the first vector and setting dimension to it's length.
// Set the number of neighbors to use for baseline calculation (e.g., K = 50)

// * in all of these cases except #1, the K represents the number of vectors with highest cosine similarity

// TODO: Pick one or more methods for getting K
// * After reading up a bit, we have some options:
// 1. We could compute the mean of K where K is the entire data set //!(NO HNSW Required)
// 2. We could compute the mean of K nearest Neighbors //!(K is a fixed number)
// 3. We could compute the mean of several K values simultaneously, and then "evaluate stability" (10, 25, 50, 100)  //!See Excalidraw (near chart) for explanation
// 4. We could compute the mean of K where K is a % of total vectors (5%, 10%)
// 5. We could compute the mean of K where K is the square root of the total data size (223 of 50k, 30 for 1000)
// 6. We could compute the mean of K where K is determined by a clustering algorithm (like k-means) //!Requires offline computation step (ie, nightly CronJob)

// ? if we are checking cosine similarities, then don't we need to know what K is before we
// NO. Why? Because the HNSW index is built specifically to find the nearest neighbors for a given input vector, using cosine similarity (or another distance metric)
// This means that THIS file needs to import 'getCosSimilarity' it as a dependency, since the HNSW will need to invoke it 100s or 1000s of times.

// TODO: Do we want to make these are separate utils or have them embedded (heh) here?
// *Define function: loadEmbeddings(filePath)
// Read the contents of the embedding file from the provided file path
// Parse the contents as a JSON array of vectors
// Return the array of vectors

// *Define function: buildIndex(embeddings)
// Create a new HNSW index using cosine similarity and the given vector dimension //!Ive installed the hnswlib-node NPM package
// Initialize the index with the total number of embeddings
// For each embedding in the array:
// Add the embedding to the index with its corresponding ID
// Return the index

// *Define function: computeBaselineVector(index, queryVector)
// Search the index for the nearest N neighbors of the query vector
// For each neighbor ID returned:
// Retrieve the vector from the index using the ID
// Initialize an empty vector of the same dimension with all values set to zero
// For each retrieved vector, add its values to the corresponding positions in the empty vector
// Divide each element in the resulting vector by the number of neighbors to get the average
// Return the average vector as the baseline

//? Can HNSW indices be saved to a disk so it doesn't have to be rebuilt (for training/ideal data at least?)
/*
*After building your index:
Call index.writeIndex(filePath)
This writes the entire HNSW graph and its internal structure to a binary file
This is very fast and makes your system efficient because you only build the index once (e.g., nightly or at deployment), and load it later for real-time querying.

*Later on, when you want to use it (e.g., in getBaseline.js):
Create a new HierarchicalNSW object using the same space (e.g. "cosine")
Call index.readIndex(filePath, maxElements)
maxElements is the number of elements the index will contain — should match or exceed what you originally built
After that, the index is ready for use — no need to re-add points manually.
*/
