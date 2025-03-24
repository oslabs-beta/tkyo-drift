// EXPECTED INPUTS/OUTPUT
// This file is meant to take in text data from the Existing Workflow and spit out a vector.
// ? The input for this function is meant to be un-nested TEXT DATA.
// ? The output for this function is an object that contains model type and vector as an object.
// If this step is failing, it is likely because the input data is not text.

// TODO: We are currently loading the model on EVERY call.
// If we want to have the models loaded locally, we need to have the
// user download the models themselves.
// *Notably, this is only a problem at scale. Testing with loaders is really quick on my end -Tico
import { pipeline } from '@xenova/transformers';
import { MODELS, PRECISION_VALUE } from '../tkyoDrift.js';

// * Convert the I/O to a Vector using an Embedding Model
export default async function makeEmbedding(text) {
  try {
    // Check the data type of the input
    if (typeof text !== 'string' || text.trim() === '') {
      console.error(
        '↗️ Embedding failed: Data sent to embedding model is not a non-empty string.'
      );
      return null;
    }

    // Precision Control helper function (to remove extra decimals)
    function roundVector(vector, PRECISION_VALUE) {
      return vector.map((x) => Number(x.toFixed(PRECISION_VALUE)));
    }

    // Load the embedding models
    const loadedModel = await Promise.all(
      Object.entries(MODELS).map(([name, modelID]) =>
        pipeline('feature-extraction', modelID).then((model) => [name, model])
      )
    );

    // Get the model names from the pipeline object
    const modelMap = Object.fromEntries(loadedModel);

    // Create an empty object to house the results
    const output = {};

    // Assemble results (in general the model will have the .tolist method)
    for (const [name, model] of Object.entries(modelMap)) {
      const result = await model(text, { pooling: 'mean', normalize: true });
      output[name] = roundVector(result.tolist()[0][0], PRECISION_VALUE);
    }

    // Return the output
    return output;
  } catch (error) {
    console.error('❌ Error generating embedding:', error.message);
    return null;
  }
}
