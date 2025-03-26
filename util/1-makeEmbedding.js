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
import { MODELS } from '../tkyoDrift.js';

// * Convert the I/O to a Vector using an Embedding Model
export default async function makeEmbedding(text) {
  // Create an empty object for the output
  const output = {};

  try {
    // Load the embedding models
    const loadedModel = await Promise.all(
      Object.entries(MODELS).map(([name, modelID]) =>
        pipeline('feature-extraction', modelID).then((model) => [name, model])
      )
    );

    // Assemble results (in general the model will have the .tolist method)
    for (const [name, model] of loadedModel) {
      const result = await model(text, { pooling: 'mean', normalize: true });
      output[name] = {
        modelOutput: result.data,
        byteOffset: result.data.byteOffset,
        dims: result.data.length
      }
    }

    // Return the output
    return output;
  } catch (error) {
    console.error('❌ Error generating embedding:', error.message);
    return null;
  }
}
