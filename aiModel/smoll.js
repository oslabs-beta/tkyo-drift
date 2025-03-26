import { pipeline } from '@xenova/transformers';
import tkyoDrift from '../tkyoDrift.js';

const run = async () => {
  const input_text = 'How do you calculate the circumference of a sphere?'

  // ! This is the model that we have training data for, but it does not work in JS
  // const generator = await pipeline('text-generation', 'assskelad/smollm2-360M-sft_SmallThoughts');
  
  // Load the text-generation model
  const generator = await pipeline('text-generation', 'Xenova/distilgpt2');


  // Generate output
  const output = await generator(input_text, {
    max_new_tokens: 100,
    pad_token_id: generator.tokenizer.pad_token_id ?? generator.tokenizer.eos_token_id,
  });

  const decoded_output = output[0].generated_text;

  console.log('User Input:', input_text);
  console.log('AI Output:', decoded_output);

  // Directly call the drift analysis tool
  await tkyoDrift(input_text, decoded_output);
};

run();
