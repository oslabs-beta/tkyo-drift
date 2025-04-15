import { tkyoDrift, tkyoDriftSetTrainingHook } from 'tkyodrift';

const dataSetPath = '../data';
const inputName = 'problem';
const outputName = 'solution';
tkyoDriftSetTrainingHook(dataSetPath,inputName,outputName)

const input =
  'Describe how the context surrounding the shape of a vector determines how much drift might occur when analyzed using cosine similarity.';
const output = 'I am sorry, but I do know know how to respond to this request.';
tkyoDrift(input, output);
