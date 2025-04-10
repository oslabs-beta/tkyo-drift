import express, { ErrorRequestHandler } from 'express';
import cors from 'cors';
import 'dotenv/config';

import { parseUserInputQuery } from './controllers/naturalLanguageController.js';
import { queryOpenAI } from './controllers/openaiController.js';
import { queryStarWarsDatabase } from './controllers/databaseController.js';

import { ServerError } from './types.js';
// import { Configuration, OpenAIApi } from 'openai';

const openAIMain = express();

openAIMain.use(cors());
openAIMain.use(express.json());

// const configuration = new Configuration({
//     apiKey: process.env.OPENAI_API_KEY,
//   });
//   const openai = new OpenAIApi(configuration);

openAIMain.post(
  '/api',
  parseUserInputQuery,
  queryOpenAI,
  queryStarWarsDatabase,
  (_req, res) => {
    res.status(200).json({
      databaseQuery: res.locals.databaseQuery,
      databaseQueryResult: res.locals.databaseQueryResult,
    });
  }
);

const errorHandler: ErrorRequestHandler = (
  err: ServerError,
  _req,
  res,
  _next
) => {
  const defaultErr: ServerError = {
    log: 'Express error handler caught unknown middleware error',
    status: 500,
    message: { err: 'An error occurred' },
  };
  const errorObj: ServerError = { ...defaultErr, ...err };
  console.log(errorObj.log);
  res.status(errorObj.status).json(errorObj.message);
};

openAIMain.use(errorHandler);

export default openAIMain;
