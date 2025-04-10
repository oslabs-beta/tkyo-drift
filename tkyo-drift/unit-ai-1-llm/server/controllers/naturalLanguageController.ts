import { Request, RequestHandler } from 'express';
import { ServerError } from '../types';

// Middleware to parse and validate the user query and optional filters
export const parseUserInputQuery: RequestHandler = async (
  req: Request<unknown, unknown, Record<string, unknown>>,
  res,
  next
) => {
  //console.time("full run time")
  
  if (!req.body.naturalLanguageQuery) {
    const error: ServerError = {
      log: 'User query not provided',
      status: 400,
      message: { err: 'An error occurred while parsing the user query' },
    };
    return next(error);
  }

  const { naturalLanguageQuery } = req.body;
  //const prompt = `${userInputQuery}`;

  if (typeof naturalLanguageQuery !== 'string') {
    const error: ServerError  = {
      log: 'User query is not a string',
      status: 400,
      message: { err: 'An error occurred while parsing the user query' },
    };
    return next(error);
  }

  res.locals.userInputQuery = naturalLanguageQuery;
  return next();
};
