import { RequestHandler } from 'express';
import { OpenAI } from 'openai/index';
import dotenv from 'dotenv';
import { ServerError } from '../types';
import tkyoDrift from '../../../../tkyoDrift.js';

dotenv.config(); // Load environment variables

// Initialize OpenAI client
// const openai = new OpenAI();
const openai = new OpenAI({
  apiKey: process.env.OPENAI_API_KEY,
});

export const queryOpenAI: RequestHandler = async (_req, res, next) => {
  const { userInputQuery } = res.locals;
  console.log(userInputQuery);
  const prompt = `You are an expert postgreSQL generator for a Star Wars database. The database has the following tables:
  TABLE: people
  _id (BIGINT, PRIMARY KEY)
  name (VARCHAR)
  mass (VARCHAR)
  hair_color (VARCHAR)
  skin_color (VARCHAR)
  eye_color (VARCHAR)
  gender (VARCHAR)
  birth_year (VARCHAR)
  species_id (BIGINT, FOREIGN KEY -> species._id)
  homeworld_id (BIGINT, FOREIGN KEY -> planets._id)
  height (INTEGER)
  TABLE: species
  _id (BIGINT, PRIMARY KEY)
  name (VARCHAR)
  classification (VARCHAR)
  average_height (VARCHAR)
  average_lifespan (VARCHAR)
  hair_colors (VARCHAR)
  skin_colors (VARCHAR)
  eye_colors (VARCHAR)
  language (VARCHAR)
  homeworld_id (BIGINT, FOREIGN KEY -> planets._id)
  TABLE: planets
  _id (BIGINT, PRIMARY KEY)
  name (VARCHAR)
  rotation_period (INTEGER)
  orbital_period (INTEGER)
  diameter (INTEGER)
  climate (VARCHAR)
  gravity (VARCHAR)
  terrain (VARCHAR)
  surface_water (VARCHAR)
  population (BIGINT)
  TABLE: films
  _id (BIGINT, PRIMARY KEY)
  title (VARCHAR)
  episode_id (INTEGER)
  opening_crawl (VARCHAR)
  director (VARCHAR)
  producer (VARCHAR)
  release_date (DATE)
  TABLE: people_in_films
  _id (BIGINT, PRIMARY KEY)
  person_id (BIGINT, FOREIGN KEY -> people._id)
  film_id (BIGINT, FOREIGN KEY -> films._id)
  TABLE: species_in_films
  _id (BIGINT, PRIMARY KEY)
  species_id (BIGINT, FOREIGN KEY -> species._id)
  film_id (BIGINT, FOREIGN KEY -> films._id)
  TABLE: planets_in_films
  _id (BIGINT, PRIMARY KEY)
  planet_id (BIGINT, FOREIGN KEY -> planets._id)
  film_id (BIGINT, FOREIGN KEY -> films._id)
  TABLE: vessels
  _id (BIGINT, PRIMARY KEY)
  name (VARCHAR)
  manufacturer (VARCHAR)
  model (VARCHAR)
  vessel_type (VARCHAR)
  vessel_class (VARCHAR)
  cost_in_credits (BIGINT)
  length (VARCHAR)
  max_atmosphering_speed (VARCHAR)
  crew (INTEGER)
  passengers (INTEGER)
  cargo_capacity (VARCHAR)
  consumables (VARCHAR)
  TABLE: vessels_in_films
  _id (BIGINT, PRIMARY KEY)
  vessel_id (BIGINT, FOREIGN KEY -> vessels._id)
  film_id (BIGINT, FOREIGN KEY -> films._id)
  TABLE: pilots
  _id (BIGINT, PRIMARY KEY)
  person_id (BIGINT, FOREIGN KEY -> people._id)
  vessel_id (BIGINT, FOREIGN KEY -> vessels._id)
  TABLE: starship_specs
  _id (BIGINT, PRIMARY KEY)
  vessel_id (BIGINT, FOREIGN KEY -> vessels._id)
  hyperdrive_rating (VARCHAR)
  MGLT (VARCHAR)
    Given the database information and structure above, generate a SQL query based on the the request below.
    You are going to receive a series of requests that will be dependent on one another. The answer for the first question will inform the second request, and so on. For example, if the first request yields a planet, we will include it in the second request. Please make note of this in your subsequent queries.
    All valid queries have to start with SELECT
    If you can not provide a valid response, return with an empty string
    When making a SQL query, you should prefer using "=" over the LIKE operator.
    when looking at numbers, make sure to account for the appropriate context, example, 6 times higher would have to involve multiplication
    Make sure to read each word of the request, and check if they apply to any of the tables. If they do, add them to the response query e.g. woman should refer to gender in the table.
    You may be provided the previous answer. The previous answer will be relevant to your query, but make sure to read the entire question carefully. Just because a name was given, it doesn't mean you should search for that.
    Verify if the response contains markdown, and if it does, remove it from the response. An example of markdown includes \`\`\`.
    Request: ${userInputQuery}
    SQL Query:`;
  if (!userInputQuery) {
    const error: ServerError = {
      log: 'queryOpenAI did not receive a user query',
      status: 500,
      message: { err: 'An error occurred before querying OpenAI' },
    };
    return next(error);
  }

  try {
    const response = await openai.chat.completions.create({
      model: 'gpt-4o',
      messages: [{ role: 'user', content: prompt }], // create a const with prompt to hold the content, previous content: naturalLanguageQuery
      temperature: 0,
    });
    // Store the generated output message in res.locals for next middleware
    res.locals.userOutput =
      response.choices[0]?.message?.content || 'No response';
    //console.time("before and after tkyoDrift")
    tkyoDrift(res.locals.userInputQuery, res.locals.userOutput);
    //console.timeEnd("before and after tkyoDrift")
    return next();
  } catch (err) {
    const error: ServerError = {
      log: `OpenAI request failed: ${(err as Error).message}`,
      status: 500,
      message: { err: 'An error occurred while querying OpenAI' },
    };
    return next(error);
  }
};
