import { RequestHandler } from 'express';
import pkg from 'pg'; // Import the PostgreSQL Pool
import dotenv from 'dotenv';
import { ServerError } from '../types';

const { Pool } = pkg;
dotenv.config(); // Load environment variables

// Create a connection pool
const pool = new Pool({
  connectionString: process.env.SUPABASE_DB_URL, // Your Supabase connection string
  ssl: { rejectUnauthorized: false }, // Required for secure connection
});


export const queryStarWarsDatabase: RequestHandler = async (
  _req,
  res,
  next
) => {
  const { userOutput } = res.locals;
  console.log(userOutput)
  if (!userOutput) {
    const error: ServerError = {
      log: 'Database query middleware did not receive a query',
      status: 500,
      message: { err: 'An error occurred before querying the database' },
    };
    return next(error);
  }
  try {
    // Query the database using the connection pool
    // DB will be called, and the query will be executed
    const { rows } = await pool.query(userOutput);
    console.log(rows)
    // Store the query result in res.locals
    res.locals.databaseQueryResult = rows;
    //console.timeEnd("full run time")
    return next();
  } catch (err) {
    const error: ServerError = {
      log: `Database query failed: ${(err as Error).message}`,
      status: 500,
      message: { err: 'An error occurred while querying the database' },
    };
    return next(error);
  }
};
//   res.locals.databaseQueryResult = [{ name: 'Sly Moore' }];
//   return next();
// };










// import { RequestHandler } from 'express';
// import pkg from 'pg'; // Import the PostgreSQL Pool
// import dotenv from 'dotenv';
// import { ServerError } from '../types';

// const { Pool } = pkg;
// dotenv.config(); // Load environment variables

// // Create a connection pool
// const pool = new Pool({
//   connectionString: `postgresql://postgres:allmypets1@db.ealrfckyghjcgrwbfpnj.supabase.co:5432/postgres`,// Your Supabase connection string
//   ssl: { rejectUnauthorized: false }, // Required for secure connection
// });

// export const queryStarWarsDatabase: RequestHandler = async (
//   _req,
//   res,
//   next
// ) => {
//   const { databaseQuery } = res.locals;
//   if (!databaseQuery) {
//     const error: ServerError = {
//       log: 'Database query middleware did not receive a query',
//       status: 500,
//       message: { err: 'An error occurred before querying the database' },
//     };
//     return next(error);
//   }
//   try {
//     // Query the database using the connection pool
//     // DB will be called, and the query will be executed
//     const { rows } = await pool.query(databaseQuery);

//     // Store the query result in res.locals
//     res.locals.databaseQueryResult = data.rows;
//     return next();
//   } catch (err) {
//     const error: ServerError = {
//       log: `Database query failed: ${(err as Error).message}`,
//       status: 500,
//       message: { err: 'An error occurred while querying the database' },
//     };
//     return next(error);
//   }
// };
// //   res.locals.databaseQueryResult = [{ name: 'Sly Moore' }];
// //   return next();
// // };









