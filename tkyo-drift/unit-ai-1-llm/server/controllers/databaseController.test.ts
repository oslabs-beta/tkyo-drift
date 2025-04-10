import { Request, Response, NextFunction } from 'express';
import { queryStarWarsDatabase } from './databaseController';
import pg from 'pg';

jest.mock('pg', () => {
  const mPool = {
    query: jest.fn(),
  };
  return { Pool: jest.fn(() => mPool) };
});

describe('databaseController', () => {
  let req: Partial<Request>;
  let res: Partial<Response>;
  let next: NextFunction;
  let mockPool: jest.Mocked<Partial<pg.Pool>>;

  beforeEach(() => {
    req = {};
    res = {
      locals: {},
    };
    next = jest.fn();
    mockPool = new pg.Pool();
    jest.clearAllMocks();
  });

  describe('queryStarWarsDatabase middleware', () => {
    it('should return an error if databaseQuery is not provided', async () => {
      await queryStarWarsDatabase(req as Request, res as Response, next);
      expect(next).toHaveBeenCalledWith({
        log: 'Database query middleware did not receive a query',
        status: 500,
        message: { err: 'An error occurred before querying the database' },
      });
    });

    it('should return an error if databaseQuery is not a SELECT statement', async () => {
      res.locals!.databaseQuery = 'INSERT INTO users (name) VALUES ($1)';
      await queryStarWarsDatabase(req as Request, res as Response, next);
      expect(next).toHaveBeenCalledWith({
        log: 'Database query middleware received a non-SELECT query',
        status: 500,
        message: { err: 'An error occurred before querying the database' },
      });
    });

    it('should set databaseQueryResult on res.locals if query is successful', async () => {
      res.locals!.databaseQuery =
        "SELECT name FROM public.people WHERE eye_color = 'white';";
      const queryResult = { rows: [{ name: 'Sly Moore' }] };
      (mockPool.query as jest.Mock).mockResolvedValue(queryResult);

      await queryStarWarsDatabase(req as Request, res as Response, next);
      expect(mockPool.query).toHaveBeenCalledWith(
        "SELECT name FROM public.people WHERE eye_color = 'white';"
      );
      expect(res.locals?.databaseQueryResult).toEqual(queryResult.rows);
      expect(next).toHaveBeenCalled();
    });

    it('should return an error if pool.query throws an error', async () => {
      res.locals!.databaseQuery = 'SELECT * FROM users';
      const error = new Error('Database error');
      (mockPool.query as jest.Mock).mockRejectedValue(error);

      await queryStarWarsDatabase(req as Request, res as Response, next);
      expect(mockPool.query).toHaveBeenCalledWith('SELECT * FROM users');
      expect(next).toHaveBeenCalledWith(
        expect.objectContaining({
          status: 500,
          message: { err: 'An error occurred while querying database' },
        })
      );
    });
  });
});
