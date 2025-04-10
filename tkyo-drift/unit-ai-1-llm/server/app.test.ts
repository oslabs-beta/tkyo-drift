import app from './app';
import request from 'supertest';

jest.mock('./controllers/naturalLanguageController', () => ({
  parseNaturalLanguageQuery: jest.fn((req, res, next) => {
    res.locals.naturalLanguageQuery = req.body.naturalLanguageQuery;
    next();
  }),
}));

jest.mock('./controllers/openaiController', () => ({
  queryOpenAI: jest.fn((_req, res, next) => {
    res.locals.databaseQuery =
      "SELECT name FROM public.people WHERE eye_color = 'white';";
    next();
  }),
}));

jest.mock('./controllers/databaseController', () => ({
  queryStarWarsDatabase: jest.fn((_req, res, next) => {
    res.locals.databaseQueryResult = [{ name: 'Sly Moore' }];
    next();
  }),
}));

describe('POST /api', () => {
  it('should return 200 and the expected response', async () => {
    const response = await request(app)
      .post('/api')
      .send(
        JSON.stringify({
          naturalLanguageQuery: 'Name the person with white eyes',
        })
      )
      .set('Content-Type', 'application/json');

    expect(response.status).toBe(200);
    expect(response.body).toHaveProperty('databaseQuery');
    expect(response.body).toHaveProperty('databaseQueryResult');
    expect(response.body.databaseQuery).toBe(
      "SELECT name FROM public.people WHERE eye_color = 'white';"
    );
    expect(response.body.databaseQueryResult).toEqual([{ name: 'Sly Moore' }]);
  });
});
