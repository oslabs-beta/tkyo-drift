import { queryOpenAI } from './openaiController';
import { Request, Response, NextFunction } from 'express';
import OpenAI from 'openai';

jest.mock('openai', () => {
  const mOpenAI = {
    chat: {
      completions: {
        create: jest.fn(),
      },
    },
  };
  return jest.fn(() => mOpenAI);
});

describe('queryOpenAI', () => {
  let req: Partial<Request>;
  let res: Partial<Response>;
  let next: NextFunction;
  let mockOpenAI: jest.Mocked<Partial<OpenAI>>;

  beforeEach(() => {
    req = {};
    res = {
      locals: {},
    };
    next = jest.fn();
    mockOpenAI = new OpenAI();
    jest.clearAllMocks();
  });

  it('should return an error if no naturalLanguageQuery is provided', async () => {
    await queryOpenAI(req as Request, res as Response, next);

    expect(next).toHaveBeenCalledWith({
      log: 'OpenAI query middleware did not receive a query',
      status: 500,
      message: { err: 'An error occurred before querying OpenAI' },
    });
  });

  it('should return an error if OpenAI does not return a completion', async () => {
    res.locals!.naturalLanguageQuery = '???';
    (mockOpenAI.chat?.completions.create as jest.Mock).mockResolvedValue({
      choices: [{ message: { content: '' } }],
    });

    await queryOpenAI(req as Request, res as Response, next);

    expect(next).toHaveBeenCalledWith(
      expect.objectContaining({
        log: 'OpenAI did not return a completion',
        status: 500,
        message: { err: 'An error occurred while querying OpenAI' },
      })
    );
  });

  it('should set res.locals.databaseQuery if OpenAI returns a valid completion', async () => {
    res.locals!.naturalLanguageQuery = 'Name the person with white eyes';
    (mockOpenAI.chat?.completions.create as jest.Mock).mockResolvedValue({
      choices: [
        {
          message: {
            content:
              "```sql\nSELECT name FROM public.people WHERE eye_color = 'white';\n```",
          },
        },
      ],
    });

    await queryOpenAI(req as Request, res as Response, next);

    expect(res.locals!.databaseQuery).toEqual(
      expect.stringContaining(
        "SELECT name FROM public.people WHERE eye_color = 'white';"
      )
    );

    expect(next).toHaveBeenCalled();
  });

  it('should return an error if OpenAI throws an error', async () => {
    res.locals!.naturalLanguageQuery = 'Name the person with white eyes';
    (mockOpenAI.chat?.completions.create as jest.Mock).mockRejectedValue(
      new Error('OpenAI error')
    );

    await queryOpenAI(req as Request, res as Response, next);

    expect(next).toHaveBeenCalledWith(
      expect.objectContaining({
        log: expect.stringContaining('queryOpenAI: Error: OpenAI error'),
        status: 500,
        message: { err: 'An error occurred while querying OpenAI' },
      })
    );
  });
});
