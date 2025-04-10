declare namespace Express {
  export interface Locals {
    naturalLanguageQuery?: string;
    databaseQuery?: string;
    databaseQueryResult?: Record<string, unknown>[];
  }
}
