import type { Config } from 'jest';

const config: Config = {
  moduleNameMapper: {
    './controllers/(.*).js': './controllers/$1.ts',
  },
};

export default config;
