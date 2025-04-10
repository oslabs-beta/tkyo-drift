/**
 * @jest-environment jsdom
 */

import React from 'react';
import { render, screen, fireEvent, waitFor } from '@testing-library/react';
import '@testing-library/jest-dom';
import Dashboard from './Dashboard';
import 'whatwg-fetch';
import nock from 'nock';

describe('Dashboard Component', () => {
  beforeAll(() => {
    nock(window.location.origin)
      .persist()
      .post(
        '/api',
        JSON.stringify({
          naturalLanguageQuery: 'Name the person with white eyes',
        }),
        {
          reqheaders: {
            'Content-Type': 'application/json',
          },
        }
      )
      .reply(200, {
        databaseQuery:
          "SELECT name FROM public.people WHERE eye_color = 'white';",
        databaseQueryResult: [{ name: 'Sly Moore' }],
      });
    nock(window.location.origin)
      .persist()
      .post(
        '/api',
        JSON.stringify({
          naturalLanguageQuery: 'This API call fails',
        }),
        {
          reqheaders: {
            'Content-Type': 'application/json',
          },
        }
      )
      .reply(500, {
        err: 'An error occurred',
      });
  });

  test('renders the form elements correctly', () => {
    render(<Dashboard />);
    expect(
      screen.getByPlaceholderText('Enter your natural language query here...')
    ).toBeInTheDocument();
    expect(screen.getByText('Convert and Execute')).toBeInTheDocument();
  });

  test('displays loading state when form is submitted', async () => {
    render(<Dashboard />);
    fireEvent.change(
      screen.getByPlaceholderText('Enter your natural language query here...'),
      {
        target: { value: 'Name the person with white eyes' },
      }
    );
    fireEvent.click(screen.getByText('Convert and Execute'));

    expect(screen.getByText('Converting...')).toBeInTheDocument();
  });

  test('displays error message on failed API call', async () => {
    render(<Dashboard />);
    fireEvent.change(
      screen.getByPlaceholderText('Enter your natural language query here...'),
      {
        target: { value: 'This API call fails' },
      }
    );
    fireEvent.click(screen.getByText('Convert and Execute'));

    await waitFor(() =>
      expect(screen.getByText('An error occurred')).toBeInTheDocument()
    );
  });

  test('displays generated SQL query and results on successful API call', async () => {
    render(<Dashboard />);
    fireEvent.change(
      screen.getByPlaceholderText('Enter your natural language query here...'),
      {
        target: { value: 'Name the person with white eyes' },
      }
    );
    fireEvent.click(screen.getByText('Convert and Execute'));

    await waitFor(() =>
      expect(screen.getByText('Generated SQL Query:')).toBeInTheDocument()
    );
    expect(
      screen.getByText(
        "SELECT name FROM public.people WHERE eye_color = 'white';"
      )
    ).toBeInTheDocument();
    expect(screen.getByText('Query Results:')).toBeInTheDocument();
    expect(screen.getByText('name')).toBeInTheDocument();
    expect(screen.getByText('Sly Moore')).toBeInTheDocument();
  });

  test('clears previous results and error messages on new submission', async () => {
    render(<Dashboard />);
    fireEvent.change(
      screen.getByPlaceholderText('Enter your natural language query here...'),
      {
        target: { value: 'This API call fails' },
      }
    );
    fireEvent.click(screen.getByText('Convert and Execute'));

    await waitFor(() =>
      expect(screen.getByText('An error occurred')).toBeInTheDocument()
    );

    fireEvent.change(
      screen.getByPlaceholderText('Enter your natural language query here...'),
      {
        target: { value: 'Name the person with white eyes' },
      }
    );
    fireEvent.click(screen.getByText('Convert and Execute'));

    await waitFor(() =>
      expect(screen.queryByText('Error occurred')).not.toBeInTheDocument()
    );
    await waitFor(() =>
      expect(screen.getByText('Generated SQL Query:')).toBeInTheDocument()
    );
    expect(
      screen.getByText(
        "SELECT name FROM public.people WHERE eye_color = 'white';"
      )
    ).toBeInTheDocument();
    expect(screen.getByText('Query Results:')).toBeInTheDocument();
    expect(screen.getByText('name')).toBeInTheDocument();
    expect(screen.getByText('Sly Moore')).toBeInTheDocument();
  });
});
