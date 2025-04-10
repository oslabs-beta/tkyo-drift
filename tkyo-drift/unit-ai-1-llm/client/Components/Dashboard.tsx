import React, { useState } from 'react';

type ResultRow = Record<string, string | number>;

interface ParsedConverterResponse {
  databaseQuery: string;
  databaseQueryResult: ResultRow[];
}

const Dashboard = () => {
  const [naturalLanguageQuery, setNaturalLanguageQuery] = useState('');
  const [databaseQuery, setDatabaseQuery] = useState('');
  const [databaseQueryResults, setDatabaseQueryResults] = useState<ResultRow[]>(
    []
  );
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setLoading(true);
    setError('');
    setDatabaseQuery('');
    setDatabaseQueryResults([]);

    try {
      const converterResponse = await fetch('/api', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ naturalLanguageQuery }),
      });

      if (converterResponse.status !== 200) {
        const parsedError: { err: string } = await converterResponse.json();
        setError(parsedError.err);
      } else {
        const parsedConverterResponse: ParsedConverterResponse =
          await converterResponse.json();
        setDatabaseQuery(parsedConverterResponse.databaseQuery);
        setDatabaseQueryResults(parsedConverterResponse.databaseQueryResult);
      }
    } catch (_err) {
      setError('Error processing your request.');
    } finally {
      setLoading(false);
    }
  };

  const renderTable = () => {
    if (databaseQueryResults.length === 0) return null;

    const columns = Object.keys(databaseQueryResults[0]);

    return (
      <table>
        <thead>
          <tr>
            {columns.map((col) => (
              <th key={col}>{col}</th>
            ))}
          </tr>
        </thead>
        <tbody>
          {databaseQueryResults.map((row, index) => (
            <tr key={index}>
              {columns.map((col) => (
                <td key={col}>{row[col]}</td>
              ))}
            </tr>
          ))}
        </tbody>
      </table>
    );
  };

  return (
    <div className="container">
      <form onSubmit={handleSubmit}>
        <textarea
          value={naturalLanguageQuery}
          onChange={(e) => setNaturalLanguageQuery(e.target.value)}
          placeholder="Enter your natural language query here..."
          required
        />
        <button type="submit" disabled={loading}>
          {loading ? 'Converting...' : 'Convert and Execute'}
        </button>
      </form>
      {error && <p className="error">{error}</p>}
      {databaseQuery && (
        <div className="result">
          <h2>Generated SQL Query:</h2>
          <pre>{databaseQuery}</pre>
        </div>
      )}
      {databaseQueryResults.length > 0 && (
        <div className="result">
          <h2>Query Results:</h2>
          {renderTable()}
        </div>
      )}
    </div>
  );
};

export default Dashboard;
