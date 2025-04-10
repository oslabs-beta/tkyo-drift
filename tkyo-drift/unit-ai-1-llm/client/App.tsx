import React from 'react';
import { createRoot } from 'react-dom/client';

import './styles.css';
import Dashboard from './Components/Dashboard';

const App = () => {
  return (
    <div>
      <Dashboard />
    </div>
  );
};

createRoot(document.querySelector('#root')!).render(<App />);

export default App;
