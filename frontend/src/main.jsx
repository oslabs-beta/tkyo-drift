import React from 'react';
import ReactDOM from 'react-dom/client';
import { StrictMode } from 'react'
// eslint-disable-next-line no-unused-vars
import { createRoot } from 'react-dom/client'
import './index.css'
import App from './App.jsx'
import './styles/globals.css'

ReactDOM.createRoot(document.getElementById('root')).render(
  <React.StrictMode>
    <App />
  </React.StrictMode>,
);
