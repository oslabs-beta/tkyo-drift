/** @type {import('tailwindcss').Config} */
export default {
  content: [
    "./index.html",
    "./src/**/*.{js,ts,jsx,tsx}",
  ],
  theme: {
    extend: {
      colors: {
        neonPurple: '#B94EFF',
      },
      boxShadow: {
        neon: '0 0 18px #B94EFF',
      },
      fontFamily: {
        Audiowide: ['"Audiowide"', 'sans-serif'],
        code: ['"Source Code Pro"', 'monospace'],
      },
    },
  },
  plugins: [],
};

