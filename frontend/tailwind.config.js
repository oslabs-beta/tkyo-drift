import textshadow from 'tailwindcss-textshadow'

export default ({
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
        glow: '0 0 10px #B94EFF, 0 0 40px #B94EFF',
      },
      fontFamily: {
        Audiowide: ['"Audiowide"', 'sans-serif'],
        code: ['"Source Code Pro"', 'monospace'],
      },
      textShadow: {
        glow: '0 0 12px #B94EFF, 0 0 24px #B94EFF',
      }
    },
  },
  plugins: [
    textshadow
  ],
});