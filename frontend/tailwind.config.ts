import type { Config } from "tailwindcss";

const config: Config = {
  content: [
    "./app/**/*.{js,ts,jsx,tsx,mdx}",
    "./components/**/*.{js,ts,jsx,tsx,mdx}",
  ],
  theme: { 
    extend: {
      colors: {
        navy: {
          DEFAULT: '#111827', // Dark navy/slate
          light: '#1f2937',
          dark: '#030712',
          deep: '#0f172a', // More blueish navy
        },
        gold: {
          DEFAULT: '#d97706', // Deep yellow/gold
          light: '#f59e0b',
          dark: '#b45309',
        }
      }
    } 
  },
  plugins: [],
};

export default config;
