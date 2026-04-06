/*
  vite.config.js — Build tool configuration
  ==========================================
  Vite needs two things configured:
    1. The React plugin (transforms JSX → JavaScript the browser understands)
    2. The Tailwind plugin (processes Tailwind utility classes into real CSS)

  This file is the equivalent of webpack.config.js but much simpler.
  Vite reads it automatically when you run 'npm run dev'.
*/

import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";
import tailwindcss from "@tailwindcss/vite";

export default defineConfig({
  plugins: [react(), tailwindcss()],

  // Proxy API requests to your FastAPI backend during development.
  // When your React app calls '/api/query', Vite forwards it to
  // http://localhost:8000/api/query. This avoids CORS issues in dev
  // and mimics how it'll work in production (same domain).
  server: {
    proxy: {
      "/api": {
        target: "http://localhost:8000",
        changeOrigin: true,
      },
    },
  },
});