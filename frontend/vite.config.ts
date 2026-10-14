import tailwindcss from "@tailwindcss/vite";
import react from "@vitejs/plugin-react";
import { defineConfig } from "vite";

export default defineConfig({
  plugins: [react(), tailwindcss()],
  server: {
    // In development the API runs on :8000; in Docker nginx proxies /api instead
    proxy: { "/api": "http://localhost:8000" },
  },
});
