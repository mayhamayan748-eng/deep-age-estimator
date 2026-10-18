import tailwindcss from "@tailwindcss/vite";
import react from "@vitejs/plugin-react";
import { defineConfig } from "vite";

export default defineConfig({
  plugins: [react(), tailwindcss()],
  define: {
    // Changes on every build; appended to model/data URLs so browsers never reuse files from an older deploy
    __BUILD_ID__: JSON.stringify(Date.now().toString(36)),
  },
  server: {
    // In development the API runs on :8000; in Docker nginx proxies /api instead
    proxy: { "/api": "http://localhost:8000" },
  },
});
