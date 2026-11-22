import { defineConfig } from "vite";

export default defineConfig({
  esbuild: { jsx: "automatic" },
  server: { proxy: { "/api/workbench": "http://127.0.0.1:8765" } },
  preview: { proxy: { "/api/workbench": "http://127.0.0.1:8765" } },
});
