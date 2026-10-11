import { defineConfig } from "vite";

export default defineConfig({
  base: "./",
  worker: { format: "es" },
  build: { target: "es2022" },
  server: { port: 3000 },
});
