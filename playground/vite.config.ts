import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';

// Relative base: the build works from any static path (GitHub Pages, a subfolder, file servers).
export default defineConfig({
  base: './',
  plugins: [react()],
  build: { target: 'es2022', chunkSizeWarningLimit: 2000 },
});
