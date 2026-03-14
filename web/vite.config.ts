import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'
import tailwindcss from '@tailwindcss/vite'

export default defineConfig({
  plugins: [react(), tailwindcss()],
  base: '/dashboard/',
  build: {
    outDir: '../src/uniinfer/api/static',
    emptyOutDir: true,
  },
  server: {
    proxy: {
      '/api/dashboard': 'http://localhost:8000',
    },
  },
})
