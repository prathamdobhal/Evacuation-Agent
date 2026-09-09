import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'
import tailwindcss from '@tailwindcss/vite'

export default defineConfig({
  plugins: [react(), tailwindcss()],
  server: {
    port: 5173,
    proxy: {
      '/calamity': 'http://localhost:8000',
      '/route': 'http://localhost:8000',
      '/camps': 'http://localhost:8000',
      '/health': 'http://localhost:8000',
    }
  }
})