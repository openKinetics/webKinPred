import { defineConfig, loadEnv } from 'vite';
import react from '@vitejs/plugin-react';
import svgr from 'vite-plugin-svgr';
import tsconfigPaths from 'vite-tsconfig-paths';

export default defineConfig(({ mode }) => {
  const env = loadEnv(mode, process.cwd(), '');
  const DEV_API_TARGET =
    env.DEV_API_TARGET || env.VITE_DEV_API_TARGET || 'http://127.0.0.1:8000';

  return {
    plugins: [react(), svgr({ svgrOptions: { icon: true } }), tsconfigPaths()],
    base: '/',
    server: {
      port: 5173,
      strictPort: true,
      proxy: {
      '/api': {
        target: DEV_API_TARGET,
        changeOrigin: true,
      },
    }},
    preview: { port: 4173, strictPort: true },
    build: { outDir: 'build', sourcemap: true },
    resolve: { alias: {} },
  };
});