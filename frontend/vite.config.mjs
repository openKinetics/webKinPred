import { defineConfig, loadEnv } from 'vite';
import react from '@vitejs/plugin-react';
import svgr from 'vite-plugin-svgr';
import tsconfigPaths from 'vite-tsconfig-paths';

export default defineConfig(({ mode }) => {
  const env = loadEnv(mode, process.cwd(), '');
  
  // In Docker, use the backend service name, otherwise use localhost
  const DEV_API_TARGET =
    env.DEV_API_TARGET || 
    env.VITE_DEV_API_TARGET || 
    (process.env.NODE_ENV === 'development' && process.env.DOCKER ? 'http://backend:8000' : 'http://127.0.0.1:8000');

  console.log('🔧 Vite proxy target:', DEV_API_TARGET);

  return {
    plugins: [react(), svgr({ svgrOptions: { icon: true } }), tsconfigPaths()],
    base: '/',
    server: {
      host: '0.0.0.0', // Allow external connections for Docker
      port: 5173,
      strictPort: true,
      proxy: {
        '/api': {
          target: DEV_API_TARGET,
          changeOrigin: true,
          secure: false,
          configure: (proxy, options) => {
            proxy.on('error', (err, req, res) => {
              console.log('🚨 Proxy error:', err);
            });
            proxy.on('proxyReq', (proxyReq, req, res) => {
              console.log('🔄 Proxying request:', req.method, req.url, 'to', DEV_API_TARGET);
            });
            proxy.on('proxyRes', (proxyRes, req, res) => {
              console.log('✅ Proxy response:', proxyRes.statusCode, req.url);
            });
          }
        },
        '/media': {
          target: DEV_API_TARGET,
          changeOrigin: true,
          secure: false,
        },
      }
    },
    preview: { port: 4173, strictPort: true },
    build: { outDir: 'build', sourcemap: true },
    resolve: { alias: {} },
  };
});