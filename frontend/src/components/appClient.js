import axios from 'axios';
export async function ensureCsrfCookie(api = apiClient) {
  try { await api.get('/csrf/'); } catch (_) { /* swallow */ }
}
const apiClient = axios.create({
  baseURL:import.meta.env.VITE_API_BASE_URL || '/api',
});

export default apiClient;