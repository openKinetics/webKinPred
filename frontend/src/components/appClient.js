import axios from 'axios';
export async function ensureCsrfCookie(api = apiClient) {
  console.log('Making call to ' + api.defaults.baseURL + '/csrf/');
  try { await api.get('/csrf/'); } catch (_) { /* swallow */ }
}
const apiClient = axios.create({
  baseURL: import.meta.env.VITE_API_BASE_URL || '/api',
});

export default apiClient;