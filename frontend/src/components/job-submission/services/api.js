// /home/saleh/webKinPred/frontend/src/components/job-submission/services/api.js
import apiClient from '../../appClient';

export async function detectCsvFormat(file) {
  const formData = new FormData();
  formData.append('file', file);
  const { data } = await apiClient.post('/detect-csv-format/', formData);
  return data;
}

export async function validateCsv({ file, predictionType, kcatMethod, kmMethod}) {
  const formData = new FormData();
  formData.append('file', file);
  formData.append('predictionType', predictionType);
  formData.append('kcatMethod', kcatMethod);
  formData.append('kmMethod', kmMethod);
  const { data } = await apiClient.post('/validate-input/', formData);
  return data;
}

export async function fetchSequenceSimilaritySummary({ file, useExperimental, validationSessionId }) {
  const formData = new FormData();
  formData.append('file', file);
  formData.append('useExperimental', useExperimental);
  if (validationSessionId) formData.append('validationSessionId', validationSessionId);
  const { data } = await apiClient.post('/sequence-similarity-summary/', formData);
  return data;
}

export async function submitJob({
  predictionType,
  kcatMethod,
  kmMethod,
  file,
  handleLongSequences,
  useExperimental,
}) {
  const formData = new FormData();
  formData.append('predictionType', predictionType);
  formData.append('kcatMethod', kcatMethod);
  formData.append('kmMethod', kmMethod);
  formData.append('file', file);
  formData.append('handleLongSequences', handleLongSequences);
  formData.append('useExperimental', useExperimental);
  const { data } = await apiClient.post('/submit-job/', formData, {
    headers: { 'Content-Type': 'multipart/form-data' },
  });
  return data;
}
export function openProgressStream(sessionId) {
  const baseURL = import.meta.env.VITE_API_BASE_URL || '/api/';
  const url = `${baseURL}/progress-stream/?session_id=${encodeURIComponent(sessionId)}`;
  return new EventSource(url);
}
export async function cancelValidationApi(sessionId) {
  const formData = new FormData();
  formData.append('session_id', sessionId);
  const { data } = await apiClient.post('/cancel-validation/', formData);
  return data;
}
