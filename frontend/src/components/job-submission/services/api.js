import apiClient from '../../appClient'; 

export async function detectCsvFormat(file) {
  const formData = new FormData();
  formData.append('file', file);
  const { data } = await apiClient.post('/api/detect-csv-format/', formData);
  return data;
}

export async function validateCsv({ file, predictionType, kcatMethod, kmMethod }) {
  const formData = new FormData();
  formData.append('file', file);
  formData.append('predictionType', predictionType);
  formData.append('kcatMethod', kcatMethod);
  formData.append('kmMethod', kmMethod);
  const { data } = await apiClient.post('/api/validate-input/', formData);
  return data;
}

export async function fetchSequenceSimilaritySummary({ file, useExperimental }) {
  const formData = new FormData();
  formData.append('file', file);
  formData.append('useExperimental', useExperimental);
  const { data } = await apiClient.post('/api/sequence-similarity-summary/', formData);
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
  const { data } = await apiClient.post('/api/submit-job/', formData, {
    headers: { 'Content-Type': 'multipart/form-data' },
  });
  return data;
}
