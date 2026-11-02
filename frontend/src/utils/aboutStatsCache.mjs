export const ABOUT_STATS_STORAGE_KEY = 'about_stats_payload_v3';
export const ABOUT_STATS_MAX_AGE_MS = 20 * 60 * 1000;
const COUNTS = ['jobs_completed', 'reactions_completed', 'unique_protein_sequences',
  'parameter_predictions_completed', 'kcat_predictions_completed',
  'km_predictions_completed', 'kcat_km_predictions_completed'];

export function isFreshAboutStats(payload, now = Date.now()) {
  if (!payload || payload.schema_version !== 3) return false;
  const age = now - Date.parse(payload.generated_at);
  return Number.isFinite(age) && age >= 0 && age < ABOUT_STATS_MAX_AGE_MS &&
    COUNTS.every(key => Number.isSafeInteger(payload[key]) && payload[key] >= 0);
}

export function readAboutStats(storage, now = Date.now()) {
  try {
    const payload = JSON.parse(storage.getItem(ABOUT_STATS_STORAGE_KEY));
    return isFreshAboutStats(payload, now) ? payload : null;
  } catch {
    return null;
  }
}
