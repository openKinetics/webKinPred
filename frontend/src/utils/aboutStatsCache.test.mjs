import test from 'node:test';
import assert from 'node:assert/strict';
import { isFreshAboutStats, readAboutStats, ABOUT_STATS_MAX_AGE_MS } from './aboutStatsCache.mjs';
const now = Date.parse('2026-11-02T12:00:00Z');
const payload = { schema_version: 3, generated_at: new Date(now).toISOString(),
  jobs_completed: 1, reactions_completed: 2, unique_protein_sequences: 1,
  parameter_predictions_completed: 3, kcat_predictions_completed: 1,
  km_predictions_completed: 1, kcat_km_predictions_completed: 1 };
test('cache uses the snapshot date rather than resetting freshness on page visits', () => {
  assert.ok(isFreshAboutStats(payload, now + ABOUT_STATS_MAX_AGE_MS - 1));
  assert.equal(isFreshAboutStats(payload, now + ABOUT_STATS_MAX_AGE_MS), false);
  assert.equal(readAboutStats({ getItem: () => JSON.stringify(payload) }, now + ABOUT_STATS_MAX_AGE_MS), null);
});
test('rejects old schemas, bad dates, malformed data, and unavailable storage', () => {
  for (const p of [null, {}, { ...payload, schema_version: 1 }, { ...payload, generated_at: 'bad' },
    { ...payload, jobs_completed: -1 }, { ...payload, km_predictions_completed: '1' }]) {
    assert.equal(isFreshAboutStats(p, now), false);
  }
  assert.equal(readAboutStats({ getItem: () => '{' }, now), null);
  assert.equal(readAboutStats({ getItem() { throw new Error('blocked'); } }, now), null);
});
test('loads a fresh complete preview', () => {
  assert.deepEqual(readAboutStats({ getItem: () => JSON.stringify(payload) }, now), payload);
});
