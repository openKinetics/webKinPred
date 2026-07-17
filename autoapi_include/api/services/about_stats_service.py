"""
About-page usage metrics aggregation.

This module stores a persistent cache in the Django database so /api/about-stats/
can serve quickly without rescanning output CSV files on every request.
"""

from __future__ import annotations

import csv
import json
import os
from typing import Any

from django.db.utils import OperationalError, ProgrammingError
from django.utils import timezone

from api.models import AboutStatsCache, Job

KCAT_COL = "kcat (1/s)"
KM_COL = "KM (mM)"
KCAT_KM_COL = "kcat/Km (1/(s*mM))"
PROTEIN_SEQUENCE_COL = "Protein Sequence"
CACHE_KEY = "about_stats"


def _has_non_empty_value(value: Any) -> bool:
    if value is None:
        return False
    return str(value).strip() != ""


def _iter_available_completed_output_paths() -> tuple[int, list[str]]:
    completed_jobs = Job.objects.filter(status="Completed").only("output_file")
    jobs_completed = int(completed_jobs.count())

    output_paths: list[str] = []
    for job in completed_jobs.iterator(chunk_size=200):
        output_file = getattr(job, "output_file", None)
        if not output_file:
            continue
        try:
            file_path = output_file.path
        except Exception:
            continue
        if file_path and os.path.isfile(file_path):
            output_paths.append(file_path)

    return jobs_completed, output_paths


def _compute_about_stats_payload(now_iso: str) -> dict[str, Any]:
    jobs_completed, output_paths = _iter_available_completed_output_paths()

    reactions_completed = 0
    kcat_predictions_completed = 0
    km_predictions_completed = 0
    kcat_km_predictions_completed = 0
    unique_sequences: set[str] = set()

    for file_path in output_paths:
        try:
            with open(file_path, "r", encoding="utf-8", newline="") as handle:
                reader = csv.DictReader(handle)
                for row in reader:
                    kcat_has = _has_non_empty_value(row.get(KCAT_COL))
                    km_has = _has_non_empty_value(row.get(KM_COL))
                    kcat_km_has = _has_non_empty_value(row.get(KCAT_KM_COL))

                    if kcat_has:
                        kcat_predictions_completed += 1
                    if km_has:
                        km_predictions_completed += 1
                    if kcat_km_has:
                        kcat_km_predictions_completed += 1
                    if kcat_has or km_has or kcat_km_has:
                        reactions_completed += 1

                    sequence = str(row.get(PROTEIN_SEQUENCE_COL, "")).strip()
                    if sequence:
                        unique_sequences.add(sequence)
        except Exception:
            # Policy choice: silently skip missing/unreadable/invalid output files.
            continue

    parameter_predictions_completed = (
        kcat_predictions_completed
        + km_predictions_completed
        + kcat_km_predictions_completed
    )

    return {
        "scope": "all_time",
        "generated_at": now_iso,
        "jobs_completed": jobs_completed,
        "reactions_completed": reactions_completed,
        "unique_protein_sequences": len(unique_sequences),
        "parameter_predictions_completed": parameter_predictions_completed,
        "kcat_predictions_completed": kcat_predictions_completed,
        "km_predictions_completed": km_predictions_completed,
        "kcat_km_predictions_completed": kcat_km_predictions_completed,
    }


def _with_derived_about_stats(payload: dict[str, Any]) -> dict[str, Any]:
    """Add derived fields to older cached payloads without forcing a refresh."""
    if "parameter_predictions_completed" in payload:
        return payload

    prediction_counts = [
        payload.get("kcat_predictions_completed"),
        payload.get("km_predictions_completed"),
        payload.get("kcat_km_predictions_completed"),
    ]
    if all(isinstance(value, int) and not isinstance(value, bool) for value in prediction_counts):
        return {
            **payload,
            "parameter_predictions_completed": sum(prediction_counts),
        }
    return payload


def _deserialise_payload(raw: str) -> dict[str, Any] | None:
    if not raw:
        return None
    try:
        payload = json.loads(raw)
    except Exception:
        return None
    return _with_derived_about_stats(payload) if isinstance(payload, dict) else None


def _compute_fallback_without_db() -> dict[str, Any]:
    return _compute_about_stats_payload(timezone.now().isoformat())


def _get_or_create_cache_row() -> AboutStatsCache:
    row, _ = AboutStatsCache.objects.get_or_create(
        key=CACHE_KEY,
        defaults={
            "payload": "",
            "is_stale": True,
            "generated_at": None,
        },
    )
    return row


def mark_about_stats_cache_stale() -> None:
    """Mark cache stale so next refresh recomputes payload."""
    try:
        row = _get_or_create_cache_row()
        if row.is_stale:
            return
        row.is_stale = True
        row.save(update_fields=["is_stale", "updated_at"])
    except (OperationalError, ProgrammingError):
        # Database schema may not be migrated yet.
        return


def refresh_about_stats_cache(*, force: bool = False) -> dict[str, Any]:
    """
    Refresh and persist About stats cache.

    If force=False and cache is fresh, return cached payload as-is.
    """
    now = timezone.now()

    try:
        row = _get_or_create_cache_row()
        cached_payload = _deserialise_payload(row.payload)

        if not force and not row.is_stale and cached_payload is not None:
            return cached_payload

        payload = _compute_about_stats_payload(now.isoformat())
        row.payload = json.dumps(payload, separators=(",", ":"), ensure_ascii=False)
        row.generated_at = now
        row.is_stale = False
        row.save(update_fields=["payload", "generated_at", "is_stale", "updated_at"])
        return payload
    except (OperationalError, ProgrammingError):
        # Graceful fallback until migration exists.
        return _compute_fallback_without_db()


def get_about_stats() -> dict[str, Any]:
    """Return cached About stats, recomputing only when stale/missing."""
    try:
        row = _get_or_create_cache_row()
        payload = _deserialise_payload(row.payload)
        if not row.is_stale and payload is not None:
            return payload
        return refresh_about_stats_cache(force=True)
    except (OperationalError, ProgrammingError):
        return _compute_fallback_without_db()
