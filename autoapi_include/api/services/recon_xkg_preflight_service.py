"""Request-time, cache-only qualification for ReconXKG jobs."""

from __future__ import annotations

import logging
import math
import time
from dataclasses import dataclass
from numbers import Real
from typing import Any

import pandas as pd
from api.services import prediction_store
from api.services.prediction_batch_service import (
    build_sequence_batch_plan,
    build_target_batch_plan,
)
from api.services.similarity_service import (
    kcat_similarity_sequences_for_output_rows,
    similarity_cache_label_for_method,
)

_log = logging.getLogger(__name__)


@dataclass(frozen=True)
class ReconXkgCacheSnapshot:
    """Values captured by a successful full-cache preflight."""

    predictions: dict[str, Any]
    similarities: dict[str, tuple[float | None, float | None]]


@dataclass(frozen=True)
class ReconXkgPreflightResult:
    complete: bool
    snapshot: ReconXkgCacheSnapshot | None
    reason: str
    prediction_units: int
    unique_prediction_keys: int
    similarity_sequences: int


def preflight_recon_xkg_cache(
    *,
    dataframe: pd.DataFrame,
    targets: list[str],
    descriptors: dict[str, Any],
    handle_long_sequences: str,
    canonicalize_substrates: bool,
    include_similarity_columns: bool,
    job_public_id: str,
) -> ReconXkgPreflightResult:
    """Return a snapshot only when every required cached value is present."""
    started = time.monotonic()
    prediction_units = 0
    similarity_count = 0
    unique_keys: set[str] = set()
    reason = "cache-read-error"

    try:
        plan_started = time.monotonic()
        sequence_plan = build_sequence_batch_plan(
            dataframe,
            descriptors.values(),
            handle_long_sequences,
        )
        _log.info(
            "ReconXKG preflight sequence planning completed",
            extra={
                "event": "recon_xkg.preflight_sequence_plan",
                "job_public_id": job_public_id,
                "rows": len(dataframe),
                "valid_reactions": len(sequence_plan.valid_reaction_indices),
                "sequence_children": len(sequence_plan.expansion.children),
                "elapsed_ms": round((time.monotonic() - plan_started) * 1000, 2),
            },
        )
        target_batches: dict[str, Any] = {}
        target_keys: dict[str, list[str | None]] = {}
        key_started = time.monotonic()
        for target in targets:
            descriptor = descriptors[target]
            batch = build_target_batch_plan(
                descriptor,
                target,
                dataframe,
                sequence_plan,
            )
            target_batches[target] = batch
            keys, _components, _params_fp = prediction_store.build_unit_keys(
                descriptor,
                target,
                batch.sequences,
                batch.call_kwargs,
                canonicalize_substrates,
            )
            target_keys[target] = keys
            prediction_units += len(keys)
            unique_keys.update(key for key in keys if key is not None)
            if any(key is None for key in keys):
                reason = "uncacheable-prediction-unit"
                return _logged_result(
                    False,
                    None,
                    reason,
                    prediction_units,
                    len(unique_keys),
                    0,
                    job_public_id,
                    started,
                )

        _log.info(
            "ReconXKG preflight key planning completed",
            extra={
                "event": "recon_xkg.preflight_key_plan",
                "job_public_id": job_public_id,
                "prediction_units": prediction_units,
                "unique_prediction_keys": len(unique_keys),
                "elapsed_ms": round((time.monotonic() - key_started) * 1000, 2),
            },
        )
        read_started = time.monotonic()
        prediction_values = prediction_store.get_many(unique_keys)
        _log.info(
            "ReconXKG preflight prediction cache read completed",
            extra={
                "event": "recon_xkg.preflight_prediction_read",
                "job_public_id": job_public_id,
                "requested_keys": len(unique_keys),
                "cache_hits": len(prediction_values),
                "elapsed_ms": round((time.monotonic() - read_started) * 1000, 2),
            },
        )
        if any(
            key not in prediction_values
            or not prediction_store.cached_outcome_is_valid(prediction_values[key])
            for key in unique_keys
        ):
            reason = "prediction-cache-miss"
            return _logged_result(
                False,
                None,
                reason,
                prediction_units,
                len(unique_keys),
                0,
                job_public_id,
                started,
            )

        similarity_values: dict[str, tuple[float | None, float | None]] = {}
        if include_similarity_columns and "kcat" in targets:
            method_key = descriptors["kcat"].key
            cache_label = similarity_cache_label_for_method(method_key)
            if not cache_label:
                reason = "similarity-cache-unavailable"
                return _logged_result(
                    False,
                    None,
                    reason,
                    prediction_units,
                    len(unique_keys),
                    0,
                    job_public_id,
                    started,
                )

            unique_sequences = _cached_similarity_sequences_for_kcat(
                dataframe=dataframe,
                kcat_batch=target_batches.get("kcat"),
                kcat_keys=target_keys.get("kcat", []),
                prediction_values=prediction_values,
            )
            similarity_count = len(unique_sequences)
            sequence_hashes = {
                sequence: prediction_store.sha256_text(sequence)
                for sequence in unique_sequences
            }
            similarity_started = time.monotonic()
            similarity_values = prediction_store.get_similarity_many(
                sequence_hashes,
                cache_label,
            )
            _log.info(
                "ReconXKG preflight similarity cache read completed",
                extra={
                    "event": "recon_xkg.preflight_similarity_read",
                    "job_public_id": job_public_id,
                    "requested_sequences": len(unique_sequences),
                    "cache_hits": len(similarity_values),
                    "elapsed_ms": round((time.monotonic() - similarity_started) * 1000, 2),
                },
            )
            if any(
                sequence not in similarity_values
                or not _valid_similarity_entry(similarity_values[sequence])
                for sequence in unique_sequences
            ):
                reason = "similarity-cache-miss"
                return _logged_result(
                    False,
                    None,
                    reason,
                    prediction_units,
                    len(unique_keys),
                    similarity_count,
                    job_public_id,
                    started,
                )

        snapshot = ReconXkgCacheSnapshot(
            predictions=dict(prediction_values),
            similarities=dict(similarity_values),
        )
        return _logged_result(
            True,
            snapshot,
            "full-cache-hit",
            prediction_units,
            len(unique_keys),
            similarity_count,
            job_public_id,
            started,
        )
    except Exception:
        _log.warning(
            "ReconXKG immediate cache preflight failed; queueing normally",
            extra={
                "event": "recon_xkg.preflight_error",
                "job_public_id": job_public_id,
                "prediction_units": prediction_units,
                "unique_prediction_keys": len(unique_keys),
                "similarity_sequences": similarity_count,
            },
            exc_info=True,
        )
        return _logged_result(
            False,
            None,
            reason,
            prediction_units,
            len(unique_keys),
            similarity_count,
            job_public_id,
            started,
        )


def _logged_result(
    complete: bool,
    snapshot: ReconXkgCacheSnapshot | None,
    reason: str,
    prediction_units: int,
    unique_prediction_keys: int,
    similarity_sequences: int,
    job_public_id: str,
    started: float,
) -> ReconXkgPreflightResult:
    _log.info(
        "ReconXKG immediate cache preflight completed",
        extra={
            "event": "recon_xkg.preflight_hit" if complete else "recon_xkg.preflight_miss",
            "job_public_id": job_public_id,
            "reason": reason,
            "prediction_units": prediction_units,
            "unique_prediction_keys": unique_prediction_keys,
            "similarity_sequences": similarity_sequences,
            "elapsed_ms": round((time.monotonic() - started) * 1000, 2),
        },
    )
    return ReconXkgPreflightResult(
        complete=complete,
        snapshot=snapshot,
        reason=reason,
        prediction_units=prediction_units,
        unique_prediction_keys=unique_prediction_keys,
        similarity_sequences=similarity_sequences,
    )


def _valid_similarity_entry(value: Any) -> bool:
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        return False
    return all(
        item is None or (isinstance(item, Real) and math.isfinite(float(item)))
        for item in value
    )


def _cached_similarity_sequences_for_kcat(
    *,
    dataframe: pd.DataFrame,
    kcat_batch: Any,
    kcat_keys: list[str | None],
    prediction_values: dict[str, Any],
) -> list[str]:
    if kcat_batch is None:
        return []

    selected_sequences: list[str] = [""] * len(dataframe)
    if kcat_batch.unit_expansion is not None:
        from api.utils.sequence_expansion import reduce_sequence_predictions

        child_predictions: list[Any] = []
        child_errors: dict[int, str] = {}
        for index, key in enumerate(kcat_keys):
            outcome = prediction_values.get(key) if key else None
            if isinstance(outcome, prediction_store.CachedFailure):
                child_predictions.append(None)
                child_errors[index] = outcome.reason
            else:
                child_predictions.append(prediction_store.coerce_value(outcome))

        reduced = reduce_sequence_predictions(
            plan=kcat_batch.unit_expansion,
            target="kcat",
            child_predictions=child_predictions,
            child_sources=[""] * len(child_predictions),
            child_errors=child_errors,
            child_details=None,
            reaction_count=len(dataframe),
        )
        selected_sequences = list(reduced.selected_sequences)

    output_like_dataframe = dataframe.copy()
    output_like_dataframe["Extra Info kcat"] = [""] * len(output_like_dataframe)
    return kcat_similarity_sequences_for_output_rows(
        output_like_dataframe,
        selected_sequences_by_row=selected_sequences,
    )
