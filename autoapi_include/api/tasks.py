# api/tasks.py
#
# Celery tasks for running kinetic-parameter predictions.
#
# There are two entry-point tasks:
#
#   run_prediction(public_id, method_key, target, experimental_results)
#       Used for single-target jobs (prediction_type = "kcat" or "Km").
#
#   run_both_prediction(public_id, kcat_key, km_key, experimental_results)
#       Legacy dual-target helper kept for compatibility with internal tools.
#       New submissions use run_multi_prediction.
#
#   run_multi_prediction(public_id, targets, methods, experimental_results)
#       Used by the current submission flow for one or more selected targets.
#
# Both tasks delegate to internal helpers (_execute_prediction /
# _execute_both_prediction) that contain the shared prediction logic.
# Adding a new method requires no changes here — it is picked up automatically
# by the method registry.

from __future__ import annotations

import json
import logging
import os
import time
from typing import Any

import pandas as pd
from django.conf import settings
from django.utils import timezone

from api.methods.base import PredictionError
from api.methods.registry import get as get_method
from api.models import Job, JobProgressStage
from api.observability.context import log_context
from api.prediction_engines.generic_subprocess import run_generic_subprocess_prediction
from api.services.about_stats_service import (
    mark_about_stats_cache_stale,
    refresh_about_stats_cache,
)
from api.services.gpu_precompute_status_service import clear_gpu_precompute_status
from api.services.job_progress_service import (
    initialise_job_progress_stages,
    mark_running_stage_failed,
    mark_stage_completed,
    mark_stage_failed,
    mark_stage_running,
    set_stage_prediction_snapshot,
)
from api.services.prediction_batch_service import (
    SequenceBatchPlan,
    build_sequence_batch_plan,
    build_target_batch_plan,
)
from api.services.recon_xkg_preflight_service import preflight_recon_xkg_cache
from api.services.similarity_service import append_kcat_similarity_columns_to_output_csv
from api.utils.extra_info import _source, build_extra_info
from api.utils.job_utils import canonicalise_targets
from api.utils.quotas import credit_back
from api.utils.safe_read import safe_read_csv
from api.utils.substrate_expansion import (
    SubstrateExpansionPlan,
    reduce_substrate_predictions,
)
from api.utils.sequence_expansion import (
    SequenceExpansionPlan,
    TargetExpansionPlan,
    reduce_sequence_predictions,
)
from celery import shared_task

_log = logging.getLogger(__name__)


class ReconXkgCacheOnlyMiss(PredictionError):
    """The strict synchronous executor could not consume its cache snapshot."""


_REALKCAT_CLASS_RANGES: dict[str, dict[int, tuple[float, float]]] = {
    "kcat": {
        0: (0.0, 3.32e-8),
        1: (3.33e-8, 1.0e-2),
        2: (1.01e-2, 1.0e-1),
        3: (1.01e-1, 1.0),
        4: (1.001, 10.0),
        5: (1.004e1, 1.0e2),
        6: (1.0025e2, 1.0e3),
        7: (1.002e3, 7.0e7),
    },
    "Km": {
        0: (1.0e-10, 1.0e-5),
        1: (1.01e-5, 1.0e-4),
        2: (1.002e-4, 1.0e-3),
        3: (1.002e-3, 1.0e-2),
        4: (1.008e-2, 1.0e-1),
        5: (1.01e-1, 1.02e2),
    },
}


# ---------------------------------------------------------------------------
# Celery tasks
# ---------------------------------------------------------------------------


def _safe_clear_gpu_precompute_status(public_id: str) -> None:
    try:
        clear_gpu_precompute_status(public_id)
    except Exception:
        # Redis telemetry cleanup is best-effort only.
        pass


def _safe_refresh_about_stats_cache() -> None:
    try:
        refresh_about_stats_cache(force=True)
    except Exception:
        # About stats are non-critical and should never break jobs.
        pass


def _safe_mark_about_stats_cache_stale() -> None:
    try:
        mark_about_stats_cache_stale()
    except Exception:
        # About stats are non-critical and should never break jobs.
        pass


def _format_prediction_value_and_source(desc, target: str, prediction: Any) -> tuple[Any, str]:
    """
    Return the output value and source string for one predicted row.

    For RealKcat only, convert class-index outputs to class-range mean values
    and include classifier context in the source text.
    """
    default_source = f"Prediction from {desc.display_name}"

    if desc.key != "RealKcat":
        return prediction, default_source

    if prediction is None:
        return None, default_source

    ranges = _REALKCAT_CLASS_RANGES.get(target)
    if not ranges:
        return prediction, default_source

    try:
        class_idx = int(prediction)
    except (TypeError, ValueError):
        return prediction, default_source

    class_range = ranges.get(class_idx)
    if class_range is None:
        return prediction, default_source

    low, high = class_range
    mean_value = (low + high) / 2.0
    source = (
        "Prediction from RealKcat (classifier): "
        f"class {class_idx}, range {low:.3e} to {high:.3e}; "
        "value is class-range mean."
    )
    return mean_value, source


@shared_task
def run_prediction(
    public_id: str,
    method_key: str,
    target: str,
    experimental_results: list | None = None,
    canonicalize_substrates: bool = True,
    include_similarity_columns: bool = True,
    disable_gpu_precompute: bool = False,
) -> None:
    """
    Run a single-target prediction job.

    Parameters
    ----------
    public_id : str
        The job's public identifier.
    method_key : str
        Registry key of the prediction method (e.g. ``"DLKcat"``).
    target : str
        Prediction target: ``"kcat"`` or ``"Km"``.
    experimental_results : list | None
        Pre-fetched experimental values to merge into the output, or None.
    """
    job = Job.objects.get(public_id=public_id)
    _safe_clear_gpu_precompute_status(public_id)
    job.status = "Processing"
    job.start_time = timezone.now()
    job.save(update_fields=["status", "start_time"])

    desc = get_method(method_key)

    try:
        df = _load_input(job)
        _execute_prediction(
            job,
            desc,
            df,
            target,
            experimental_results or [],
            canonicalize_substrates=canonicalize_substrates,
            include_similarity_columns=include_similarity_columns,
            disable_gpu_precompute=disable_gpu_precompute,
        )
        Job.objects.filter(pk=job.pk).update(
            status="Completed",
            completion_time=timezone.now(),
        )
        _safe_refresh_about_stats_cache()

    except PredictionError as e:
        Job.objects.filter(pk=job.pk).update(
            status="Failed",
            error_message=str(e),
            completion_time=timezone.now(),
        )

    except MemoryError:
        _handle_oom(job, desc.display_name)

    except Exception as e:
        Job.objects.filter(pk=job.pk).update(
            status="Failed",
            error_message=_sanitise_unexpected(e, desc.display_name),
            completion_time=timezone.now(),
        )


@shared_task
def run_both_prediction(
    public_id: str,
    kcat_key: str,
    km_key: str,
    experimental_results: list | None = None,
    canonicalize_substrates: bool = True,
    include_similarity_columns: bool = True,
    disable_gpu_precompute: bool = False,
) -> None:
    """
    Run a dual-target prediction job (kcat and KM in sequence).

    Parameters
    ----------
    public_id : str
        The job's public identifier.
    kcat_key : str
        Registry key of the kcat prediction method.
    km_key : str
        Registry key of the KM prediction method.
    experimental_results : list | None
        Pre-fetched experimental values to merge into the output, or None.
    """
    job = Job.objects.get(public_id=public_id)
    _safe_clear_gpu_precompute_status(public_id)
    job.status = "Processing"
    job.start_time = timezone.now()
    job.predictions_made = 0
    job.total_predictions = 0
    job.save(update_fields=["status", "start_time", "predictions_made", "total_predictions"])

    kcat_desc = get_method(kcat_key)
    km_desc = get_method(km_key)

    try:
        df = _load_input(job)
        _execute_both_prediction(
            job,
            kcat_desc,
            km_desc,
            df,
            experimental_results or [],
            canonicalize_substrates=canonicalize_substrates,
            include_similarity_columns=include_similarity_columns,
            disable_gpu_precompute=disable_gpu_precompute,
        )
        Job.objects.filter(pk=job.pk).update(
            status="Completed",
            completion_time=timezone.now(),
        )
        _safe_refresh_about_stats_cache()

    except PredictionError as e:
        Job.objects.filter(pk=job.pk).update(
            status="Failed",
            error_message=str(e),
            completion_time=timezone.now(),
        )

    except MemoryError:
        _handle_oom(job, f"{kcat_desc.display_name}/{km_desc.display_name}")

    except Exception as e:
        label = f"{kcat_desc.display_name}/{km_desc.display_name}"
        Job.objects.filter(pk=job.pk).update(
            status="Failed",
            error_message=_sanitise_unexpected(e, label),
            completion_time=timezone.now(),
        )


@shared_task
def run_multi_prediction(
    public_id: str,
    targets: list[str],
    methods: dict[str, str],
    experimental_results: dict | None = None,
    canonicalize_substrates: bool = True,
    include_similarity_columns: bool = True,
    disable_gpu_precompute: bool = False,
    recon_xkg: bool = False,
) -> None:
    """
    Run a multi-target prediction job.

    Parameters
    ----------
    public_id : str
        The job's public identifier.
    targets : list[str]
        Selected targets, subset of ``["kcat", "Km", "kcat/Km"]``.
    methods : dict[str, str]
        Mapping target -> method key.
    experimental_results : dict | None
        Optional pre-fetched experimental rows keyed by target.
    """
    job = Job.objects.get(public_id=public_id)
    try:
        execute_multi_prediction_job(
            public_id=public_id,
            targets=targets,
            methods=methods,
            experimental_results=experimental_results,
            canonicalize_substrates=canonicalize_substrates,
            include_similarity_columns=include_similarity_columns,
            disable_gpu_precompute=disable_gpu_precompute,
            recon_xkg=recon_xkg,
        )
    except PredictionError as e:
        mark_running_stage_failed(public_id, message=str(e))
        Job.objects.filter(pk=job.pk).update(
            status="Failed",
            error_message=str(e),
            completion_time=timezone.now(),
        )

    except MemoryError:
        mark_running_stage_failed(public_id, message="Out of memory.")
        label = "/".join(methods.get(target, target) for target in targets)
        _handle_oom(job, label)

    except Exception as e:
        mark_running_stage_failed(public_id, message=str(e))
        label = "/".join(methods.get(target, target) for target in targets)
        Job.objects.filter(pk=job.pk).update(
            status="Failed",
            error_message=_sanitise_unexpected(e, label),
            completion_time=timezone.now(),
        )


@shared_task
def run_recon_xkg_cache_prediction(
    public_id: str,
    targets: list[str],
    methods: dict[str, str],
    experimental_results: dict | None = None,
    canonicalize_substrates: bool = True,
    include_similarity_columns: bool = True,
    disable_gpu_precompute: bool = False,
) -> None:
    """
    Attempt ReconXKG full-cache completion outside the HTTP request path.

    A full cache hit is assembled with the strict cache-only executor. Any miss
    or cache-only assembly failure falls back to the normal ReconXKG task, which
    computes only cache misses and writes them back.
    """
    job = Job.objects.get(public_id=public_id)
    if job.status == "Completed":
        _log.info(
            "Skipping ReconXKG cache task — job already completed",
            extra={"event": "recon_xkg.cache_task_duplicate_skipped", "job_public_id": public_id},
        )
        return

    try:
        _log.info(
            "ReconXKG cache task started",
            extra={
                "event": "recon_xkg.cache_task_started",
                "job_public_id": public_id,
                "targets": targets,
                "methods": methods,
            },
        )
        ordered_targets = canonicalise_targets(targets)
        descriptors = {
            target: get_method(methods[target]) for target in ordered_targets
        }
        load_started = time.monotonic()
        dataframe = _load_recon_xkg_preflight_dataframe(public_id)
        _log.info(
            "ReconXKG preflight input loaded",
            extra={
                "event": "recon_xkg.preflight_input_loaded",
                "job_public_id": public_id,
                "rows": len(dataframe),
                "columns": len(dataframe.columns),
                "elapsed_ms": round((time.monotonic() - load_started) * 1000, 2),
            },
        )
        preflight = preflight_recon_xkg_cache(
            dataframe=dataframe,
            targets=ordered_targets,
            descriptors=descriptors,
            handle_long_sequences=job.handle_long_sequences,
            canonicalize_substrates=canonicalize_substrates,
            include_similarity_columns=include_similarity_columns,
            job_public_id=public_id,
        )
    except Exception as exc:
        _log.warning(
            "ReconXKG cache preflight failed; queueing normal prediction",
            extra={
                "event": "recon_xkg.cache_preflight_failed",
                "job_public_id": public_id,
                "exception_type": type(exc).__name__,
            },
            exc_info=True,
        )
        _dispatch_recon_xkg_fallback_task(
            public_id=public_id,
            targets=targets,
            methods=methods,
            experimental_results=experimental_results,
            canonicalize_substrates=canonicalize_substrates,
            include_similarity_columns=include_similarity_columns,
            disable_gpu_precompute=disable_gpu_precompute,
            reason="preflight-error",
        )
        return

    if not preflight.complete or preflight.snapshot is None:
        _log.info(
            "ReconXKG cache preflight missed; queueing normal prediction",
            extra={
                "event": "recon_xkg.cache_preflight_miss_fallback",
                "job_public_id": public_id,
                "reason": preflight.reason,
                "prediction_units": preflight.prediction_units,
                "unique_prediction_keys": preflight.unique_prediction_keys,
                "similarity_sequences": preflight.similarity_sequences,
            },
        )
        _dispatch_recon_xkg_fallback_task(
            public_id=public_id,
            targets=targets,
            methods=methods,
            experimental_results=experimental_results,
            canonicalize_substrates=canonicalize_substrates,
            include_similarity_columns=include_similarity_columns,
            disable_gpu_precompute=disable_gpu_precompute,
            reason=preflight.reason,
        )
        return

    started = time.monotonic()
    try:
        execute_multi_prediction_job(
            public_id=public_id,
            targets=ordered_targets,
            methods=methods,
            experimental_results=experimental_results or {},
            canonicalize_substrates=canonicalize_substrates,
            include_similarity_columns=include_similarity_columns,
            disable_gpu_precompute=disable_gpu_precompute,
            recon_xkg=True,
            prediction_cache_snapshot=preflight.snapshot.predictions,
            similarity_cache_snapshot=preflight.snapshot.similarities,
            cache_only=True,
        )
        _log.info(
            "ReconXKG job completed asynchronously from cache",
            extra={
                "event": "recon_xkg.async_completion",
                "job_public_id": public_id,
                "prediction_units": preflight.prediction_units,
                "unique_prediction_keys": preflight.unique_prediction_keys,
                "similarity_sequences": preflight.similarity_sequences,
                "elapsed_ms": round((time.monotonic() - started) * 1000, 2),
            },
        )
    except Exception as exc:
        _log.warning(
            "ReconXKG cache-only assembly failed; queueing normal prediction",
            extra={
                "event": "recon_xkg.async_completion_failed",
                "job_public_id": public_id,
                "exception_type": type(exc).__name__,
                "elapsed_ms": round((time.monotonic() - started) * 1000, 2),
            },
            exc_info=True,
        )
        try:
            _reset_job_after_cache_only_failure(job)
        except Exception:
            # The normal worker reinitialises status/stages on entry, so cleanup
            # failure must not prevent fallback dispatch.
            _log.warning(
                "Could not fully reset failed ReconXKG cache-only attempt",
                extra={
                    "event": "recon_xkg.async_cleanup_failed",
                    "job_public_id": public_id,
                },
                exc_info=True,
            )
        _dispatch_recon_xkg_fallback_task(
            public_id=public_id,
            targets=targets,
            methods=methods,
            experimental_results=experimental_results,
            canonicalize_substrates=canonicalize_substrates,
            include_similarity_columns=include_similarity_columns,
            disable_gpu_precompute=disable_gpu_precompute,
            reason="cache-only-assembly-failed",
        )


def _dispatch_recon_xkg_fallback_task(
    *,
    public_id: str,
    targets: list[str],
    methods: dict[str, str],
    experimental_results: dict | None,
    canonicalize_substrates: bool,
    include_similarity_columns: bool,
    disable_gpu_precompute: bool,
    reason: str,
) -> None:
    """Queue a normal ReconXKG prediction task on the model worker queue."""
    result = run_multi_prediction.apply_async(
        args=[
            public_id,
            targets,
            methods,
            experimental_results or {},
            canonicalize_substrates,
            include_similarity_columns,
            disable_gpu_precompute,
            True,
        ],
        queue="webkinpred",
    )
    _log.info(
        "ReconXKG fallback prediction task dispatched",
        extra={
            "event": "recon_xkg.fallback_task_dispatched",
            "job_public_id": public_id,
            "celery_task_id": result.id,
            "reason": reason,
            "targets": targets,
            "methods": methods,
        },
    )


def _reset_job_after_cache_only_failure(job: Job) -> None:
    """Return an attempted cache-only job to a clean queueable state."""
    JobProgressStage.objects.filter(job=job).delete()
    Job.objects.filter(pk=job.pk).update(
        status="Pending",
        start_time=None,
        completion_time=None,
        error_message="",
        output_file=None,
        total_molecules=0,
        molecules_processed=0,
        invalid_rows=0,
        total_predictions=0,
        predictions_made=0,
    )


def execute_multi_prediction_job(
    *,
    public_id: str,
    targets: list[str],
    methods: dict[str, str],
    experimental_results: dict | None = None,
    canonicalize_substrates: bool = True,
    include_similarity_columns: bool = True,
    disable_gpu_precompute: bool = False,
    recon_xkg: bool = False,
    prediction_cache_snapshot: dict[str, Any] | None = None,
    similarity_cache_snapshot: dict[
        str, tuple[float | None, float | None]
    ] | None = None,
    cache_only: bool = False,
) -> None:
    """Execute one job in-process; the Celery task and cache fast path share it."""
    job = Job.objects.get(public_id=public_id)
    if job.status == "Completed":
        _log.info(
            "Skipping duplicate task execution — job already completed",
            extra={"event": "task.duplicate_skipped", "job_public_id": public_id},
        )
        return

    ordered_targets = canonicalise_targets(targets)
    if not ordered_targets:
        raise PredictionError("No prediction targets were provided.")
    try:
        desc_by_target = {
            target: get_method(methods[target]) for target in ordered_targets
        }
    except Exception as exc:
        raise PredictionError(f"Invalid method selection: {exc}") from exc

    if not cache_only:
        _safe_clear_gpu_precompute_status(public_id)
    Job.objects.filter(pk=job.pk).update(
        status="Processing",
        start_time=timezone.now(),
        completion_time=None,
        error_message="",
        predictions_made=0,
        total_predictions=0,
    )
    job.refresh_from_db()
    progress_started = time.monotonic()
    initialise_job_progress_stages(job, ordered_targets, desc_by_target)
    _log.info(
        "Prediction progress stages initialised",
        extra={
            "event": "prediction.progress_initialised",
            "job_public_id": public_id,
            "targets": ordered_targets,
            "cache_only": cache_only,
            "elapsed_ms": round((time.monotonic() - progress_started) * 1000, 2),
        },
    )

    load_started = time.monotonic()
    df = _load_input(job)
    _log.info(
        "Prediction input loaded",
        extra={
            "event": "prediction.input_loaded",
            "job_public_id": public_id,
            "rows": len(df),
            "columns": len(df.columns),
            "cache_only": cache_only,
            "elapsed_ms": round((time.monotonic() - load_started) * 1000, 2),
        },
    )
    execute_started = time.monotonic()
    deferred_refund = _execute_multi_prediction(
        job=job,
        targets=ordered_targets,
        desc_by_target=desc_by_target,
        df=df,
        experimental_results=experimental_results or {},
        canonicalize_substrates=canonicalize_substrates,
        include_similarity_columns=include_similarity_columns,
        disable_gpu_precompute=disable_gpu_precompute,
        recon_xkg=recon_xkg,
        prediction_cache_snapshot=prediction_cache_snapshot,
        similarity_cache_snapshot=similarity_cache_snapshot,
        cache_only=cache_only,
        defer_quota_refund=cache_only,
    )
    _log.info(
        "Prediction execution completed",
        extra={
            "event": "prediction.execution_completed",
            "job_public_id": public_id,
            "cache_only": cache_only,
            "elapsed_ms": round((time.monotonic() - execute_started) * 1000, 2),
        },
    )
    if deferred_refund and cache_only:
        credit_back(_job_quota_subject(job), deferred_refund)
    Job.objects.filter(pk=job.pk).update(
        status="Completed",
        completion_time=timezone.now(),
    )
    if cache_only:
        _safe_mark_about_stats_cache_stale()
    else:
        _safe_refresh_about_stats_cache()


# ---------------------------------------------------------------------------
# Core prediction logic
# ---------------------------------------------------------------------------


def _execute_multi_prediction(
    job: Job,
    targets: list[str],
    desc_by_target: dict,
    df: pd.DataFrame,
    experimental_results: dict[str, list],
    canonicalize_substrates: bool = True,
    include_similarity_columns: bool = True,
    disable_gpu_precompute: bool = False,
    recon_xkg: bool = False,
    prediction_cache_snapshot: dict[str, Any] | None = None,
    similarity_cache_snapshot: dict[
        str, tuple[float | None, float | None]
    ] | None = None,
    cache_only: bool = False,
    defer_quota_refund: bool = False,
) -> int:
    """Run targets through the canonical reaction/child orchestration path."""
    # ReconXKG cache hit/miss accounting, aggregated across all target stages.
    cache_stats: dict[str, int] | None = (
        {"hits": 0, "misses": 0, "units": 0} if recon_xkg else None
    )
    sequence_plan_started = time.monotonic()
    sequence_plan = build_sequence_batch_plan(
        df,
        desc_by_target.values(),
        job.handle_long_sequences,
    )
    sequences = list(sequence_plan.original_sequences)
    n_rows = len(sequences)
    valid_idx = list(sequence_plan.valid_reaction_indices)
    processed_by_reaction = list(sequence_plan.processed_by_reaction)
    sequence_skips = dict(sequence_plan.skipped_reactions)
    reported_skips = dict(sequence_skips)
    target_results: dict[str, dict[str, Any]] = {}
    _log.info(
        "Prediction sequence planning completed",
        extra={
            "event": "prediction.sequence_plan",
            "job_public_id": job.public_id,
            "rows": n_rows,
            "valid_reactions": len(valid_idx),
            "sequence_children": len(sequence_plan.expansion.children),
            "recon_xkg": recon_xkg,
            "cache_only": cache_only,
            "elapsed_ms": round((time.monotonic() - sequence_plan_started) * 1000, 2),
        },
    )

    eitlem_targets = [target for target in targets if desc_by_target[target].key == "EITLEM"]
    last_eitlem_target = eitlem_targets[-1] if eitlem_targets else None
    # OmniESI and OmniESI-O2DENet share the same per-residue esm2 cache, so they
    # are coordinated as one family: embeddings are kept until the last target
    # across both methods has run.
    omniesi_family_keys = {"OmniESI", "OmniESI-O2DENet"}
    omniesi_targets = [
        target for target in targets if desc_by_target[target].key in omniesi_family_keys
    ]
    last_omniesi_target = omniesi_targets[-1] if omniesi_targets else None

    for target in targets:
        desc = desc_by_target[target]
        mark_stage_running(job.public_id, target, desc.key)
        extra_call_kwargs: dict[str, Any] = {}
        if desc.key == "EITLEM":
            extra_call_kwargs["cleanup_esm1v_embeddings"] = target == last_eitlem_target
        if desc.key in omniesi_family_keys:
            extra_call_kwargs["cleanup_embeddings_after_run"] = target == last_omniesi_target

        try:
            target_started = time.monotonic()
            result = _execute_target_batch(
                job=job,
                desc=desc,
                df=df,
                target=target,
                sequences=sequences,
                processed_by_reaction=processed_by_reaction,
                valid_reaction_indices=valid_idx,
                sequence_expansion=sequence_plan.expansion,
                experimental_results=experimental_results.get(target, []),
                canonicalize_substrates=canonicalize_substrates,
                disable_gpu_precompute=disable_gpu_precompute,
                extra_call_kwargs=extra_call_kwargs,
                recon_xkg=recon_xkg,
                cache_stats=cache_stats,
                prediction_cache_snapshot=prediction_cache_snapshot,
                cache_only=cache_only,
            )
            _log.info(
                "Prediction target execution completed",
                extra={
                    "event": "prediction.target_completed",
                    "job_public_id": job.public_id,
                    "target": target,
                    "method_key": desc.key,
                    "recon_xkg": recon_xkg,
                    "cache_only": cache_only,
                    "failed_reactions": len(result["failed_reactions"]),
                    "elapsed_ms": round((time.monotonic() - target_started) * 1000, 2),
                },
            )
        except Exception as exc:
            mark_stage_failed(job.public_id, target, desc.key, message=str(exc))
            raise

        for reaction_index, reason in result["failed_reactions"].items():
            _append_skip_reason(reported_skips, reaction_index, f"{target}: {reason}")
        target_results[target] = result
        mark_stage_completed(job.public_id, target, desc.key)

    for reaction_index, reason in sequence_skips.items():
        for result in target_results.values():
            result["preds"][reaction_index] = ""
            result["sources"][reaction_index] = reason
            result["extra"][reaction_index] = ""

    results_df = df.copy()
    preferred_cols: list[str] = []
    for target in targets:
        result = target_results[target]
        pred_col = result["output_col"]
        source_col = f"Source {target}"
        extra_col = f"Extra Info {target}"
        results_df[pred_col] = result["preds"]
        results_df[source_col] = result["sources"]
        results_df[extra_col] = result["extra"]
        preferred_cols.extend([pred_col, source_col, extra_col])

    results_df = results_df[
        preferred_cols + [column for column in results_df.columns if column not in preferred_cols]
    ]
    out_path = _output_path(job.public_id)
    write_started = time.monotonic()
    results_df.to_csv(out_path, index=False)
    _log.info(
        "Prediction output CSV written",
        extra={
            "event": "prediction.output_csv_written",
            "job_public_id": job.public_id,
            "rows": len(results_df),
            "columns": len(results_df.columns),
            "recon_xkg": recon_xkg,
            "cache_only": cache_only,
            "elapsed_ms": round((time.monotonic() - write_started) * 1000, 2),
        },
    )
    if include_similarity_columns and "kcat" in targets:
        selected_sequences = target_results.get("kcat", {}).get("selected_sequences")
        similarity_started = time.monotonic()
        append_kcat_similarity_columns_to_output_csv(
            out_path,
            desc_by_target["kcat"].key,
            recon_xkg=recon_xkg,
            cached_similarity_snapshot=similarity_cache_snapshot,
            cache_only=cache_only,
            selected_sequences_by_row=selected_sequences,
        )
        _log.info(
            "Prediction similarity columns appended",
            extra={
                "event": "prediction.similarity_columns_appended",
                "job_public_id": job.public_id,
                "method_key": desc_by_target["kcat"].key,
                "recon_xkg": recon_xkg,
                "cache_only": cache_only,
                "elapsed_ms": round((time.monotonic() - similarity_started) * 1000, 2),
            },
        )

    if cache_stats is not None:
        from api.utils.recon_xkg import log_recon_xkg_outcome

        log_recon_xkg_outcome(
            api_key_id=_job_quota_subject(job),
            public_id=job.public_id,
            row_count=n_rows,
            hit_count=cache_stats["hits"],
            miss_count=cache_stats["misses"],
        )

    fully_predicted = pd.Series(True, index=results_df.index)
    for target in targets:
        pred_col = target_results[target]["output_col"]
        fully_predicted = (
            fully_predicted & (results_df[pred_col] != "") & results_df[pred_col].notna()
        )
    processed_reactions = int(fully_predicted.sum())
    to_refund = max(0, int(job.requested_rows) - processed_reactions)
    Job.objects.filter(pk=job.pk).update(
        output_file=os.path.relpath(out_path, settings.MEDIA_ROOT),
        error_message=_build_skipped_message(reported_skips),
    )
    if to_refund and not defer_quota_refund:
        credit_back(_job_quota_subject(job), to_refund)
    return to_refund


def _execute_target_batch(
    *,
    job: Job,
    desc,
    df: pd.DataFrame,
    target: str,
    sequences: list[Any],
    processed_by_reaction: list[Any],
    valid_reaction_indices: list[int],
    experimental_results: list[dict[str, Any]],
    sequence_expansion: SequenceExpansionPlan | None = None,
    canonicalize_substrates: bool,
    disable_gpu_precompute: bool,
    extra_call_kwargs: dict[str, Any],
    recon_xkg: bool = False,
    cache_stats: dict[str, int] | None = None,
    prediction_cache_snapshot: dict[str, Any] | None = None,
    cache_only: bool = False,
) -> dict[str, Any]:
    """Execute one method/target either natively or as an expanded child batch."""
    n_rows = len(sequences)
    if sequence_expansion is None:
        sequence_expansion = SequenceExpansionPlan.build(
            sequences,
            range(len(sequences)),
            limit=max((len(str(sequence)) for sequence in sequences), default=0),
            handle_long_sequences="truncate",
        )
    sequence_plan = SequenceBatchPlan(
        original_sequences=tuple(sequences),
        processed_by_reaction=tuple(processed_by_reaction),
        valid_reaction_indices=tuple(valid_reaction_indices),
        skipped_reactions={},
        expansion=sequence_expansion,
    )
    batch = build_target_batch_plan(desc, target, df, sequence_plan)
    input_behavior = batch.input_behavior
    if batch.unit_expansion is not None:
        unit_plan = batch.unit_expansion
        unit_sequences = list(batch.sequences)
        call_kwargs = dict(batch.call_kwargs)
        call_kwargs.update(extra_call_kwargs)

        if unit_sequences:
            child_predictions, child_errors = _invoke_method_prediction(
                desc=desc,
                sequences=unit_sequences,
                public_id=job.public_id,
                target=target,
                canonicalize_substrates=canonicalize_substrates,
                disable_gpu_precompute=disable_gpu_precompute,
                recon_xkg=recon_xkg,
                cache_stats=cache_stats,
                cache_snapshot=prediction_cache_snapshot,
                cache_only=cache_only,
                **call_kwargs,
            )
        else:
            child_predictions, child_errors = [], {}
            set_stage_prediction_snapshot(
                job_public_id=job.public_id,
                target=target,
                method_key=desc.key,
                molecules_total=n_rows,
                molecules_processed=0,
                invalid_rows=0,
                predictions_total=0,
                predictions_made=0,
            )

        formatted_values: list[Any] = []
        formatted_sources: list[str] = []
        child_details: list[str] = [""] * len(unit_plan.units)
        for child_index in range(len(unit_plan.units)):
            value, source = _format_prediction_value_and_source(
                desc,
                target,
                child_predictions[child_index],
            )
            formatted_values.append(value)
            formatted_sources.append(source if value is not None else "")

        if input_behavior == "expanded_pair":
            _apply_unit_experimental_overrides(
                job=job,
                desc=desc,
                target=target,
                plan=unit_plan,
                experimental_results=experimental_results,
                values=formatted_values,
                sources=formatted_sources,
                details=child_details,
                child_errors=child_errors,
            )

        try:
            reduced = reduce_sequence_predictions(
                plan=unit_plan,
                target=target,
                child_predictions=formatted_values,
                child_sources=formatted_sources,
                child_errors=child_errors,
                child_details=child_details,
                reaction_count=n_rows,
            )
        except ValueError as exc:
            raise PredictionError(f"{desc.display_name} result mapping failed: {exc}") from exc

        return {
            "preds": reduced.predictions,
            "sources": reduced.sources,
            "extra": reduced.extra_info,
            "failed_reactions": reduced.failed_reactions,
            "selected_sequences": reduced.selected_sequences,
            "output_col": desc.output_cols[target],
        }

    if input_behavior == "expanded_pair" and batch.expansion is not None:
        plan = batch.expansion
        child_sequences = list(batch.sequences)
        call_kwargs = dict(batch.call_kwargs)
        call_kwargs.update(extra_call_kwargs)

        if child_sequences:
            child_predictions, child_errors = _invoke_method_prediction(
                desc=desc,
                sequences=child_sequences,
                public_id=job.public_id,
                target=target,
                canonicalize_substrates=canonicalize_substrates,
                disable_gpu_precompute=disable_gpu_precompute,
                recon_xkg=recon_xkg,
                cache_stats=cache_stats,
                cache_snapshot=prediction_cache_snapshot,
                cache_only=cache_only,
                **call_kwargs,
            )
        else:
            child_predictions, child_errors = [], {}
            set_stage_prediction_snapshot(
                job_public_id=job.public_id,
                target=target,
                method_key=desc.key,
                molecules_total=0,
                molecules_processed=0,
                invalid_rows=0,
                predictions_total=0,
                predictions_made=0,
            )

        formatted_values: list[Any] = []
        formatted_sources: list[str] = []
        child_details: list[str] = [""] * len(plan.children)
        for child_index in range(len(plan.children)):
            value, source = _format_prediction_value_and_source(
                desc,
                target,
                child_predictions[child_index],
            )
            formatted_values.append(value)
            formatted_sources.append(source if value is not None else "")

        _apply_expanded_experimental_overrides(
            job=job,
            desc=desc,
            target=target,
            sequences=sequences,
            plan=plan,
            experimental_results=experimental_results,
            values=formatted_values,
            sources=formatted_sources,
            details=child_details,
            child_errors=child_errors,
        )
        try:
            reduced = reduce_substrate_predictions(
                plan=plan,
                target=target,
                child_predictions=formatted_values,
                child_sources=formatted_sources,
                child_errors=child_errors,
                child_details=child_details,
                reaction_count=n_rows,
            )
        except ValueError as exc:
            raise PredictionError(f"{desc.display_name} result mapping failed: {exc}") from exc
        return {
            "preds": reduced.predictions,
            "sources": reduced.sources,
            "extra": reduced.extra_info,
            "failed_reactions": reduced.failed_reactions,
            "selected_sequences": [""] * n_rows,
            "output_col": desc.output_cols[target],
        }

    return _execute_native_target_batch(
        job=job,
        desc=desc,
        df=df,
        target=target,
        sequences=sequences,
        processed_by_reaction=processed_by_reaction,
        valid_reaction_indices=valid_reaction_indices,
        experimental_results=experimental_results,
        canonicalize_substrates=canonicalize_substrates,
        disable_gpu_precompute=disable_gpu_precompute,
        extra_call_kwargs=extra_call_kwargs,
        call_kwargs_override=dict(batch.call_kwargs),
        apply_experimental_overrides=input_behavior == "expanded_pair",
        recon_xkg=recon_xkg,
        cache_stats=cache_stats,
        prediction_cache_snapshot=prediction_cache_snapshot,
        cache_only=cache_only,
    )


def _execute_native_target_batch(
    *,
    job: Job,
    desc,
    df: pd.DataFrame,
    target: str,
    sequences: list[Any],
    processed_by_reaction: list[Any],
    valid_reaction_indices: list[int],
    experimental_results: list[dict[str, Any]],
    canonicalize_substrates: bool,
    disable_gpu_precompute: bool,
    extra_call_kwargs: dict[str, Any],
    call_kwargs_override: dict[str, list[Any]] | None = None,
    apply_experimental_overrides: bool = True,
    recon_xkg: bool = False,
    cache_stats: dict[str, int] | None = None,
    prediction_cache_snapshot: dict[str, Any] | None = None,
    cache_only: bool = False,
) -> dict[str, Any]:
    n_rows = len(sequences)
    predictions: list[Any] = [""] * n_rows
    sources: list[str] = [""] * n_rows
    extra: list[str] = [""] * n_rows
    failed_reactions: dict[int, str] = {}
    call_kwargs: dict[str, Any] = dict(call_kwargs_override or {})
    if call_kwargs_override is None:
        for column, kwarg_name in desc.col_to_kwarg.items():
            if column not in df.columns:
                raise PredictionError(
                    f"Missing column required for {desc.display_name}: {column}"
                )
            call_kwargs[kwarg_name] = [
                df[column].iloc[reaction_index] for reaction_index in valid_reaction_indices
            ]
    call_kwargs.update(desc.target_kwargs.get(target, {}))
    call_kwargs.update(extra_call_kwargs)
    processed_sequences = [processed_by_reaction[index] for index in valid_reaction_indices]

    if processed_sequences:
        subset, invalid_subset = _invoke_method_prediction(
            desc=desc,
            sequences=processed_sequences,
            public_id=job.public_id,
            target=target,
            canonicalize_substrates=canonicalize_substrates,
            disable_gpu_precompute=disable_gpu_precompute,
            recon_xkg=recon_xkg,
            cache_stats=cache_stats,
            cache_snapshot=prediction_cache_snapshot,
            cache_only=cache_only,
            **call_kwargs,
        )
        for local_index, reaction_index in enumerate(valid_reaction_indices):
            value, source = _format_prediction_value_and_source(
                desc,
                target,
                subset[local_index],
            )
            if _prediction_is_missing(value):
                reason = invalid_subset.get(local_index, "Prediction could not be made")
                sources[reaction_index] = reason
                failed_reactions[reaction_index] = reason
            else:
                predictions[reaction_index] = value
                sources[reaction_index] = source

        for local_index, reason in invalid_subset.items():
            if 0 <= local_index < len(valid_reaction_indices):
                reaction_index = valid_reaction_indices[local_index]
                predictions[reaction_index] = ""
                sources[reaction_index] = reason
                failed_reactions[reaction_index] = reason
    else:
        set_stage_prediction_snapshot(
            job_public_id=job.public_id,
            target=target,
            method_key=desc.key,
            molecules_total=n_rows,
            molecules_processed=0,
            invalid_rows=0,
            predictions_total=0,
            predictions_made=0,
        )

    if apply_experimental_overrides:
        _apply_native_experimental_overrides(
            job=job,
            desc=desc,
            target=target,
            sequences=sequences,
            experimental_results=experimental_results,
            predictions=predictions,
            sources=sources,
            extra=extra,
            failed_reactions=failed_reactions,
        )
    return {
        "preds": predictions,
        "sources": sources,
        "extra": extra,
        "failed_reactions": failed_reactions,
        "selected_sequences": [""] * n_rows,
        "output_col": desc.output_cols[target],
    }


def _apply_expanded_experimental_overrides(
    *,
    job: Job,
    desc,
    target: str,
    sequences: list[Any],
    plan: SubstrateExpansionPlan,
    experimental_results: list[dict[str, Any]],
    values: list[Any],
    sources: list[str],
    details: list[str],
    child_errors: dict[int, str],
) -> None:
    if target not in {"kcat", "Km"}:
        return
    exp_key = "kcat_value" if target == "kcat" else "km_value"
    by_position = {
        (item.get("reaction_idx"), item.get("substrate_idx")): item
        for item in experimental_results
        if item.get("found")
    }
    for child_index, child in enumerate(plan.children):
        exp = by_position.get((child.reaction_position, child.substrate_position))
        if not exp or exp_key not in exp:
            continue
        if not _experimental_sequence_matches(exp, sequences[child.reaction_position]):
            _log_experimental_mismatch(job, desc, target, child.reaction_position)
            continue
        previous = values[child_index]
        values[child_index] = exp[exp_key]
        sources[child_index] = _source(exp)
        details[child_index] = build_extra_info(exp, target, previous, desc.display_name)
        child_errors.pop(child_index, None)


def _apply_unit_experimental_overrides(
    *,
    job: Job,
    desc,
    target: str,
    plan: TargetExpansionPlan,
    experimental_results: list[dict[str, Any]],
    values: list[Any],
    sources: list[str],
    details: list[str],
    child_errors: dict[int, str],
) -> None:
    if target not in {"kcat", "Km"}:
        return
    exp_key = "kcat_value" if target == "kcat" else "km_value"
    by_position = {
        (
            item.get("reaction_idx"),
            item.get("sequence_idx"),
            item.get("substrate_idx"),
        ): item
        for item in experimental_results
        if item.get("found")
    }
    for unit_index, unit in enumerate(plan.units):
        substrate_idx = unit.substrate_position if unit.substrate_position is not None else 0
        exp = by_position.get(
            (
                unit.reaction_position,
                unit.sequence_position,
                substrate_idx,
            )
        )
        if not exp or exp_key not in exp:
            continue
        if not _experimental_sequence_matches(exp, unit.sequence):
            _log_experimental_mismatch(job, desc, target, unit.reaction_position)
            continue
        previous = values[unit_index]
        values[unit_index] = exp[exp_key]
        sources[unit_index] = _source(exp)
        details[unit_index] = build_extra_info(exp, target, previous, desc.display_name)
        child_errors.pop(unit_index, None)


def _apply_native_experimental_overrides(
    *,
    job: Job,
    desc,
    target: str,
    sequences: list[Any],
    experimental_results: list[dict[str, Any]],
    predictions: list[Any],
    sources: list[str],
    extra: list[str],
    failed_reactions: dict[int, str],
) -> None:
    if target not in {"kcat", "Km"}:
        return
    exp_key = "kcat_value" if target == "kcat" else "km_value"
    for exp in experimental_results:
        if not exp.get("found") or exp_key not in exp:
            continue
        reaction_index = exp.get("reaction_idx", exp.get("idx"))
        if not isinstance(reaction_index, int) or not 0 <= reaction_index < len(sequences):
            continue
        if not _experimental_sequence_matches(exp, sequences[reaction_index]):
            _log_experimental_mismatch(job, desc, target, reaction_index)
            continue
        previous = predictions[reaction_index]
        predictions[reaction_index] = exp[exp_key]
        sources[reaction_index] = _source(exp)
        extra[reaction_index] = build_extra_info(exp, target, previous, desc.display_name)
        failed_reactions.pop(reaction_index, None)


def _experimental_sequence_matches(exp: dict[str, Any], sequence: Any) -> bool:
    recorded = exp.get("protein_sequence")
    return recorded is None or recorded == sequence


def _log_experimental_mismatch(job: Job, desc, target: str, reaction_index: int) -> None:
    _log.warning(
        "Skipping experimental overwrite because protein sequence mismatched",
        extra={
            "event": "prediction.experimental_overwrite_mismatch",
            "job_public_id": job.public_id,
            "method_key": desc.key,
            "target": target,
            "row_index": reaction_index,
        },
    )


def _prediction_is_missing(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, str):
        return not value.strip() or value.strip().lower() in {"none", "nan"}
    try:
        return bool(pd.isna(value))
    except (TypeError, ValueError):
        return False


def _append_skip_reason(reasons: dict[int, str], row_index: int, reason: str) -> None:
    previous = reasons.get(row_index, "")
    if reason and reason not in previous:
        reasons[row_index] = f"{previous}; {reason}" if previous else reason


def _execute_prediction(
    job: Job,
    desc,
    df: pd.DataFrame,
    target: str,
    experimental_results: list,
    canonicalize_substrates: bool = True,
    include_similarity_columns: bool = True,
    disable_gpu_precompute: bool = False,
) -> None:
    """Compatibility wrapper around the canonical multi-target executor."""
    _execute_multi_prediction(
        job=job,
        targets=[target],
        desc_by_target={target: desc},
        df=df,
        experimental_results={target: experimental_results},
        canonicalize_substrates=canonicalize_substrates,
        include_similarity_columns=include_similarity_columns,
        disable_gpu_precompute=disable_gpu_precompute,
    )


def _execute_both_prediction(
    job: Job,
    kcat_desc,
    km_desc,
    df: pd.DataFrame,
    experimental_results: list,
    canonicalize_substrates: bool = True,
    include_similarity_columns: bool = True,
    disable_gpu_precompute: bool = False,
) -> None:
    """Compatibility wrapper around the canonical multi-target executor."""
    _execute_multi_prediction(
        job=job,
        targets=["kcat", "Km"],
        desc_by_target={"kcat": kcat_desc, "Km": km_desc},
        df=df,
        experimental_results={
            "kcat": [item for item in experimental_results if "kcat_value" in item],
            "Km": [item for item in experimental_results if "km_value" in item],
        },
        canonicalize_substrates=canonicalize_substrates,
        include_similarity_columns=include_similarity_columns,
        disable_gpu_precompute=disable_gpu_precompute,
    )


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _invoke_method_prediction(
    desc,
    sequences: list[str],
    public_id: str,
    target: str,
    canonicalize_substrates: bool = True,
    disable_gpu_precompute: bool = False,
    recon_xkg: bool = False,
    cache_stats: dict[str, int] | None = None,
    cache_snapshot: dict[str, Any] | None = None,
    cache_only: bool = False,
    **call_kwargs,
) -> tuple[list, dict[int, str]]:
    """
    Invoke a method, optionally served from the ReconXKG memoization store.

    Without ``recon_xkg`` this calls the real engine directly. With it, each
    positional input unit (one element of ``sequences`` plus its aligned
    substrate/products) is looked up in the cache; only misses are sent to the
    engine, fresh results are written back, and the merged list is returned in
    the original order. Either way the return contract is identical:
    ``(predictions, invalid_reasons)`` where ``invalid_reasons`` maps local
    indices (into ``sequences``) to human-readable skip reasons.
    """
    if cache_only and not recon_xkg:
        raise ReconXkgCacheOnlyMiss(
            "Cache-only prediction execution requires ReconXKG mode."
        )
    if not recon_xkg:
        return _run_method_engine(
            desc,
            sequences,
            public_id,
            target,
            canonicalize_substrates=canonicalize_substrates,
            disable_gpu_precompute=disable_gpu_precompute,
            **call_kwargs,
        )

    return _invoke_method_prediction_cached(
        desc,
        sequences,
        public_id,
        target,
        canonicalize_substrates=canonicalize_substrates,
        disable_gpu_precompute=disable_gpu_precompute,
        cache_stats=cache_stats,
        cache_snapshot=cache_snapshot,
        cache_only=cache_only,
        call_kwargs=call_kwargs,
    )


def _run_method_engine(
    desc,
    sequences: list[str],
    public_id: str,
    target: str,
    canonicalize_substrates: bool = True,
    disable_gpu_precompute: bool = False,
    **call_kwargs,
) -> tuple[list, dict[int, str]]:
    """
    Run a method's real prediction engine, either:

    1) a custom `pred_func` (legacy/current methods), or
    2) the built-in generic subprocess engine (recommended for new methods).

    Always returns (predictions, invalid_reasons) where invalid_reasons maps
    local indices (into sequences) to human-readable skip reasons.
    """
    call_kwargs = dict(call_kwargs)
    call_kwargs.setdefault("canonicalize_substrates", canonicalize_substrates)
    call_kwargs.setdefault("disable_gpu_precompute", disable_gpu_precompute)

    with log_context(job_public_id=public_id, method_key=desc.key, target=target):
        if desc.pred_func is not None:
            preds, invalid_result = desc.pred_func(
                sequences=sequences,
                public_id=public_id,
                **call_kwargs,
            )
            if isinstance(invalid_result, dict):
                invalid_reasons = invalid_result
            else:
                invalid_reasons = {
                    idx: "Prediction could not be made" for idx in (invalid_result or [])
                }
            return _validate_method_result(desc, sequences, preds, invalid_reasons)

        if desc.subprocess is not None:
            preds, invalid_reasons = run_generic_subprocess_prediction(
                desc=desc,
                sequences=sequences,
                public_id=public_id,
                target=target,
                **call_kwargs,
            )
            return _validate_method_result(desc, sequences, preds, invalid_reasons)

    raise PredictionError(f"{desc.display_name} is not configured with a prediction engine.")


def _recon_xkg_unit_keys(
    desc,
    target: str,
    sequences: list[Any],
    call_kwargs: dict[str, Any],
    canonicalize_substrates: bool,
) -> tuple[list[str | None], list[tuple[str, str, str] | None], str]:
    """
    Compute the per-unit cache lookup keys for one engine batch.

    Returns ``(keys, components, params_fp)`` where, for each input position:
      - ``keys[i]`` is the lookup key, or None if the unit is uncacheable
        because required aligned input is structurally absent;
      - ``components[i]`` is ``(sequence_sha256, substrate_canon, products_canon)``
        retained so a freshly computed value or deterministic validation failure
        can be written back without recomputing the key. Invalid chemistry uses
        a non-reversible raw fingerprint in the component.
    """
    from api.services import prediction_store as store

    return store.build_unit_keys(
        desc,
        target,
        sequences,
        call_kwargs,
        canonicalize_substrates,
    )


def _invoke_method_prediction_cached(
    desc,
    sequences: list[str],
    public_id: str,
    target: str,
    *,
    canonicalize_substrates: bool,
    disable_gpu_precompute: bool,
    cache_stats: dict[str, int] | None,
    cache_snapshot: dict[str, Any] | None,
    cache_only: bool,
    call_kwargs: dict[str, Any],
) -> tuple[list, dict[int, str]]:
    """Serve a batch from the ReconXKG store, computing only the misses."""
    from api.services import prediction_store as store

    n = len(sequences)
    model_version = getattr(desc, "model_version", "1")
    key_started = time.monotonic()
    keys, components, params_fp = _recon_xkg_unit_keys(
        desc, target, sequences, call_kwargs, canonicalize_substrates
    )
    _log.info(
        "ReconXKG prediction cache keys built",
        extra={
            "event": "recon_xkg.cache_keys_built",
            "job_public_id": public_id,
            "method_key": desc.key,
            "target": target,
            "units": n,
            "keyed_units": sum(1 for key in keys if key),
            "cache_only": cache_only,
            "elapsed_ms": round((time.monotonic() - key_started) * 1000, 2),
        },
    )

    read_started = time.monotonic()
    if cache_only:
        cached = cache_snapshot or {}
    elif cache_snapshot is not None:
        cached = cache_snapshot
    else:
        cached = store.get_many([key for key in keys if key])
    _log.info(
        "ReconXKG prediction cache source loaded",
        extra={
            "event": "recon_xkg.cache_source_loaded",
            "job_public_id": public_id,
            "method_key": desc.key,
            "target": target,
            "units": n,
            "cached_values": len(cached),
            "cache_only": cache_only,
            "source": "snapshot" if cache_only or cache_snapshot is not None else "store",
            "elapsed_ms": round((time.monotonic() - read_started) * 1000, 2),
        },
    )

    scan_started = time.monotonic()
    predictions: list[Any] = [None] * n
    invalid: dict[int, str] = {}
    miss_indices: list[int] = []
    hit_count = 0
    for i, key in enumerate(keys):
        outcome = cached.get(key) if key else None
        if not store.cached_outcome_is_valid(outcome):
            miss_indices.append(i)
            continue
        hit_count += 1
        if isinstance(outcome, store.CachedFailure):
            invalid[i] = outcome.reason
        else:
            predictions[i] = store.coerce_value(outcome)
    _log.info(
        "ReconXKG prediction cache outcomes scanned",
        extra={
            "event": "recon_xkg.cache_outcomes_scanned",
            "job_public_id": public_id,
            "method_key": desc.key,
            "target": target,
            "units": n,
            "hits": hit_count,
            "misses": len(miss_indices),
            "invalid_hits": len(invalid),
            "cache_only": cache_only,
            "elapsed_ms": round((time.monotonic() - scan_started) * 1000, 2),
        },
    )

    if cache_only and miss_indices:
        raise ReconXkgCacheOnlyMiss(
            "A prediction value was absent from the ReconXKG preflight snapshot."
        )

    if miss_indices:
        set_stage_prediction_snapshot(
            job_public_id=public_id,
            target=target,
            method_key=desc.key,
            molecules_total=n,
            molecules_processed=hit_count,
            invalid_rows=len(invalid),
            predictions_total=n,
            predictions_made=hit_count,
        )

    if miss_indices:
        # Slice only the positionally-aligned (per-row) kwargs; pass scalar
        # kwargs (target flags, cleanup toggles) through unchanged.
        aligned = set(desc.col_to_kwarg.values())
        miss_sequences = [sequences[i] for i in miss_indices]
        miss_kwargs: dict[str, Any] = {}
        for kwarg, value in call_kwargs.items():
            if kwarg in aligned and isinstance(value, (list, tuple)) and len(value) == n:
                miss_kwargs[kwarg] = [value[i] for i in miss_indices]
            else:
                miss_kwargs[kwarg] = value

        miss_preds, miss_invalid = _run_method_engine(
            desc,
            miss_sequences,
            public_id,
            target,
            canonicalize_substrates=canonicalize_substrates,
            disable_gpu_precompute=disable_gpu_precompute,
            **miss_kwargs,
        )

        rows_to_store: list[dict[str, Any]] = []
        numeric_rows_to_store = 0
        failure_rows_to_store = 0
        uncacheable_misses = 0
        for local_index, global_index in enumerate(miss_indices):
            predictions[global_index] = miss_preds[local_index]
            key = keys[global_index]
            component = components[global_index]
            if local_index in miss_invalid:
                reason = miss_invalid[local_index]
                invalid[global_index] = reason
                if (
                    key
                    and component is not None
                    and store.is_cacheable_failure_reason(reason)
                ):
                    seq_sha, sub_canon, products_canon = component
                    rows_to_store.append(
                        {
                            "lookup_key": key,
                            "target": target,
                            "method": desc.key,
                            "model_version": model_version,
                            "params_fingerprint": params_fp,
                            "sequence_sha256": seq_sha,
                            "substrate_canon": sub_canon,
                            "products_canon": products_canon,
                            "value": None,
                            "failure_reason": reason,
                        }
                    )
                    failure_rows_to_store += 1
                else:
                    uncacheable_misses += 1
                continue
            if not key or component is None:
                uncacheable_misses += 1
                continue  # uncacheable unit — never written back
            value = store.coerce_value(miss_preds[local_index])
            if value is None:
                seq_sha, sub_canon, products_canon = component
                rows_to_store.append(
                    {
                        "lookup_key": key,
                        "target": target,
                        "method": desc.key,
                        "model_version": model_version,
                        "params_fingerprint": params_fp,
                        "sequence_sha256": seq_sha,
                        "substrate_canon": sub_canon,
                        "products_canon": products_canon,
                        "value": None,
                        "failure_reason": "Prediction could not be made",
                    }
                )
                invalid[global_index] = "Prediction could not be made"
                failure_rows_to_store += 1
                continue
            seq_sha, sub_canon, products_canon = component
            rows_to_store.append(
                {
                    "lookup_key": key,
                    "target": target,
                    "method": desc.key,
                    "model_version": model_version,
                    "params_fingerprint": params_fp,
                    "sequence_sha256": seq_sha,
                    "substrate_canon": sub_canon,
                    "products_canon": products_canon,
                    "value": value,
                    "failure_reason": "",
                }
            )
            numeric_rows_to_store += 1
        rows_written = store.upsert_many(rows_to_store)
        unique_rows_prepared = len({row["lookup_key"] for row in rows_to_store})
        _log.info(
            "ReconXKG prediction cache write summary",
            extra={
                "event": "recon_xkg.cache_write_summary",
                "job_public_id": public_id,
                "method_key": desc.key,
                "target": target,
                "units": n,
                "hits": hit_count,
                "misses": len(miss_indices),
                "rows_prepared": len(rows_to_store),
                "unique_rows_prepared": unique_rows_prepared,
                "rows_written": rows_written,
                "numeric_rows": numeric_rows_to_store,
                "failure_rows": failure_rows_to_store,
                "uncacheable_misses": uncacheable_misses,
            },
        )
        if rows_written < unique_rows_prepared:
            _log.warning(
                "ReconXKG prediction cache write stored fewer rows than prepared",
                extra={
                    "event": "recon_xkg.cache_write_incomplete",
                    "job_public_id": public_id,
                    "method_key": desc.key,
                    "target": target,
                    "unique_rows_prepared": unique_rows_prepared,
                    "rows_written": rows_written,
                },
            )

    if cache_stats is not None:
        cache_stats["hits"] += hit_count
        cache_stats["misses"] += len(miss_indices)
        cache_stats["units"] += n

    # Reflect full unit counts in progress so a (partly) cached stage does not
    # under-report after the engine streamed only the miss subset.
    set_stage_prediction_snapshot(
        job_public_id=public_id,
        target=target,
        method_key=desc.key,
        molecules_total=n,
        molecules_processed=n,
        invalid_rows=len(invalid),
        predictions_total=n,
        predictions_made=n,
    )

    return predictions, invalid


def _validate_method_result(
    desc,
    sequences: list[Any],
    predictions: Any,
    invalid_reasons: Any,
) -> tuple[list, dict[int, str]]:
    """Enforce the positional prediction-engine contract at one boundary."""
    if not isinstance(predictions, (list, tuple)):
        raise PredictionError(
            f"{desc.display_name} returned an invalid prediction result."
        )
    if len(predictions) != len(sequences):
        raise PredictionError(
            f"{desc.display_name} produced {len(predictions)} prediction(s) "
            f"for {len(sequences)} input(s)."
        )

    normalised_invalid: dict[int, str] = {}
    if isinstance(invalid_reasons, dict):
        for raw_index, raw_reason in invalid_reasons.items():
            try:
                index = int(raw_index)
            except (TypeError, ValueError):
                continue
            if 0 <= index < len(sequences):
                reason = str(raw_reason or "").strip()
                normalised_invalid[index] = reason or "Prediction could not be made"
            else:
                _log.warning(
                    "Prediction method returned an out-of-range invalid index",
                    extra={
                        "event": "prediction.invalid_index_out_of_range",
                        "method_key": desc.key,
                        "row_index": index,
                    },
                )
    return list(predictions), normalised_invalid


def _map_subset_invalid_reasons(
    global_indices: list[int],
    invalid_reasons: dict[int, str],
) -> dict[int, str]:
    """Map local invalid reasons (keyed by position in sequences subset) to global row indices."""
    mapped: dict[int, str] = {}
    for local_idx, reason in invalid_reasons.items():
        try:
            idx = int(local_idx)
        except (TypeError, ValueError):
            continue
        if 0 <= idx < len(global_indices):
            mapped[global_indices[idx]] = reason
    return mapped


def _build_skipped_message(skipped_reasons: dict[int, str]) -> str:
    """Serialize per-row skip reasons as a JSON array grouped by reason."""
    if not skipped_reasons:
        return ""
    groups: dict[str, list[int]] = {}
    for idx, reason in skipped_reasons.items():
        groups.setdefault(reason, []).append(idx)
    return json.dumps([{"rows": sorted(rows), "reason": reason} for reason, rows in groups.items()])


def _load_input(job: Job) -> pd.DataFrame:
    """Read the job's input CSV, crediting back quota on failure."""
    path = os.path.join(settings.MEDIA_ROOT, "jobs", str(job.public_id), "input.csv")
    df = safe_read_csv(path, _job_quota_subject(job), job.requested_rows)
    if df is None:
        raise PredictionError(
            "The uploaded CSV file could not be read. "
            "Please ensure it is a valid CSV and try again."
        )
    return df


def _load_recon_xkg_preflight_dataframe(public_id: str) -> pd.DataFrame:
    """Read input for cache preflight without quota side effects."""
    path = os.path.join(settings.MEDIA_ROOT, "jobs", str(public_id), "input.csv")
    return pd.read_csv(path)


def _output_path(public_id: str) -> str:
    return os.path.join(settings.MEDIA_ROOT, "jobs", str(public_id), "output.csv")


def _handle_oom(job: Job, label: str) -> None:
    """Mark job as failed with an out-of-memory message and credit back quota."""
    msg = (
        f"{label} prediction terminated due to insufficient memory. "
        "Try reducing the number of rows or the sequence lengths."
    )
    Job.objects.filter(pk=job.pk).update(
        status="Failed",
        error_message=msg,
        completion_time=timezone.now(),
    )
    credit_back(_job_quota_subject(job), job.requested_rows)


def _job_quota_subject(job: Job) -> str:
    """
    Resolve the quota subject used for this job.

    Older rows may not have ``quota_subject`` populated; fallback to the legacy
    IP-based key in that case.
    """
    return job.quota_subject or job.ip_address


def _sanitise_unexpected(exc: Exception, label: str) -> str:
    """
    Convert an unexpected (non-PredictionError) exception to a user-facing
    message, stripping internal paths and stack traces.
    """
    import re

    msg = str(exc)
    # If the message contains file paths or the word "Traceback", it's too
    # technical for a user.  Replace with a generic fallback.
    if re.search(r"/[a-z_/]+\.[a-z]+", msg, re.IGNORECASE) or "Traceback" in msg:
        return (
            f"{label} prediction encountered an unexpected error. "
            "Please verify your input and try again."
        )
    return msg or (
        f"{label} prediction encountered an unexpected error. "
        "Please verify your input and try again."
    )
