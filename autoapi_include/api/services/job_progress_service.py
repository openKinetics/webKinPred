"""
Stage-based job progress tracking helpers.

This service keeps per-target progress in ``JobProgressStage`` rows while
maintaining legacy job-level counters for backward compatibility.
"""

from __future__ import annotations

from typing import Any

from django.db.models import F
from django.utils import timezone

from api.models import Job, JobProgressStage


def _stage_queryset(
    *,
    job_public_id: str,
    target: str | None = None,
    method_key: str | None = None,
):
    qs = JobProgressStage.objects.filter(job__public_id=job_public_id)
    if target is not None:
        qs = qs.filter(target=target)
    if method_key is not None:
        qs = qs.filter(method_key=method_key)
    return qs


def _embedding_defaults(method_key: str, target: str) -> dict[str, Any]:
    if method_key == "DLKcat":
        return {
            "embedding_enabled": False,
            "embedding_state": "not_required",
            "embedding_method_key": "",
            "embedding_target": "",
        }
    return {
        "embedding_enabled": True,
        "embedding_state": "pending",
        "embedding_method_key": method_key,
        "embedding_target": target,
    }


def initialise_job_progress_stages(
    job: Job,
    targets: list[str],
    desc_by_target: dict[str, Any],
) -> None:
    """
    Replace any existing stage rows for this job with fresh pending rows.
    """
    JobProgressStage.objects.filter(job=job).delete()

    stages: list[JobProgressStage] = []
    for stage_index, target in enumerate(targets):
        desc = desc_by_target.get(target)
        method_key = getattr(desc, "key", "") or ""
        method_display_name = getattr(desc, "display_name", "") or method_key
        stages.append(
            JobProgressStage(
                job=job,
                stage_index=stage_index,
                target=target,
                method_key=method_key,
                method_display_name=method_display_name,
                status="pending",
                **_embedding_defaults(method_key, target),
            )
        )

    if stages:
        JobProgressStage.objects.bulk_create(stages)

    Job.objects.filter(pk=job.pk).update(
        total_molecules=0,
        molecules_processed=0,
        invalid_rows=0,
        total_predictions=0,
        predictions_made=0,
    )


def mark_stage_running(job_public_id: str, target: str, method_key: str | None = None) -> None:
    now = timezone.now()
    qs = _stage_queryset(job_public_id=job_public_id, target=target, method_key=method_key)
    qs.update(
        status="running",
        started_at=now,
        completed_at=None,
        message="",
        updated_at=now,
    )


def mark_stage_completed(job_public_id: str, target: str, method_key: str | None = None) -> None:
    now = timezone.now()
    stage = _stage_queryset(
        job_public_id=job_public_id,
        target=target,
        method_key=method_key,
    ).first()
    if stage is None:
        return

    updates: dict[str, Any] = {
        "status": "completed",
        "completed_at": now,
        "updated_at": now,
    }
    if stage.embedding_enabled and stage.embedding_state in {"", "pending", "running"}:
        updates["embedding_state"] = "done"
    JobProgressStage.objects.filter(pk=stage.pk).update(**updates)


def mark_stage_failed(
    job_public_id: str,
    target: str,
    method_key: str | None = None,
    message: str | None = None,
) -> None:
    now = timezone.now()
    updates: dict[str, Any] = {
        "status": "failed",
        "completed_at": now,
        "updated_at": now,
    }
    if message:
        updates["message"] = str(message)
    _stage_queryset(
        job_public_id=job_public_id,
        target=target,
        method_key=method_key,
    ).update(**updates)


def mark_running_stage_failed(job_public_id: str, message: str | None = None) -> None:
    if JobProgressStage.objects.filter(job__public_id=job_public_id, status="failed").exists():
        return

    now = timezone.now()
    updates: dict[str, Any] = {
        "status": "failed",
        "completed_at": now,
        "updated_at": now,
    }
    if message:
        updates["message"] = str(message)
    running_qs = JobProgressStage.objects.filter(job__public_id=job_public_id, status="running")
    if running_qs.exists():
        running_qs.update(**updates)
        return

    pending_qs = JobProgressStage.objects.filter(
        job__public_id=job_public_id,
        status="pending",
    ).order_by("stage_index")
    pending_stage = pending_qs.first()
    if pending_stage is not None:
        JobProgressStage.objects.filter(pk=pending_stage.pk).update(**updates)
        return

    last_stage = (
        JobProgressStage.objects.filter(job__public_id=job_public_id)
        .order_by("-stage_index")
        .first()
    )
    if last_stage is not None:
        JobProgressStage.objects.filter(pk=last_stage.pk).update(**updates)


def reset_stage_prediction_metrics(
    *,
    job_public_id: str,
    target: str,
    method_key: str,
    total_rows: int,
) -> None:
    now = timezone.now()
    qs = _stage_queryset(job_public_id=job_public_id, target=target, method_key=method_key)
    stage = qs.first()
    if _has_cached_prediction_progress_context(stage, total_rows):
        qs.update(
            status="running",
            started_at=now,
            completed_at=None,
            molecules_total=int(stage.molecules_total),
            molecules_processed=int(stage.molecules_processed),
            invalid_rows=int(stage.invalid_rows),
            predictions_total=int(stage.predictions_total),
            predictions_made=int(stage.predictions_made),
            updated_at=now,
        )
        Job.objects.filter(public_id=job_public_id).update(
            total_molecules=int(stage.molecules_total),
            molecules_processed=int(stage.molecules_processed),
            invalid_rows=int(stage.invalid_rows),
            total_predictions=int(stage.predictions_total),
            predictions_made=int(stage.predictions_made),
        )
        return

    qs.update(
        status="running",
        started_at=now,
        completed_at=None,
        molecules_total=int(total_rows),
        molecules_processed=0,
        invalid_rows=0,
        predictions_total=0,
        predictions_made=0,
        updated_at=now,
    )
    Job.objects.filter(public_id=job_public_id).update(
        total_molecules=int(total_rows),
        molecules_processed=0,
        invalid_rows=0,
        total_predictions=0,
        predictions_made=0,
    )


def increment_stage_validation(
    *,
    job_public_id: str,
    target: str,
    method_key: str,
    processed_inc: int = 1,
    invalid_inc: int = 0,
) -> None:
    if processed_inc == 0 and invalid_inc == 0:
        return
    now = timezone.now()
    _stage_queryset(job_public_id=job_public_id, target=target, method_key=method_key).update(
        molecules_processed=F("molecules_processed") + int(processed_inc),
        invalid_rows=F("invalid_rows") + int(invalid_inc),
        updated_at=now,
    )
    Job.objects.filter(public_id=job_public_id).update(
        molecules_processed=F("molecules_processed") + int(processed_inc),
        invalid_rows=F("invalid_rows") + int(invalid_inc),
    )


def set_stage_prediction_total(
    *,
    job_public_id: str,
    target: str,
    method_key: str,
    total_predictions: int,
) -> None:
    total_predictions = int(total_predictions)
    now = timezone.now()
    qs = _stage_queryset(job_public_id=job_public_id, target=target, method_key=method_key)
    stage = qs.first()
    predictions_made = 0
    if _has_cached_prediction_progress_context(stage, total_predictions):
        full_total = int(stage.predictions_total)
        predictions_made = min(
            full_total,
            max(int(stage.predictions_made), full_total - total_predictions),
        )
        total_predictions = full_total

    qs.update(
        predictions_total=int(total_predictions),
        predictions_made=int(predictions_made),
        updated_at=now,
    )
    Job.objects.filter(public_id=job_public_id).update(
        total_predictions=int(total_predictions),
        predictions_made=int(predictions_made),
    )


def set_stage_prediction_progress(
    *,
    job_public_id: str,
    target: str,
    method_key: str,
    done: int,
    total: int | None = None,
) -> None:
    done = int(done)
    if total is not None:
        total = int(total)
        stage = _stage_queryset(
            job_public_id=job_public_id,
            target=target,
            method_key=method_key,
        ).first()
        done, total = _scale_cached_prediction_progress(stage, done, total)

    now = timezone.now()
    stage_updates: dict[str, Any] = {
        "predictions_made": done,
        "updated_at": now,
    }
    job_updates: dict[str, Any] = {
        "predictions_made": done,
    }
    if total is not None:
        stage_updates["predictions_total"] = int(total)
        job_updates["total_predictions"] = int(total)

    _stage_queryset(job_public_id=job_public_id, target=target, method_key=method_key).update(
        **stage_updates
    )
    Job.objects.filter(public_id=job_public_id).update(**job_updates)


def _has_cached_prediction_progress_context(stage: Any, incoming_total: int) -> bool:
    """Return true when a partial ReconXKG cache hit seeded full-stage progress."""
    if stage is None:
        return False
    try:
        full_total = int(stage.predictions_total)
        current_done = int(stage.predictions_made)
        molecules_total = int(stage.molecules_total)
        molecules_processed = int(stage.molecules_processed)
        incoming = int(incoming_total)
    except (TypeError, ValueError):
        return False
    if incoming < 0 or full_total <= incoming:
        return False
    offset = max(0, full_total - incoming)
    if current_done >= offset:
        return True
    return molecules_total == full_total and molecules_processed >= full_total


def _scale_cached_prediction_progress(
    stage: Any,
    done: int,
    total: int,
) -> tuple[int, int]:
    """Map miss-only subprocess progress onto the full cached+miss denominator."""
    if not _has_cached_prediction_progress_context(stage, total):
        return max(0, int(done)), max(0, int(total))

    full_total = int(stage.predictions_total)
    miss_total = max(0, int(total))
    offset = max(0, full_total - miss_total)
    scaled_done = min(full_total, offset + max(0, int(done)))
    return scaled_done, full_total


def set_stage_prediction_snapshot(
    *,
    job_public_id: str,
    target: str,
    method_key: str,
    molecules_total: int,
    molecules_processed: int,
    invalid_rows: int,
    predictions_total: int,
    predictions_made: int,
) -> None:
    now = timezone.now()
    _stage_queryset(job_public_id=job_public_id, target=target, method_key=method_key).update(
        molecules_total=int(molecules_total),
        molecules_processed=int(molecules_processed),
        invalid_rows=int(invalid_rows),
        predictions_total=int(predictions_total),
        predictions_made=int(predictions_made),
        updated_at=now,
    )
    Job.objects.filter(public_id=job_public_id).update(
        total_molecules=int(molecules_total),
        molecules_processed=int(molecules_processed),
        invalid_rows=int(invalid_rows),
        total_predictions=int(predictions_total),
        predictions_made=int(predictions_made),
    )


def set_stage_embedding_progress(
    *,
    job_public_id: str,
    target: str,
    method_key: str,
    enabled: bool,
    state: str,
    total: int,
    cached_already: int,
    need_computation: int,
    computed: int,
    remaining: int,
) -> None:
    now = timezone.now()
    _stage_queryset(job_public_id=job_public_id, target=target, method_key=method_key).update(
        embedding_enabled=bool(enabled),
        embedding_state=(state or ""),
        embedding_method_key=method_key,
        embedding_target=target,
        embedding_total=int(total),
        embedding_cached_already=int(cached_already),
        embedding_need_computation=int(need_computation),
        embedding_computed=int(computed),
        embedding_remaining=int(remaining),
        updated_at=now,
    )


def set_stage_embedding_state(
    *,
    job_public_id: str,
    target: str,
    method_key: str,
    state: str,
) -> None:
    now = timezone.now()
    enabled = state != "not_required"
    _stage_queryset(job_public_id=job_public_id, target=target, method_key=method_key).update(
        embedding_enabled=enabled,
        embedding_state=(state or ""),
        embedding_method_key=method_key if enabled else "",
        embedding_target=target if enabled else "",
        updated_at=now,
    )


def get_progress_stages(job: Job) -> list[dict[str, Any]]:
    stages = list(job.progress_stages.all().order_by("stage_index"))
    payload: list[dict[str, Any]] = []
    for stage in stages:
        prediction = {
            "molecules_total": int(stage.molecules_total),
            "molecules_processed": int(stage.molecules_processed),
            "invalid_rows": int(stage.invalid_rows),
            "predictions_total": int(stage.predictions_total),
            "predictions_made": int(stage.predictions_made),
            "moleculesTotal": int(stage.molecules_total),
            "moleculesProcessed": int(stage.molecules_processed),
            "invalidRows": int(stage.invalid_rows),
            "predictionsTotal": int(stage.predictions_total),
            "predictionsMade": int(stage.predictions_made),
        }
        embedding = {
            "enabled": bool(stage.embedding_enabled),
            "state": stage.embedding_state
            or ("not_required" if not stage.embedding_enabled else "pending"),
            "method_key": stage.embedding_method_key or stage.method_key,
            "methodKey": stage.embedding_method_key or stage.method_key,
            "target": stage.embedding_target or stage.target,
            "total": int(stage.embedding_total),
            "cached_already": int(stage.embedding_cached_already),
            "need_computation": int(stage.embedding_need_computation),
            "computed": int(stage.embedding_computed),
            "remaining": int(stage.embedding_remaining),
            "cachedAlready": int(stage.embedding_cached_already),
            "needComputation": int(stage.embedding_need_computation),
        }

        payload.append(
            {
                "index": int(stage.stage_index),
                "target": stage.target,
                "method_key": stage.method_key,
                "methodKey": stage.method_key,
                "method_name": stage.method_display_name or stage.method_key,
                "methodName": stage.method_display_name or stage.method_key,
                "status": stage.status,
                "started_at": stage.started_at.isoformat() if stage.started_at else None,
                "completed_at": stage.completed_at.isoformat() if stage.completed_at else None,
                "message": stage.message or "",
                "prediction": prediction,
                "embedding": embedding,
            }
        )
    return payload


def get_progress_summary(job: Job) -> dict[str, Any]:
    stages = get_progress_stages(job)
    active_stage_index = None
    for stage in stages:
        if stage["status"] == "running":
            active_stage_index = stage["index"]
            break
    if active_stage_index is None:
        for stage in stages:
            if stage["status"] == "pending":
                active_stage_index = stage["index"]
                break
    if active_stage_index is None and stages:
        active_stage_index = stages[-1]["index"]

    completed_stage_count = sum(1 for stage in stages if stage["status"] == "completed")
    return {
        "stages": stages,
        "active_stage_index": active_stage_index,
        "completed_stage_count": completed_stage_count,
        "total_stage_count": len(stages),
    }


def get_active_stage_embedding(job: Job) -> dict[str, Any] | None:
    stages = get_progress_stages(job)
    if not stages:
        return None

    active_stage = next((s for s in stages if s["status"] == "running"), None)
    if active_stage is None:
        active_stage = stages[-1]

    embedding = active_stage.get("embedding") or {}
    if not embedding:
        return None
    return embedding
