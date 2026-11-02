"""Project concurrent stage events into the existing target progress contract."""

from django.db import connection, models, transaction
from django.db.models.expressions import RawSQL
from django.utils import timezone

from api.models import Job, JobProgressStage


def stage_summaries(execution):
    stages = execution.stages.exclude(state="canceled")
    if connection.vendor == "postgresql":
        # Payloads can contain thousands of inputs and immutable manifests.
        # Read only their counters on each five-second progress heartbeat.
        return list(
            stages.annotate(
                has_rows=RawSQL("payload ? 'rows'", [], output_field=models.BooleanField()),
                units=RawSQL(
                    "COALESCE((payload->'shape'->>'units')::bigint, 0)",
                    [],
                    output_field=models.BigIntegerField(),
                ),
                output_count=RawSQL(
                    "CASE WHEN jsonb_typeof(payload->'outputs') = 'array' "
                    "THEN jsonb_array_length(payload->'outputs') ELSE 0 END",
                    [],
                    output_field=models.BigIntegerField(),
                ),
            ).values("target", "kind", "state", "progress", "has_rows", "units", "output_count")
        )
    return [
        dict(
            target=s.target,
            kind=s.kind,
            state=s.state,
            progress=s.progress,
            has_rows="rows" in s.payload,
            units=s.payload.get("shape", {}).get("units", 0),
            output_count=len(s.payload.get("outputs", [])),
        )
        for s in stages
    ]


def refresh(execution):
    if not execution.job_id:
        return
    # Serialize target summaries across completions/heartbeats from both hosts.
    with transaction.atomic():
        job = Job.objects.select_for_update().get(pk=execution.job_id)
        stages = stage_summaries(execution)
        total, done = 0, 0
        for target in execution.specification.get("targets", []):
            predictions = [
                s
                for s in stages
                if s["target"] == target and s["kind"] == "predict" and s["has_rows"]
            ]
            count = sum(s["units"] for s in predictions)
            completed = sum(
                s["units"] if s["state"] == "completed" else min(s["progress"], s["units"])
                for s in predictions
            )
            embeddings = [s for s in stages if s["target"] == target and s["kind"] == "embedding"]
            finished = bool(predictions) and all(s["state"] == "completed" for s in predictions)
            running = any(s["state"] == "running" for s in predictions + embeddings)
            embedding_total = sum(s["output_count"] for s in embeddings)
            embedding_done = sum(s["output_count"] for s in embeddings if s["state"] == "completed")
            JobProgressStage.objects.filter(job=job, target=target).update(
                status="completed" if finished else "running" if running else "pending",
                predictions_total=count,
                predictions_made=completed,
                molecules_total=count,
                molecules_processed=completed,
                embedding_state="done"
                if all(s["state"] == "completed" for s in embeddings)
                else "running"
                if any(s["state"] == "running" for s in embeddings)
                else "pending",
                embedding_enabled=bool(embeddings),
                embedding_total=embedding_total,
                embedding_need_computation=embedding_total,
                embedding_computed=embedding_done,
                embedding_remaining=embedding_total - embedding_done,
                updated_at=timezone.now(),
            )
            total += count
            done += completed
        Job.objects.filter(pk=job.pk).update(total_predictions=total, predictions_made=done)
