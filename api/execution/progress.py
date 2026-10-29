"""Project concurrent stage events into the existing target progress contract."""

from django.db import transaction
from django.utils import timezone

from api.models import Job, JobProgressStage


def refresh(execution):
    if not execution.job_id:
        return
    # Serialize target summaries across completions/heartbeats from both hosts.
    with transaction.atomic():
        job = Job.objects.select_for_update().get(pk=execution.job_id)
        stages = list(execution.stages.all())
        total, done = 0, 0
        for target in execution.specification.get("targets", []):
            predictions = [
                s
                for s in stages
                if s.target == target and s.kind == "predict" and "rows" in s.payload
            ]
            count = sum(s.payload["shape"].get("units", 0) for s in predictions)
            completed = sum(
                s.payload["shape"].get("units", 0)
                if s.state == "completed"
                else min(s.progress, s.payload["shape"].get("units", 0))
                for s in predictions
            )
            embeddings = [s for s in stages if s.target == target and s.kind == "embedding"]
            finished = bool(predictions) and all(s.state == "completed" for s in predictions)
            running = any(s.state == "running" for s in predictions + embeddings)
            embedding_total = sum(len(s.payload.get("outputs", [])) for s in embeddings)
            embedding_done = sum(
                len(s.payload.get("outputs", [])) for s in embeddings if s.state == "completed"
            )
            JobProgressStage.objects.filter(job=job, target=target).update(
                status="completed" if finished else "running" if running else "pending",
                predictions_total=count,
                predictions_made=completed,
                molecules_total=count,
                molecules_processed=completed,
                embedding_state="done"
                if all(s.state == "completed" for s in embeddings)
                else "running"
                if any(s.state == "running" for s in embeddings)
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
