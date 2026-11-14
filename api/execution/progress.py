"""Project concurrent stage events into the existing target progress contract."""

import time

from django.db import OperationalError, connection, models, transaction
from django.db.models.expressions import RawSQL
from django.utils import timezone

from api.models import Job, JobProgressStage

# A large request's projection waits at least LARGE_COOLDOWN seconds, and at
# least SCAN_SHARE times its last scan: on 12 November the 40,130-row LJEc7C9
# took 6-9 s per scan (shared embedding links joined to the artifact table),
# so a fixed 15 s kept production's database reading about half of the time.
LARGE_COOLDOWN = 15
SCAN_SHARE = 9


def _cooldown(large, scan_seconds):
    return max(LARGE_COOLDOWN, SCAN_SHARE * scan_seconds) if large else 0


def _spacing(scan_seconds, background):
    """Seconds until a large request's next projection.

    The cooldown bounds one request. A background request also waits for every
    other large projection pending, so together they keep within about
    1/SCAN_SHARE of the projector: on 14 November some 40 background requests of
    5,000 rows, each within its own cooldown, kept one connection projecting for
    over half of the time. Normal requests keep their own cooldown.
    """
    spacing = _cooldown(True, scan_seconds)
    if background:
        from api.execution.models import ExecutionOutbox

        competing = sum(
            1
            for cooldown in ExecutionOutbox.objects.filter(
                published_at__isnull=True, event__has_key="progress_execution_ids"
            ).values_list("event__cooldown", flat=True)
            if cooldown
        )
        spacing *= max(1, competing)
    return spacing


@transaction.atomic
def request_refresh(execution, *, force=False):
    """Coalesce expensive progress work independently of execution leases."""
    large = bool(execution.job_id and execution.job.requested_rows > 1024)
    if execution.job_id and not large:
        return refresh(execution)
    shared = None if execution.job_id else execution.specification.get("shared_protein_batch")
    if not execution.job_id and not shared:
        return
    from api.execution.models import ExecutionOutbox

    owners = (
        [execution.pk] if execution.job_id else [row["id"] for row in shared["source_requests"]]
    )
    event, _ = ExecutionOutbox.objects.get_or_create(
        key=f"progress:{execution.pk}", defaults={"event": {}}
    )
    event = ExecutionOutbox.objects.select_for_update().get(pk=event.pk)
    previous = event.event
    scan_seconds = previous.get("scan_seconds", 0)
    # Keep the spacing the projector last chose for this request.
    cooldown = max(_cooldown(large, scan_seconds), previous.get("cooldown", 0)) if large else 0
    forced = force or previous.get("force", False)
    event.event = {
        "progress_execution_ids": list(
            dict.fromkeys([*previous.get("progress_execution_ids", []), *owners])
        ),
        "generation": previous.get("generation", 0) + 1,
        "cooldown": cooldown,
        "scan_seconds": scan_seconds,
        "last_projected_at": previous.get("last_projected_at", 0),
        "not_before": 0 if forced else previous.get("last_projected_at", 0) + cooldown,
        "force": forced,
    }
    event.published_at = None
    event.save(update_fields=["event", "published_at"])


def _project_one():
    from api.execution.models import ExecutionOutbox, ExecutionSpec

    # Do not hold the outbox row during the scan: incoming progress can mark it
    # dirty without waiting. Generation checking below prevents lost updates.
    event = (
        ExecutionOutbox.objects.filter(
            published_at__isnull=True,
            event__has_key="progress_execution_ids",
        )
        .filter(
            models.Q(event__not_before__isnull=True)
            | models.Q(event__not_before__lte=timezone.now().timestamp())
        )
        .order_by("pk")
        .first()
    )
    if event is None:
        return False
    owners = event.event["progress_execution_ids"]
    generation = event.event.get("generation", 0)
    began = time.monotonic()
    if owners:
        owner = (
            ExecutionSpec.objects.filter(pk=owners[0])
            .only("id", "job_id", "background")
            .annotate(projected_targets=models.F("specification__targets"))
            .first()
        )
        if owner and owner.job_id:
            try:
                with transaction.atomic():
                    if connection.vendor == "postgresql":
                        with connection.cursor() as cursor:
                            cursor.execute(
                                "SET LOCAL jit = off; "
                                "SET LOCAL max_parallel_workers_per_gather = 0; "
                                "SET LOCAL lock_timeout = '250ms'; "
                                "SET LOCAL statement_timeout = '10000ms'"
                            )
                    refresh(owner, scan_before_lock=True)
            except OperationalError:
                # Otherwise a scan that keeps timing out reruns at every control
                # tick, back to back. Wait as after a scan of that length.
                with transaction.atomic():
                    current = ExecutionOutbox.objects.select_for_update().get(pk=event.pk)
                    scan = time.monotonic() - began
                    if current.event.get("cooldown"):
                        now = timezone.now().timestamp()
                        spacing = _spacing(scan, owner.background)
                        current.event.update(
                            scan_seconds=round(scan, 3),
                            cooldown=spacing,
                            last_projected_at=now,
                            not_before=now + spacing,
                        )
                        current.save(update_fields=["event"])
                raise
    scan = time.monotonic() - began
    with transaction.atomic():
        current = ExecutionOutbox.objects.select_for_update().get(pk=event.pk)
        payload = current.event
        unchanged = payload.get("generation", 0) == generation
        remaining = owners[1:] if unchanged else payload["progress_execution_ids"]
        now = timezone.now()
        spacing = 0
        if payload.get("cooldown"):
            spacing = _spacing(scan, bool(owners and owner and owner.background))
        payload.update(
            progress_execution_ids=remaining,
            generation=payload.get("generation", 0) + 1,
            scan_seconds=round(scan, 3),
            cooldown=spacing,
            last_projected_at=now.timestamp(),
            not_before=0 if unchanged or payload.get("force") else now.timestamp() + spacing,
            force=False if unchanged else payload.get("force", False),
        )
        current.event = payload
        current.published_at = None if remaining else now
        current.save(update_fields=["event", "published_at"])
    return True


def project_pending(limit=4, seconds=0.5):
    deadline = time.monotonic() + seconds
    count = 0
    while count < limit and time.monotonic() < deadline:
        try:
            projected = _project_one()
        except OperationalError:
            # Projection is optional; its transaction has rolled back and the
            # durable event remains pending. Do not delay control reconciliation.
            break
        if not projected:
            break
        count += 1
    return count


def shared_embedding_summaries(execution):
    """Count only this job's referenced outputs, without loading shared manifests."""
    from api.execution.models import ArtifactDependency

    rows = (
        ArtifactDependency.objects.filter(
            stage__execution=execution, artifact__producer__kind="embedding"
        )
        .exclude(stage__state="canceled")
        .exclude(artifact__producer__execution=execution)
        .values(target=models.F("stage__target"), state=models.F("artifact__producer__state"))
        .annotate(output_count=models.Count("artifact_id", distinct=True))
    )
    return [
        {**row, "kind": "embedding", "progress": 0, "has_rows": False, "units": 0} for row in rows
    ]


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


def refresh(execution, *, scan_before_lock=False):
    if not execution.job_id:
        shared = execution.specification.get("shared_protein_batch")
        if shared:
            from api.execution.models import ExecutionSpec

            # Source membership is bounded by the coalescing scan. Read targets
            # alone: complete immutable requests can contain large input context.
            for owner in (
                ExecutionSpec.objects.filter(pk__in=[r["id"] for r in shared["source_requests"]])
                .only("id", "job_id")
                .annotate(projected_targets=models.F("specification__targets"))
                .order_by("job_id")
            ):
                refresh(owner)
        return
    # Background projections scan before taking the job lock. A newer summary
    # or final publication invalidates that snapshot rather than regressing it.
    captured_at = timezone.now()
    stages = None
    if scan_before_lock:
        stages = stage_summaries(execution)
        stages.extend(shared_embedding_summaries(execution))
    with transaction.atomic():
        job = Job.objects.select_for_update().get(pk=execution.job_id)
        if scan_before_lock:
            if (
                job.completion_time and job.completion_time > captured_at
            ) or JobProgressStage.objects.filter(job=job, updated_at__gt=captured_at).exists():
                return
        else:
            stages = stage_summaries(execution)
            stages.extend(shared_embedding_summaries(execution))
        total, done = 0, 0
        targets = getattr(execution, "projected_targets", None)
        if targets is None:
            targets = execution.specification.get("targets", [])
        for target in targets:
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
