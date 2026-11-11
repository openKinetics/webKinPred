from __future__ import annotations

import threading
import time
from bisect import bisect_left

from django.conf import settings
from django.db import connection
from django.db.models import Exists, OuterRef, Q
from django.utils import timezone

from api.execution.models import ExecutionAttempt, ExecutionSpec, WorkStage

_queue_snapshot = None
_queue_snapshot_lock = threading.Lock()
_QUEUE_SNAPSHOT_LIMIT = 20_000


def waiting_jobs():
    held = WorkStage.objects.filter(execution_id=OuterRef("pk")).filter(
        Q(state="held") | Q(waiting_reason="profiling_required")
    )
    return (
        ExecutionSpec.objects.filter(
            paused=False, job__status="Pending", job__start_time__isnull=True
        )
        .annotate(needs_attention=Exists(held))
        .filter(needs_attention=False)
    )


def waiting_queue_snapshot():
    """A short-lived display hint; execution ownership stays in PostgreSQL."""
    global _queue_snapshot
    database = (connection.alias, connection.settings_dict["NAME"])
    with _queue_snapshot_lock:
        if (
            _queue_snapshot is not None
            and _queue_snapshot["database"] == database
            and _queue_snapshot["expires"] > time.monotonic()
        ):
            return _queue_snapshot
        jobs = tuple(
            waiting_jobs()
            .order_by("background", "-priority", "job__submission_time", "job_id")
            .values_list("background", "priority", "job__submission_time", "job_id")[
                : _QUEUE_SNAPSHOT_LIMIT + 1
            ]
        )
        jobs = tuple(
            (background, -priority, submitted, ident)
            for background, priority, submitted, ident in jobs
        )
        truncated = len(jobs) > _QUEUE_SNAPSHOT_LIMIT
        total = waiting_jobs().count() if truncated else len(jobs)
        _queue_snapshot = {
            "database": database,
            "expires": time.monotonic() + 10,
            "updated_at": timezone.now().isoformat(),
            "jobs": jobs[:_QUEUE_SNAPSHOT_LIMIT],
            "total": total,
            "truncated": truncated,
        }
        return _queue_snapshot


def queue_estimate(job, state, priority=0, background=False):
    if job.status != "Pending" or job.start_time or state not in {"waiting_cpu", "waiting_gpu"}:
        return None
    snapshot = waiting_queue_snapshot()
    key = (background, -priority, job.submission_time, job.pk)
    jobs = snapshot["jobs"]
    ahead = bisect_left(jobs, key)
    present = ahead < len(jobs) and jobs[ahead] == key
    if snapshot["truncated"] and key > jobs[-1]:
        ahead = (
            waiting_jobs()
            .filter(
                Q(background__lt=background)
                | Q(background=background, priority__gt=priority)
                | Q(
                    background=background,
                    priority=priority,
                    job__submission_time__lt=job.submission_time,
                )
                | Q(
                    background=background,
                    priority=priority,
                    job__submission_time=job.submission_time,
                    job_id__lt=job.pk,
                )
            )
            .count()
        )
    return {
        "position": ahead + 1,
        "jobs_ahead": ahead,
        "waiting_jobs": max(snapshot["total"], ahead + 1)
        if snapshot["truncated"]
        else len(jobs) + (0 if present else 1),
        "approximate": True,
        "basis": "priority_and_submission_order"
        if background or priority or any(row[0] or row[1] < 0 for row in jobs)
        else "submission_order",
        "updated_at": snapshot["updated_at"],
    }


def execution_status(job):
    if not settings.EXECUTION_V2_ENABLED:
        return {}
    execution = (
        ExecutionSpec.objects.only("id", "paused", "priority", "background", "created_at")
        .filter(job=job)
        .first()
    )
    if not execution:
        return {}
    counters = ("id", "kind", "state", "waiting_reason", "failures")
    stages = list(execution.stages.exclude(state="canceled").only("execution", *counters))
    waiting = list(stages)
    seen = {stage.pk for stage in stages}
    frontier = [stage.pk for stage in stages if stage.state != "completed"]
    while frontier:
        dependencies = list(
            WorkStage.objects.only(*counters)
            .filter(dependants__pk__in=frontier)
            .exclude(state="completed")
            .exclude(pk__in=seen)
            .distinct()
        )
        frontier = [stage.pk for stage in dependencies]
        seen.update(frontier)
        waiting.extend(dependencies)
    if job.status in {"Completed", "Failed"}:
        state = job.status.lower()
    elif execution.paused:
        state = "paused"
    elif any(s.state == "held" or s.waiting_reason == "profiling_required" for s in waiting):
        state = "needs_attention"
    elif any(s.state == "running" for s in waiting):
        state = "running"
    elif any(s.failures for s in stages if s.state != "completed"):
        state = "retrying"
    elif any(s.kind == "embedding" and s.state != "completed" for s in waiting):
        state = "waiting_gpu"
    else:
        state = "waiting_cpu"
    now = timezone.now()
    # Resolve the request's stages and the producers of their inputs with indexed
    # lookups. A single UNION subquery let the planner sequentially scan the
    # whole stage table for every public status poll. Attaching an input links
    # its consumer to the producer stage (cohort and recovery rewrites move those
    # links), so read the stage graph instead of joining every input artifact:
    # for a 6,707-row request that join read 134,460 rows per poll (about 19 s).
    own_stages = list(WorkStage.objects.filter(execution=execution).values_list("pk", flat=True))
    related_stages = set(own_stages) | set(
        WorkStage.dependencies.through.objects.filter(
            from_workstage_id__in=own_stages
        ).values_list("to_workstage_id", flat=True)
    )
    intervals = sorted(
        (max(started, execution.created_at), min(finished or now, lease))
        for started, finished, lease in ExecutionAttempt.objects.filter(
            stage_id__in=related_stages
        ).values_list("started_at", "finished_at", "lease_until")
        if min(finished or now, lease) > execution.created_at
    )
    merged = []
    for start, end in intervals:
        if merged and start <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(end, merged[-1][1]))
        else:
            merged.append((start, end))
    active = sum(max(0, (end - start).total_seconds()) for start, end in merged)
    # Legacy timing fields retain submission/start history. These measured durations
    # cover this executor's lifetime; unknown legacy compute time is not called waiting.
    elapsed = max(0, ((job.completion_time or now) - execution.created_at).total_seconds())
    return {
        "execution_state": state,
        "queue_estimate": queue_estimate(job, state, execution.priority, execution.background),
        "scheduling_priority": execution.priority,
        "scheduling_class": "background" if execution.background else "normal",
        "waiting_reason": {
            "paused": "maintenance",
            "needs_attention": "operator_attention",
            "waiting_gpu": "gpu_capacity",
            "waiting_cpu": "processing_capacity",
            "retrying": "recovering_work",
        }.get(state),
        "active_execution_seconds": int(active),
        "resource_wait_seconds": int(max(0, elapsed - active)),
        "work_stages_completed": sum(s.state == "completed" for s in stages),
        "work_stages_total": len(stages),
    }
