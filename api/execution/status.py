from __future__ import annotations

from django.conf import settings
from django.utils import timezone

from api.execution.models import ExecutionAttempt, ExecutionSpec, WorkStage


def execution_status(job):
    if not settings.EXECUTION_V2_ENABLED:
        return {}
    execution = ExecutionSpec.objects.filter(job=job).first()
    if not execution:
        return {}
    stages = list(execution.stages.all())
    waiting = list(stages)
    seen = {stage.pk for stage in stages}
    frontier = [stage.pk for stage in stages if stage.state != "completed"]
    while frontier:
        dependencies = list(
            WorkStage.objects.filter(dependants__pk__in=frontier)
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
    elif any(s.state == "running" for s in stages):
        state = "running"
    elif any(s.failures for s in stages if s.state != "completed"):
        state = "retrying"
    elif any(s.kind == "embedding" and s.state != "completed" for s in waiting):
        state = "waiting_gpu"
    else:
        state = "waiting_cpu"
    now = timezone.now()
    intervals = sorted(
        (a.started_at, min(a.finished_at or now, a.lease_until))
        for a in ExecutionAttempt.objects.filter(stage__execution=execution)
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
