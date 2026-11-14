"""Bound eager background planning so execution retains storage headroom."""

import shutil
from pathlib import Path

from django.conf import settings
from django.db.models import Exists, OuterRef, Q

from api.execution.models import ExecutionSpec, WorkStage
from api.execution.resources import GIB


def cpu_only(execution):
    # DLKcat's indivisible prediction forward needs no separate encoder. Do not
    # infer this from incomplete descriptor lists or from CUDA availability.
    return execution.specification.get("targets") == ["kcat"] and execution.specification.get(
        "methods"
    ) == {"kcat": "DLKcat"}


def cpu_candidates():
    # Cheap lookahead over job metadata; waiting_reason verifies the immutable
    # specification before granting this preparation class.
    return Q(job__prediction_type="kcat", job__kcat_method="DLKcat")


def preparation_candidates(rules, now, limit=4):
    """Requests whose preparation stage is due, in preparation order.

    A request has one stage per key, so the join yields one row per request.
    DISTINCT over these rows sorted every job specification on each controller
    tick and spilled to temporary files (11 MB/s on 14 November).
    """
    from django.db.models.functions import Coalesce

    return (
        ExecutionSpec.objects.filter(
            paused=False,
            stages__key="prepare",
            stages__state="ready",
            stages__available_at__lte=now,
        )
        .filter(rules)
        .select_related("job")
        .order_by("background", "-priority", Coalesce("job__submission_time", "created_at"), "pk")[
            :limit
        ]
    )


def prepared_requests(*, cpu, exclude=None):
    attention = WorkStage.objects.filter(execution_id=OuterRef("pk")).filter(
        Q(state="held") | Q(waiting_reason="profiling_required")
    )
    requests = (
        ExecutionSpec.objects.filter(
            background=True,
            paused=False,
            job__status__in=["Pending", "Processing"],
            stages__key="prepare",
            stages__state="completed",
        )
        .exclude(pk=exclude)
        .annotate(needs_attention=Exists(attention))
        .filter(needs_attention=False)
    )
    return requests.filter(cpu_candidates()) if cpu else requests.exclude(cpu_candidates())


def window_full(*, cpu, exclude=None):
    name = "EXECUTION_PREPARED_CPU_LIMIT" if cpu else "EXECUTION_PREPARED_BACKGROUND_LIMIT"
    limit = getattr(settings, name, 32 if cpu else 128)
    return (
        limit > 0
        and len(prepared_requests(cpu=cpu, exclude=exclude).values_list("pk", flat=True)[:limit])
        >= limit
    )


def candidate_rules():
    """Blocked GPU planning must not hide independent CPU work behind it."""
    gpu_full, cpu_full = window_full(cpu=False), window_full(cpu=True)
    if gpu_full and cpu_full:
        return Q(background=False)
    if gpu_full:
        return Q(background=False) | cpu_candidates()
    if cpu_full:
        return Q(background=False) | ~cpu_candidates()
    return Q()


def waiting_reason(execution, blob_root):
    # Normal requests retain strict precedence, including newly arriving requests.
    if not execution.background:
        return ""
    root = Path(blob_root)
    while not root.exists():
        root = root.parent
    headroom = (
        settings.EXECUTION_HOSTS.get("production", {}).get("budgets", {}).get("disk_headroom", 0)
    )
    reserve = getattr(settings, "EXECUTION_PREPARATION_RESERVE_BYTES", 2 * GIB)
    if shutil.disk_usage(root).free < headroom + reserve:
        return "waiting_disk"
    if window_full(cpu=cpu_only(execution), exclude=execution.pk):
        return "preparation_backlog"
    return ""
