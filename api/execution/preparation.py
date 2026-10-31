"""Bound eager background planning so execution retains storage headroom."""

import shutil
from pathlib import Path

from django.conf import settings
from django.db.models import Exists, OuterRef, Q

from api.execution.models import ExecutionSpec, WorkStage
from api.execution.resources import GIB


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
    limit = getattr(settings, "EXECUTION_PREPARED_BACKGROUND_LIMIT", 128)
    if limit > 0:
        attention = WorkStage.objects.filter(execution_id=OuterRef("pk")).filter(
            Q(state="held") | Q(waiting_reason="profiling_required")
        )
        prepared = (
            ExecutionSpec.objects.filter(
                background=True,
                paused=False,
                job__status__in=["Pending", "Processing"],
                stages__key="prepare",
                stages__state="completed",
            )
            .exclude(pk=execution.pk)
            .annotate(needs_attention=Exists(attention))
            .filter(needs_attention=False)
            .values_list("pk", flat=True)[:limit]
        )
        if len(prepared) >= limit:
            return "preparation_backlog"
    return ""
