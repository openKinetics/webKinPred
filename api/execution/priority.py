"""Operator priorities affect admission order, never resource or ownership checks."""

from collections import defaultdict

from django.db import connection, transaction

from api.execution.control import require_postgres
from api.execution.models import ExecutionSpec, WorkStage

PRIORITY_LOCK = 481726393
MAX_PRIORITY = 1_000_000


def lock_priorities():
    require_postgres()
    with connection.cursor() as cursor:
        cursor.execute("SELECT pg_advisory_xact_lock(%s)", [PRIORITY_LOCK])


@transaction.atomic
def refresh_stage_priorities():
    """Promote unfinished shared prerequisites; remove boosts no longer needed."""
    lock_priorities()
    desired = {}
    for execution in (
        ExecutionSpec.objects.filter(priority__gt=0, paused=False)
        .exclude(job__status__in=["Completed", "Failed"])
        .order_by("-priority", "pk")
    ):
        frontier = set(
            execution.stages.exclude(state__in=["completed", "canceled"]).values_list(
                "pk", flat=True
            )
        )
        visited = set()
        while frontier:
            current = frontier - visited
            if not current:
                break
            visited.update(current)
            for ident in current:
                desired[ident] = max(desired.get(ident, 0), execution.priority)
            frontier = set(
                WorkStage.objects.filter(dependants__pk__in=current)
                .exclude(state__in=["completed", "canceled"])
                .values_list("pk", flat=True)
            )
    existing = dict(WorkStage.objects.filter(priority__gt=0).values_list("pk", "priority"))
    updates = defaultdict(list)
    for ident in existing.keys() | desired.keys():
        value = desired.get(ident, 0)
        if existing.get(ident, 0) != value:
            updates[value].append(ident)
    for priority, ids in updates.items():
        # Bound SQL parameter and row-lock batches even for large dependency graphs.
        for offset in range(0, len(ids), 500):
            WorkStage.objects.filter(pk__in=ids[offset : offset + 500]).update(priority=priority)
    return sum(map(len, updates.values()))
