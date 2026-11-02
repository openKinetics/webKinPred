"""A shared producer keeps the queue age of its oldest unfinished consumer."""

from django.db import connection, transaction
from django.db.models import Exists, OuterRef

from api.execution.models import ExecutionSpec, WorkStage


@transaction.atomic
def refresh_dependency_order(after=0, limit=8, *, execution_id=None):
    """Visit a bounded page of prepared requests; return the next page cursor.

    No input manifests or immutable specifications are loaded. Queue age only
    moves earlier during a producer's lifetime: finishing one consumer must not
    repeatedly demote work still needed by another. Explicit priorities and the
    foreground/background class remain separate and outrank this timestamp.
    """
    if connection.vendor != "postgresql":
        return 0, 0
    prepared = WorkStage.objects.filter(
        execution_id=OuterRef("pk"), key="prepare", state="completed"
    )
    requests = (
        ExecutionSpec.objects.filter(paused=False, pk__gt=after)
        .exclude(job__status__in=["Completed", "Failed"])
        .annotate(prepared=Exists(prepared))
        .filter(prepared=True)
        .order_by("pk")
    )
    if execution_id is not None:
        requests = requests.filter(pk=execution_id)
    ids = list(requests.values_list("pk", flat=True)[: max(1, min(limit, 32))])
    if not ids:
        return 0, 0
    with connection.cursor() as cursor:
        cursor.execute(
            "SET LOCAL jit = off; SET LOCAL max_parallel_workers_per_gather = 0; "
            "SET LOCAL work_mem = '16MB'; SET LOCAL statement_timeout = '3000ms'; "
            "SET LOCAL lock_timeout = '500ms'"
        )
        cursor.execute(
            """
            WITH RECURSIVE needed(id, queued_at) AS (
                SELECT s.id, COALESCE(j.submission_time, e.created_at)
                FROM api_workstage s
                JOIN api_executionspec e ON e.id = s.execution_id
                LEFT JOIN api_job j ON j.job_id = e.job_id
                WHERE e.id = ANY(%s) AND s.state NOT IN ('completed', 'canceled')
                UNION
                SELECT d.to_workstage_id, n.queued_at
                FROM needed n
                JOIN api_workstage_dependencies d ON d.from_workstage_id = n.id
                JOIN api_workstage p ON p.id = d.to_workstage_id
                WHERE p.state NOT IN ('completed', 'canceled')
            ), oldest AS (
                SELECT id, MIN(queued_at) AS queued_at FROM needed GROUP BY id
            )
            UPDATE api_workstage s SET oldest_consumer_at = o.queued_at
            FROM oldest o, api_executionspec e LEFT JOIN api_job j ON j.job_id = e.job_id
            WHERE s.id = o.id AND s.execution_id = e.id
              AND s.state NOT IN ('completed', 'canceled')
              AND o.queued_at < COALESCE(s.oldest_consumer_at, j.submission_time, e.created_at)
            """,
            [ids],
        )
        changed = cursor.rowcount
    return ids[-1], changed
