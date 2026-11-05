"""Durable background selectors and live foreground dependency inheritance."""

from django.db import connection
from django.db.models import Q
from django.utils import timezone

from api.execution.models import ExecutionSpec, SchedulingRule
from api.models import ApiKey, Job
from api.utils.quotas import _parse_api_key_subject, api_key_quota_subject

POLICY_LOCK = 481726394


def lock_policy(*, shared=False):
    if connection.vendor != "postgresql":
        raise RuntimeError("Scheduling policies require PostgreSQL")
    function = "pg_advisory_xact_lock_shared" if shared else "pg_advisory_xact_lock"
    with connection.cursor() as cursor:
        cursor.execute(f"SELECT {function}(%s)", [POLICY_LOCK])


def matching_jobs():
    selectors = Q(pk__in=[])
    prefixes = Q(pk__in=[])
    for rule in SchedulingRule.objects.filter(enabled=True):
        if rule.kind == "ip":
            selectors |= Q(ip_address=rule.value)
        elif rule.kind == "api_key_prefix":
            prefixes |= Q(key__startswith=rule.value)
    subjects = [
        api_key_quota_subject(pk)
        for pk in ApiKey.objects.filter(prefixes).values_list("pk", flat=True)
    ]
    return Job.objects.filter(selectors | Q(quota_subject__in=subjects))


def background_for_job(job):
    lock_policy(shared=True)
    key_id = _parse_api_key_subject(job.quota_subject)
    for rule in SchedulingRule.objects.filter(enabled=True):
        if rule.kind == "ip" and job.ip_address == rule.value:
            return True
        if rule.kind == "api_key_prefix" and key_id is not None:
            if ApiKey.objects.filter(pk=key_id, key__startswith=rule.value).exists():
                return True
    return False


def apply_background_rules():
    """Caller holds the policy lock and transaction; never changes requests/quota."""
    lock_policy()
    specs = ExecutionSpec.objects.filter(job__isnull=False).exclude(
        job__status__in=["Completed", "Failed"]
    )
    selected = matching_jobs().values("pk")
    lower = specs.filter(job_id__in=selected).exclude(background=True).update(background=True)
    restore = specs.exclude(job_id__in=selected).exclude(background=False).update(background=False)
    return {
        "lowered": lower,
        "restored": restore,
        "background_unfinished": specs.filter(background=True).count(),
        "normal_unfinished": specs.filter(background=False).count(),
    }


def foreground_dependencies():
    """Background-owned prerequisites required by any live normal request.

    Derive this from the current graph, avoiding stale materialized promotions
    and per-job traversals in every admission request. The few live normal
    requests are resolved first and passed explicitly: the planner cannot see
    that only a handful qualify and otherwise scanned the whole stage table.
    """
    normal = list(
        ExecutionSpec.objects.filter(background=False, paused=False)
        .exclude(job__status__in=["Completed", "Failed"])
        .values_list("pk", flat=True)
    )
    if not normal:
        return set()
    with connection.cursor() as cursor:
        cursor.execute(
            """
            WITH RECURSIVE needed(id) AS (
                SELECT d.to_workstage_id
                FROM api_workstage c
                JOIN api_workstage_dependencies d ON d.from_workstage_id = c.id
                JOIN api_workstage p ON p.id = d.to_workstage_id
                JOIN api_executionspec pe ON pe.id = p.execution_id
                WHERE c.execution_id = ANY(%s)
                  AND c.state NOT IN ('completed', 'canceled')
                  AND p.state NOT IN ('completed', 'canceled') AND pe.background
                UNION
                SELECT d.to_workstage_id
                FROM needed n
                JOIN api_workstage_dependencies d ON d.from_workstage_id = n.id
                JOIN api_workstage p ON p.id = d.to_workstage_id
                WHERE p.state NOT IN ('completed', 'canceled')
            ) SELECT id FROM needed
            """,
            [normal],
        )
        return {row[0] for row in cursor.fetchall()}


def higher_preparation(execution):
    execution.refresh_from_db(fields=["background", "priority"])
    higher = Q(background=execution.background, priority__gt=execution.priority)
    if execution.background:
        higher |= Q(background=False)
    return ExecutionSpec.objects.filter(
        higher,
        paused=False,
        stages__key="prepare",
        stages__state="ready",
        stages__available_at__lte=timezone.now(),
    ).exclude(pk=execution.pk)
