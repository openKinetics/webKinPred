"""Aggregate operational metrics; never include user IDs, inputs, or ownership tokens."""

import hmac
import json
import math
import time
from datetime import timedelta

from django.conf import settings
from django.db import OperationalError, connection, transaction
from django.db.models import BigIntegerField, Count, Min, Sum
from django.db.models.functions import Cast
from django.http import HttpResponse
from django.utils import timezone

from api.execution.models import (
    Artifact,
    ExecutionAttempt,
    ExecutionOutbox,
    ExecutorHost,
    ResourceProfile,
    WorkStage,
)
from api.execution.resources import LIVE_STATES
from api.models import Job


def render():
    started = time.monotonic()
    now = timezone.now()
    lines = []
    declared = set()

    def emit(name, value, **labels):
        if not isinstance(value, (int, float)) or not math.isfinite(value):
            return
        name = "webkinpred_execution_" + name
        if name not in declared:
            lines.append(f"# TYPE {name} gauge")
            declared.add(name)
        suffix = (
            "{" + ",".join(f"{k}={json.dumps(str(v))}" for k, v in sorted(labels.items())) + "}"
            if labels
            else ""
        )
        lines.append(f"{name}{suffix} {value}")

    for group in WorkStage.objects.values("method", "kind", "state", "waiting_reason").annotate(
        count=Count("pk"), oldest=Min("execution__job__submission_time"), retries=Sum("failures")
    ):
        labels = {key: group[key] for key in ("method", "kind", "state", "waiting_reason")}
        emit("stages", group["count"], **labels)
        emit("retries", group["retries"] or 0, **labels)
        if group["oldest"] and group["state"] not in {"completed", "canceled"}:
            emit(
                "oldest_job_age_seconds", max(0, (now - group["oldest"]).total_seconds()), **labels
            )
    emit(
        "unprofiled_stages",
        WorkStage.objects.filter(state="ready", profile__isnull=True)
        .exclude(kind="prepare")
        .count(),
    )
    # Retain retired profile records for attempt provenance without keeping a
    # warning active after their unfinished work has moved to a replacement.
    unfinished_profiles = (
        WorkStage.objects.exclude(state__in=["completed", "canceled"])
        .exclude(execution__job__status__in=["Completed", "Failed"])
        .values("profile_id")
    )
    emit(
        "suspended_profiles",
        ResourceProfile.objects.filter(suspended=True, pk__in=unfinished_profiles).count(),
    )
    emit("uncertain_attempts", ExecutionAttempt.objects.filter(state="uncertain").count())
    emit(
        "expired_live_attempts",
        ExecutionAttempt.objects.filter(state__in=LIVE_STATES, lease_until__lte=now).count(),
    )
    emit("oom_attempts", ExecutionAttempt.objects.filter(outcome__category="oom").count())
    emit("outbox_pending", ExecutionOutbox.objects.filter(published_at__isnull=True).count())
    completed = Job.objects.filter(
        execution__isnull=False, status="Completed", completion_time__gte=now - timedelta(hours=1)
    )
    emit("jobs_completed_last_hour", completed.count())
    transfers = {}
    for attempt in (
        ExecutionAttempt.objects.filter(finished_at__gte=now - timedelta(hours=1))
        .values("host_id", "outcome__execution_metrics")
        .iterator()
    ):
        totals = transfers.setdefault(
            attempt["host_id"],
            {
                "transfer_seconds": 0,
                "transferred_bytes": 0,
                "local_cache_hits": 0,
                "artifact_requests": 0,
            },
        )
        for key in totals:
            value = (attempt["outcome__execution_metrics"] or {}).get(key, 0)
            if isinstance(value, (int, float)) and math.isfinite(value) and value >= 0:
                totals[key] += value
    for host, totals in transfers.items():
        for key, value in totals.items():
            emit(key + "_last_hour", value, host=host)
        emit(
            "cache_hit_ratio_last_hour",
            totals["local_cache_hits"] / max(1, totals["artifact_requests"]),
            host=host,
        )
    for stage in (
        WorkStage.objects.filter(
            kind="predict", state="completed", completed_at__gte=now - timedelta(hours=1)
        )
        .values("method", "target")
        .annotate(units=Sum(Cast("payload__shape__units", BigIntegerField())))
    ):
        emit(
            "prediction_units_completed_last_hour",
            stage["units"] or 0,
            method=stage["method"],
            target=stage["target"],
        )
    for host in ExecutorHost.objects.all():
        labels = {"host": host.pk}
        emit(
            "host_heartbeat_age_seconds",
            (now - host.heartbeat_at).total_seconds() if host.heartbeat_at else 10**9,
            **labels,
        )
        emit("host_paused", int(host.paused), **labels)
        emit("host_reconciled", int(host.reconciled), **labels)
        active = list(host.attempts.filter(state__in=LIVE_STATES).values("id", "resources"))
        emit("live_attempts", len(active), **labels)
        for resource in ("cpu", "ram", "disk", "vram"):
            reserved = sum(value["resources"].get(resource, 0) for value in active)
            observed = sum(
                value.get(resource, 0) for value in host.telemetry.get("attempts", {}).values()
            )
            emit("reserved", reserved, resource=resource, **labels)
            emit("observed", observed, resource=resource, **labels)
            if resource in host.budgets:
                emit("budget", host.budgets[resource], resource=resource, **labels)
            emit(
                "reservation_breaches",
                sum(
                    host.telemetry.get("attempts", {}).get(str(value["id"]), {}).get(resource, 0)
                    > value["resources"].get(resource, 0)
                    for value in active
                ),
                resource=resource,
                **labels,
            )
        for key in (
            "ram_available",
            "disk_free",
            "vram_free",
            "vram_total",
            "gpu_utilization",
            "cpu_external",
        ):
            if key in host.telemetry:
                emit("host_" + key, host.telemetry[key], **labels)
        if "vram_total" in host.telemetry and "vram_free" in host.telemetry:
            owned = sum(
                value.get("vram", 0) for value in host.telemetry.get("attempts", {}).values()
            )
            emit(
                "unattributed_vram",
                max(0, host.telemetry["vram_total"] - host.telemetry["vram_free"] - owned),
                **labels,
            )
        emit("disk_headroom", host.budgets.get("disk_headroom", 0), **labels)
        emit("legacy_consumers", len(host.telemetry.get("legacy_consumers", [])), **labels)
        # This optional inventory scan can touch millions of imported cache
        # records. A slow scan must not monopolize the web/heartbeat workers.
        # On the live catalogue it always exceeded its deadline, costing database
        # time on every scrape for no data, so it now runs only when enabled.
        if not getattr(settings, "EXECUTION_METRICS_ARTIFACT_INVENTORY", False):
            emit("artifact_inventory_available", 0, **labels)
            continue
        try:
            with transaction.atomic():
                size = (
                    Artifact.objects.filter(locations__host=host, locations__ready=True).aggregate(
                        total=Sum("size")
                    )["total"]
                    or 0
                )
        except OperationalError as error:
            if getattr(error.__cause__, "pgcode", None) != "57014":
                raise
            emit("artifact_inventory_available", 0, **labels)
        else:
            emit("artifact_inventory_available", 1, **labels)
            emit("artifact_bytes", size, **labels)
    emit("scrape_duration_seconds", time.monotonic() - started)
    return "\n".join(lines) + "\n"


def metrics(request):
    if request.method != "GET":
        return HttpResponse(status=405)
    token = getattr(settings, "EXECUTION_METRICS_TOKEN", "")
    if not token or not hmac.compare_digest(
        request.headers.get("Authorization", ""), "Bearer " + token
    ):
        return HttpResponse(status=401)
    try:
        with transaction.atomic():
            if connection.vendor == "postgresql":
                with connection.cursor() as cursor:
                    cursor.execute("SET LOCAL jit = off")
                    cursor.execute("SET LOCAL statement_timeout = '1000ms'")
                    cursor.execute("SET LOCAL max_parallel_workers_per_gather = 0")
                    cursor.execute("SELECT pg_try_advisory_xact_lock(481726395)")
                    if not cursor.fetchone()[0]:
                        return HttpResponse(status=503, headers={"Retry-After": "15"})
            body = render()
    except OperationalError:
        return HttpResponse(status=503, headers={"Retry-After": "15"})
    return HttpResponse(body, content_type="text/plain; version=0.0.4; charset=utf-8")
