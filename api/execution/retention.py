"""Fenced collection of managed intermediates; legacy/project files stay protected."""

from collections import defaultdict
from datetime import timedelta

from django.db import OperationalError, connection, transaction
from django.db.models import Exists, OuterRef, Q
from django.utils import timezone

from api.execution.models import (
    Artifact,
    ArtifactCleanup,
    ArtifactDependency,
    ArtifactLocation,
    ArtifactSweep,
    ExecutionAttempt,
    ExecutorHost,
)
from api.execution.resources import LIVE_STATES


def release_consumed_residues(stage):
    """Enqueue hints at consumer commit; ownership checks still run at deletion."""
    groups = defaultdict(set)
    for ident, host in ArtifactLocation.objects.filter(
        artifact__dependencies__stage=stage, artifact__retention="transient", ready=True
    ).values_list("pk", "host_id"):
        groups[host].add(ident)
    for host, ids in groups.items():
        sweep, _ = ArtifactSweep.objects.select_for_update().get_or_create(host_id=host)
        sweep.pending = list(dict.fromkeys([*sweep.pending, *sorted(ids)]))[-2048:]
        sweep.save(update_fields=["pending"])


def claim_cleanup(host_id, *, version=1):
    try:
        return _claim_cleanup(host_id, version=version)
    except OperationalError as error:
        cause = error.__cause__
        if getattr(cause, "sqlstate", None) == "57014" or getattr(cause, "pgcode", None) == "57014":
            # An optional scan must not monopolize a host lock or API worker.
            # The savepoint/transaction has rolled back before this response.
            return None
        raise


@transaction.atomic
def _claim_cleanup(host_id, *, version=1):
    # Cleanup is optional. Never occupy an HTTP worker waiting behind graph
    # preparation, or block the executor's five-second supervision deadline.
    with connection.cursor() as cursor:
        cursor.execute("SET LOCAL statement_timeout = 1500")
        cursor.execute("SET LOCAL jit = off")
        cursor.execute("SET LOCAL max_parallel_workers_per_gather = 0")
        cursor.execute("SELECT pg_try_advisory_xact_lock(%s)", [481726392])
        if not cursor.fetchone()[0]:
            return None
    host = ExecutorHost.objects.select_for_update(skip_locked=True).filter(pk=host_id).first()
    if host is None:
        return None
    if not host.reconciled:
        return None
    existing = ArtifactCleanup.objects.filter(host=host, completed_at__isnull=True).first()
    if existing:
        if existing.attempt_ids and version < 2:
            return None  # An older collector cannot acknowledge scratch deletion.
        return {
            "id": str(existing.pk),
            "token": str(existing.token),
            "blobs": existing.blobs,
            "attempt_ids": existing.attempt_ids,
        }
    target = host.budgets.get(
        "cache_free_target", host.budgets.get("disk_headroom", 0) + 10 * 1024**3
    )
    free = host.telemetry.get("disk_free")
    pressure = free is not None and free < target
    cutoff = timezone.now() - timedelta(seconds=host.budgets.get("cache_min_age_seconds", 3600))
    protected_aliases = (
        Artifact.objects.filter(checksum=OuterRef("artifact__checksum"))
        .annotate(
            unfinished=Exists(
                ArtifactDependency.objects.filter(artifact_id=OuterRef("pk")).filter(
                    ~Q(stage__state__in=["completed", "canceled"])
                    | Q(stage__attempts__state__in=LIVE_STATES)
                )
            )
        )
        .filter(
            Q(regenerable=False)
            | Q(retention__in=["protected", "permanent"])
            | Q(unfinished=True)
            | Q(producer__attempts__state__in=LIVE_STATES)
        )
    )
    # Full residue matrices are work files, not a reusable long-term cache.
    # Only classified, newly generated platform artifacts in the managed blob
    # store qualify; standalone/project requests and unknown legacy files do not.
    policy = Q(artifact__retention="transient", artifact__producer__execution__job__isnull=False)
    if pressure:
        policy |= Q(artifact__retention="cache", verified_at__lt=cutoff)
    sweep, _ = ArtifactSweep.objects.select_for_update().get_or_create(host=host)
    base = ArtifactLocation.objects.filter(
        policy, host=host, ready=True, artifact__regenerable=True
    )
    locations = ArtifactLocation.objects.filter(host=host, ready=True)
    page = list(
        locations.filter(pk__gt=sweep.cursor).order_by("pk").values_list("pk", flat=True)[:512]
    )
    if not page and sweep.cursor:
        page = list(locations.order_by("pk").values_list("pk", flat=True)[:512])
    hinted = sweep.pending[:128]
    sweep.cursor = max(page) if page else 0
    candidates = list(
        base.filter(pk__in=[*page, *hinted])
        .annotate(protected=Exists(protected_aliases))
        .filter(protected=False)
        .order_by("pk")
        .values_list("pk", "artifact__checksum", "artifact__size")
    )
    blobs, selected, total, deferred = [], set(), 0, []
    for ident, checksum, size in candidates:
        if not checksum or checksum in selected:
            continue
        if total >= 4 * 1024**3 or len(blobs) >= 64:
            deferred.append(ident)
            continue
        selected.add(checksum)
        blobs.append({"checksum": checksum, "size": size})
        total += size
    sweep.pending = list(dict.fromkeys([*deferred, *sweep.pending[128:]]))[:2048]
    sweep.save(update_fields=["cursor", "pending"])
    attempt_ids = []
    if version >= 2:
        live_stage = ExecutionAttempt.objects.filter(
            stage_id=OuterRef("stage_id"), state__in=LIVE_STATES
        )
        attempt_ids = [
            str(ident)
            for ident in ExecutionAttempt.objects.filter(
                host=host,
                state__in=["completed", "stopped"],
                stage__state="completed",
                scratch_released_at__isnull=True,
            )
            .annotate(has_live=Exists(live_stage))
            .filter(has_live=False)
            .order_by("started_at")
            .values_list("pk", flat=True)[:32]
        ]
    if not blobs and not attempt_ids:
        return None
    cleanup = ArtifactCleanup.objects.create(host=host, blobs=blobs, attempt_ids=attempt_ids)
    # Claims on this host remain blocked until deletion is acknowledged. New
    # consumers see missing locations and schedule verified copies/regeneration.
    ArtifactLocation.objects.filter(host=host, artifact__checksum__in=selected).update(ready=False)
    return {
        "id": str(cleanup.pk),
        "token": str(cleanup.token),
        "blobs": blobs,
        "attempt_ids": attempt_ids,
    }


@transaction.atomic
def finish_cleanup(host_id, data):
    ExecutorHost.objects.select_for_update().get(pk=host_id)
    cleanup = ArtifactCleanup.objects.select_for_update().get(pk=data["id"], host_id=host_id)
    if str(cleanup.token) != data["token"]:
        raise PermissionError("Cleanup ownership mismatch")
    if cleanup.completed_at is None:
        cleanup.completed_at = timezone.now()
        cleanup.save(update_fields=["completed_at"])
        ExecutionAttempt.objects.filter(
            host_id=host_id,
            pk__in=cleanup.attempt_ids,
            scratch_released_at__isnull=True,
        ).update(scratch_released_at=cleanup.completed_at)
    return {"completed": True}
