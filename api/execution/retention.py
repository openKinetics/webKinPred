"""Two-phase, host-idle garbage collection of regenerable immutable blobs."""

from datetime import timedelta

from django.db import connection, transaction
from django.db.models import Exists, OuterRef, Q
from django.utils import timezone

from api.execution.models import (
    Artifact,
    ArtifactCleanup,
    ArtifactDependency,
    ArtifactLocation,
    ExecutorHost,
)
from api.execution.resources import LIVE_STATES


@transaction.atomic
def claim_cleanup(host_id):
    # Cleanup is optional. Never occupy an HTTP worker waiting behind graph
    # preparation, or block the executor's five-second supervision deadline.
    with connection.cursor() as cursor:
        cursor.execute("SELECT pg_try_advisory_xact_lock(%s)", [481726392])
        if not cursor.fetchone()[0]:
            return None
    host = ExecutorHost.objects.select_for_update(skip_locked=True).filter(pk=host_id).first()
    if host is None:
        return None
    if not host.reconciled or host.attempts.filter(state__in=LIVE_STATES).exists():
        return None
    existing = ArtifactCleanup.objects.filter(host=host, completed_at__isnull=True).first()
    if existing:
        return {"id": str(existing.pk), "token": str(existing.token), "blobs": existing.blobs}
    target = host.budgets.get(
        "cache_free_target", host.budgets.get("disk_headroom", 0) + 10 * 1024**3
    )
    free = host.telemetry.get("disk_free")
    if free is None or free >= target:
        return None
    cutoff = timezone.now() - timedelta(seconds=host.budgets.get("cache_min_age_seconds", 3600))
    protected_aliases = (
        Artifact.objects.filter(checksum=OuterRef("artifact__checksum"))
        .annotate(
            unfinished=Exists(
                ArtifactDependency.objects.filter(artifact_id=OuterRef("pk")).exclude(
                    stage__state="completed"
                )
            )
        )
        .filter(
            Q(regenerable=False) | Q(unfinished=True) | Q(producer__attempts__state__in=LIVE_STATES)
        )
    )
    candidates = (
        ArtifactLocation.objects.filter(
            host=host, ready=True, verified_at__lt=cutoff, artifact__regenerable=True
        )
        .annotate(protected=Exists(protected_aliases))
        .filter(protected=False)
        .select_related("artifact")
        .order_by("verified_at")[:512]
    )
    blobs, selected, total = [], set(), 0
    for location in candidates:
        artifact = location.artifact
        checksum = artifact.checksum
        if not checksum or checksum in selected:
            continue
        selected.add(checksum)
        blobs.append({"checksum": checksum, "size": artifact.size})
        total += artifact.size
        if total >= target - free or len(blobs) >= 128:
            break
    if not blobs:
        return None
    cleanup = ArtifactCleanup.objects.create(host=host, blobs=blobs)
    # Claims on this host remain blocked until deletion is acknowledged. New
    # consumers see missing locations and schedule verified copies/regeneration.
    ArtifactLocation.objects.filter(host=host, artifact__checksum__in=selected).update(ready=False)
    return {"id": str(cleanup.pk), "token": str(cleanup.token), "blobs": blobs}


@transaction.atomic
def finish_cleanup(host_id, data):
    ExecutorHost.objects.select_for_update().get(pk=host_id)
    cleanup = ArtifactCleanup.objects.select_for_update().get(pk=data["id"], host_id=host_id)
    if str(cleanup.token) != data["token"]:
        raise PermissionError("Cleanup ownership mismatch")
    if cleanup.completed_at is None:
        cleanup.completed_at = timezone.now()
        cleanup.save(update_fields=["completed_at"])
    return {"completed": True}
