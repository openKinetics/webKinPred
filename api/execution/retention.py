"""Fenced collection of managed intermediates; legacy/project files stay protected."""

import time
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
    ExecutionSpec,
    ExecutorHost,
    WorkStage,
)
from api.execution.resources import LIVE_STATES

# Most ready locations are permanent and never collectible, and a host can
# hold millions of them. One fixed 512-location page per call took hours (GPU
# host: ~43 h) to revisit residue matrices whose consumers had finished, so
# each call now scans larger pages within a small time budget.
SWEEP_PAGE = 4096
SWEEP_SECONDS = 0.5
HINTS_PER_CLAIM = 1024
PENDING_HINTS = 4096
BLOBS_PER_CLEANUP = 1024


def release_consumed_residues(stage):
    """Enqueue hints at consumer commit; ownership checks still run at deletion.

    Copies are recorded per artifact id but protected per checksum. Hint every
    transient copy of the same bytes: a copy registered under an alias whose
    own consumers finished earlier would otherwise wait for the full sweep.
    """
    checksums = Artifact.objects.filter(dependencies__stage=stage, retention="transient").values(
        "checksum"
    )
    groups = defaultdict(set)
    for ident, host in ArtifactLocation.objects.filter(
        artifact__checksum__in=checksums, artifact__retention="transient", ready=True
    ).values_list("pk", "host_id"):
        groups[host].add(ident)
    for host, ids in groups.items():
        sweep, _ = ArtifactSweep.objects.select_for_update().get_or_create(host_id=host)
        sweep.pending = list(dict.fromkeys([*sweep.pending, *sorted(ids)]))[-PENDING_HINTS:]
        sweep.save(update_fields=["pending"])


def claim_cleanup(host_id, *, version=1):
    try:
        return _claim_cleanup(host_id, version=version)
    except OperationalError as error:
        cause = error.__cause__
        state = getattr(cause, "sqlstate", None) or getattr(cause, "pgcode", None)
        if state in {"57014", "55P03"}:
            # An optional scan must not monopolize a host lock or API worker
            # (statement or lock timeout). The transaction has rolled back.
            return None
        raise


@transaction.atomic
def _claim_cleanup(host_id, *, version=1):
    # Cleanup is optional. Never occupy an HTTP worker waiting behind graph
    # preparation, or block the executor's five-second supervision deadline.
    with connection.cursor() as cursor:
        cursor.execute("SET LOCAL jit = off")
        cursor.execute("SET LOCAL max_parallel_workers_per_gather = 0")
        # Graph publication holds this lock most of the time while a large backlog
        # is prepared (82% of samples on 9 November), so a non-blocking try rarely
        # found it free and cleanup starved. Queue briefly instead: a waiter is
        # granted at the next release. Preparation chunks held it for about 20 s
        # on 10 November and a 0.9 s wait succeeded in 16 of 127 calls, so wait up
        # to 3.5 s. The statement limit counts lock waits, so it applies after the
        # lock; the remaining queries normally take under a second, ending within
        # the executor's five-second request (a lost reply resumes the cleanup).
        cursor.execute("SET LOCAL lock_timeout = 3500")
        cursor.execute("SELECT pg_advisory_xact_lock(%s)", [481726392])
        cursor.execute("SET LOCAL lock_timeout = 0")
        cursor.execute("SET LOCAL statement_timeout = 1500")
    host = (
        ExecutorHost.objects.select_for_update(skip_locked=True, no_key=True)
        .filter(pk=host_id)
        .first()
    )
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
    # Full residue matrices are work files, not a reusable long-term cache.
    # Only classified, newly generated platform artifacts in the managed blob
    # store qualify; standalone/project requests and unknown legacy files do not.
    policy = Q(artifact__retention="transient")
    if pressure:
        policy |= Q(artifact__retention="cache", verified_at__lt=cutoff)
    sweep, _ = ArtifactSweep.objects.select_for_update().get_or_create(host=host)
    base = ArtifactLocation.objects.filter(
        policy, host=host, ready=True, artifact__regenerable=True
    )
    locations = ArtifactLocation.objects.filter(host=host, ready=True)
    hinted = sweep.pending[:HINTS_PER_CLAIM]

    def collectible(ids):
        # Join each page only to its artifacts, then resolve job ownership for the
        # few transient rows. Joining every page row to its stage and execution
        # specification (detoasting large job specifications) took up to 0.7 s
        # warm and exceeded this transaction's 1.5 s limit on a cold cache, which
        # rolled back every call and stalled the sweep.
        rows = list(
            base.filter(pk__in=ids).values_list(
                "pk",
                "artifact__checksum",
                "artifact__size",
                "artifact__retention",
                "artifact__producer_id",
            )
        )
        producers = {producer for *_, retention, producer in rows if retention == "transient"}
        stages = list(
            WorkStage.objects.filter(pk__in=producers).values_list(
                "pk", "execution_id", "execution__job_id"
            )
        )
        batches = set(
            ExecutionSpec.objects.filter(
                pk__in={execution for _, execution, job in stages if job is None},
                specification__shared_protein_batch__isnull=False,
            ).values_list("pk", flat=True)
        )
        # Only prediction-owned transient intermediates qualify; cache needs none.
        owned = {pk for pk, execution, job in stages if job is not None or execution in batches}
        return {
            pk: (pk, checksum, size)
            for pk, checksum, size, retention, producer in rows
            if retention == "cache" or producer in owned
        }

    found = collectible(hinted)
    began, wrapped = time.monotonic(), False
    while True:
        page = list(
            locations.filter(pk__gt=sweep.cursor)
            .order_by("pk")
            .values_list("pk", flat=True)[:SWEEP_PAGE]
        )
        if not page:
            if wrapped or not sweep.cursor:
                sweep.cursor = 0
                break
            sweep.cursor, wrapped = 0, True
            continue
        sweep.cursor = page[-1]
        found.update(collectible(page))
        if found or time.monotonic() - began >= SWEEP_SECONDS:
            break
    candidates = [found[ident] for ident in sorted(found)]
    # Separate indexed existence checks avoid expanding every alias by all of
    # its producer/consumer attempt history inside a correlated OR subquery.
    checksums = {checksum for _, checksum, _ in candidates if checksum}
    aliases = list(
        Artifact.objects.filter(checksum__in=checksums).values_list(
            "pk", "checksum", "regenerable", "retention", "producer_id"
        )
    )
    live_stages = set(
        ExecutionAttempt.objects.filter(state__in=LIVE_STATES).values_list("stage_id", flat=True)
    )
    protected = {
        checksum
        for _, checksum, regenerable, policy, producer in aliases
        if not regenerable or policy in {"protected", "permanent"} or producer in live_stages
    }
    remaining = {ident: checksum for ident, checksum, *_ in aliases if checksum not in protected}
    protected.update(
        remaining[ident]
        for ident in ArtifactDependency.objects.filter(artifact_id__in=remaining)
        .exclude(stage__state__in=["completed", "canceled"])
        .values_list("artifact_id", flat=True)
    )
    remaining = {
        ident: checksum for ident, checksum in remaining.items() if checksum not in protected
    }
    protected.update(
        remaining[ident]
        for ident in ArtifactDependency.objects.filter(
            artifact_id__in=remaining, stage_id__in=live_stages
        ).values_list("artifact_id", flat=True)
    )
    blobs, selected, total, deferred = [], set(), 0, []
    for ident, checksum, size in candidates:
        if not checksum or checksum in selected or checksum in protected:
            continue
        if total >= 4 * 1024**3 or len(blobs) >= BLOBS_PER_CLEANUP:
            deferred.append(ident)
            continue
        selected.add(checksum)
        blobs.append({"checksum": checksum, "size": size})
        total += size
    sweep.pending = list(dict.fromkeys([*deferred, *sweep.pending[HINTS_PER_CLAIM:]]))[
        :PENDING_HINTS
    ]
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
    ExecutorHost.objects.select_for_update(no_key=True).get(pk=host_id)
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
