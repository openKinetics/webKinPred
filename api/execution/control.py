from __future__ import annotations

import hashlib
import json
import logging
import uuid
from datetime import timedelta

from django.conf import settings
from django.db import connection, transaction
from django.db.models import Exists, OuterRef, Q
from django.db.models.functions import Coalesce
from django.utils import timezone

from api.execution.models import (
    Artifact,
    ArtifactCleanup,
    ArtifactDependency,
    ArtifactLocation,
    ExecutionAttempt,
    ExecutionOutbox,
    ExecutionSpec,
    ExecutorHost,
    ResourceProfile,
    WorkStage,
)
from api.execution.resources import LIVE_STATES, admit
from api.models import Job

log = logging.getLogger(__name__)


def digest(value) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()


def require_postgres():
    if connection.vendor != "postgresql":
        raise RuntimeError("Concurrent execution requires PostgreSQL; admissions remain disabled.")


def ensure_hosts():
    """Create configured offline hosts before publishing initial input locations."""
    require_postgres()
    for name, config in settings.EXECUTION_HOSTS.items():
        ExecutorHost.objects.get_or_create(pk=name, defaults={"budgets": config["budgets"]})


@transaction.atomic
def ingest(
    public_id: str, specification: dict, input_checksum: str, *, paused=True
) -> ExecutionSpec:
    job = Job.objects.select_for_update().get(public_id=public_id)
    checksum = digest(specification)
    existing = ExecutionSpec.objects.filter(job=job).first()
    if existing:
        if existing.checksum != checksum or existing.input_checksum != input_checksum:
            raise ValueError("Conflicting immutable execution specification")
        return existing
    execution = ExecutionSpec.objects.create(
        job=job,
        specification=specification,
        checksum=checksum,
        input_checksum=input_checksum,
        paused=paused,
    )
    if "methods" in specification and job.status not in {"Completed", "Failed"}:
        from api.methods.registry import get
        from api.services.job_progress_service import initialise_job_progress_stages

        initialise_job_progress_stages(
            job,
            specification["targets"],
            {target: get(method) for target, method in specification["methods"].items()},
        )
    if job.status not in {"Completed", "Failed"}:
        WorkStage.objects.create(
            execution=execution,
            key="prepare",
            kind="prepare",
            payload={"input_checksum": input_checksum},
        )
        ExecutionOutbox.objects.get_or_create(
            key=f"prepare:{execution.pk}", defaults={"event": {"execution_id": execution.pk}}
        )
    return execution


def bind_profiles():
    """Bind only explicit matching execution profiles; missing profiles remain waiting."""
    for stage in (
        WorkStage.objects.filter(
            Q(profile__isnull=True) | Q(profile__suspended=True) | Q(profile__validated=False),
            state="ready",
        )
        .exclude(kind__in=["prepare", "barrier"])
        .select_related("execution")
        .iterator()
    ):
        profiles = ResourceProfile.objects.filter(
            stage_kind=stage.kind, method=stage.method, validated=True, suspended=False
        )
        for profile in profiles.order_by("id"):
            if stage.target not in profile.settings.get("targets", [stage.target]):
                continue
            if profile.settings.get("step") != stage.payload.get("step"):
                continue
            # An envelope measured with other weights/tokenizers is not transferable.
            identity = stage.execution.specification.get("model_versions", {})
            method = stage.execution.specification.get("methods", {}).get(
                stage.target, stage.method
            )
            expected = (
                identity.get(method)
                if stage.kind in {"embedding", "predict"}
                else stage.execution.specification.get("execution_versions", {}).get(
                    stage.kind, identity
                )
            )
            if expected is None or profile.settings.get("model_identity") != expected:
                continue
            shape = stage.payload.get("shape", {})
            if all(shape.get(k, float("inf")) <= v for k, v in profile.limits.items()):
                WorkStage.objects.filter(
                    pk=stage.pk, profile_id=stage.profile_id, state="ready"
                ).update(profile=profile, waiting_reason="")
                break
        else:
            WorkStage.objects.filter(pk=stage.pk, state="ready").update(
                waiting_reason="profiling_required"
            )


def _host_settings(host_id):
    configured = settings.EXECUTION_HOSTS.get(host_id)
    if not configured:
        raise ValueError("Unknown executor")
    return configured


@transaction.atomic
def register(host_id: str, data: dict):
    config = _host_settings(host_id)
    host, _ = ExecutorHost.objects.select_for_update().get_or_create(
        pk=host_id, defaults={"budgets": config["budgets"]}
    )
    now = timezone.now()
    recovered = host.telemetry.get("control_recovered_at")
    if not recovered or not host.heartbeat_at or now - host.heartbeat_at > timedelta(seconds=10):
        recovered = now.isoformat()
    host.capabilities = data.get("capabilities", {})
    host.telemetry = {**data.get("telemetry", {}), "control_recovered_at": recovered}
    host.heartbeat_at = now
    host.budgets = config["budgets"]
    host.save(update_fields=["capabilities", "telemetry", "heartbeat_at", "budgets"])
    log.info(
        "Executor telemetry",
        extra={
            "event": "execution.telemetry",
            "executor_host": host.pk,
            **{
                key: host.telemetry.get(key)
                for key in ["ram_available", "disk_free", "vram_free", "gpu_utilization"]
            },
        },
    )
    return host


def _gpu_profile_allowed(profile, active, host):
    if not profile.resources.get("vram", 0):
        return True
    gpu_profiles = [x["profile_fingerprint"] for x in active if x["resources"].get("vram")]
    if not gpu_profiles:
        return True
    combination = sorted(gpu_profiles + [profile.fingerprint])
    return combination in host.capabilities.get("validated_gpu_combinations", [])


@transaction.atomic
def claim(host_id: str):
    require_postgres()
    host = ExecutorHost.objects.select_for_update().get(pk=host_id)
    now = timezone.now()
    if (
        not settings.EXECUTION_V2_ENABLED
        or host.paused
        or not host.reconciled
        or not host.heartbeat_at
        or (now - host.heartbeat_at).total_seconds() > 10
        or host.telemetry.get("legacy_consumers")
        or ArtifactCleanup.objects.filter(host=host, completed_at__isnull=True).exists()
    ):
        return None
    attempts = list(host.attempts.filter(state__in=LIVE_STATES).select_related("stage"))
    observed = host.telemetry.get("attempts", {})
    active = [
        {
            "resources": a.resources,
            "observed": observed.get(str(a.pk), {}),
            "profile_fingerprint": a.profile_fingerprint,
        }
        for a in attempts
    ]
    # Apply host eligibility and completed dependencies before the bounded
    # scan. Thousands of blocked or other-host stages must not hide fitting
    # work beyond the first page, or consume its row locks.
    advertised = ResourceProfile.objects.filter(
        validated=True,
        suspended=False,
        fingerprint__in=host.capabilities.get("profiles", []),
    )
    eligible = Q(profile__fingerprint__in=host.capabilities.get("profiles", []))
    for kind, method in advertised.values_list("stage_kind", "method").distinct():
        eligible |= Q(kind=kind, method=method)
    blocked = WorkStage.dependencies.through.objects.filter(
        from_workstage_id=OuterRef("pk")
    ).exclude(to_workstage__state="completed")
    reachable_location = ArtifactLocation.objects.filter(
        artifact_id=OuterRef("artifact_id"), ready=True
    ).filter(Q(host_id=host_id) | Q(host__heartbeat_at__gte=now - timedelta(seconds=10)))
    offline_input = (
        ArtifactDependency.objects.filter(stage_id=OuterRef("pk"), artifact__locations__ready=True)
        .annotate(reachable_location=Exists(reachable_location))
        .filter(reachable_location=False)
    )
    candidates = (
        WorkStage.objects.select_for_update(skip_locked=True, of=("self",))
        .filter(
            state="ready",
            execution__paused=False,
            available_at__lte=now,
            profile__validated=True,
            profile__suspended=False,
        )
        .exclude(kind="prepare")
        .filter(eligible)
        .annotate(has_blocked_dependency=Exists(blocked))
        .filter(has_blocked_dependency=False)
        .annotate(
            has_offline_input=Exists(offline_input),
            job_has_active=Exists(
                ExecutionAttempt.objects.filter(
                    stage__execution_id=OuterRef("execution_id"), state__in=LIVE_STATES
                )
            ),
        )
        .select_related("execution__job", "profile")
        .order_by(
            "has_offline_input",
            "job_has_active",
            Coalesce("execution__job__submission_time", "execution__created_at"),
            "created_at",
        )
    )
    gpu_barrier = False
    for stage in candidates[:512].iterator(chunk_size=16):
        if stage.dependencies.exclude(state="completed").exists():
            continue
        if stage.execution.job_id and stage.execution.job.status in {"Completed", "Failed"}:
            continue
        profile = stage.profile
        if gpu_barrier and profile.resources.get("vram", 0):
            continue
        if profile.fingerprint not in host.capabilities.get("profiles", []):
            alternatives = ResourceProfile.objects.filter(
                stage_kind=stage.kind,
                method=stage.method,
                validated=True,
                suspended=False,
                fingerprint__in=host.capabilities.get("profiles", []),
            )
            profile = next(
                (
                    p
                    for p in alternatives
                    if p.settings.get("step") == stage.payload.get("step")
                    and stage.target in p.settings.get("targets", [stage.target])
                    and p.settings.get("model_identity")
                    == stage.profile.settings.get("model_identity")
                    and (
                        not stage.failures
                        or p.settings.get("batch_size", 1)
                        <= stage.profile.settings.get("batch_size", 1)
                    )
                ),
                None,
            )
            if profile is None:
                continue
        compatible = profile.evidence.get("runtime", {}).get("hosts", {})
        if stage.target not in profile.settings.get("targets", [stage.target]):
            stage.waiting_reason = "profile_target_mismatch"
            stage.save(update_fields=["waiting_reason"])
            continue
        if compatible and compatible.get(host_id) != host.capabilities.get("runtime_fingerprint"):
            continue
        if stage.kind == "assemble" and host_id != "production":
            continue
        if any(
            stage.payload.get("shape", {}).get(k, float("inf")) > v
            for k, v in profile.limits.items()
        ):
            stage.waiting_reason = "profile_shape_mismatch"
            stage.save(update_fields=["waiting_reason"])
            continue
        if (
            stage.artifacts.filter(artifact__locations__ready=True).values("pk").distinct().count()
            != stage.artifacts.count()
        ):
            stage.state = "recovering"
            stage.waiting_reason = "waiting_embeddings"
            stage.save(update_fields=["state", "waiting_reason"])
            ExecutionOutbox.objects.create(
                key=f"recover-unavailable:{uuid.uuid4()}",
                event={"recover_stage": str(stage.pk), "host_id": None},
            )
            continue
        # A verified remote blob survives a host outage, but cannot be fetched
        # until that host returns. Keep ownership and artifacts intact rather
        # than reserving a container repeatedly for a known unavailable input.
        reachable = Q(artifact__locations__host_id=host_id) | Q(
            artifact__locations__host__heartbeat_at__gte=now - timedelta(seconds=10)
        )
        if (
            stage.artifacts.filter(reachable, artifact__locations__ready=True)
            .values("pk")
            .distinct()
            .count()
            != stage.artifacts.count()
        ):
            stage.waiting_reason = "network"
            stage.save(update_fields=["waiting_reason"])
            continue
        resources = {
            key: max(value, profile.runtime_floor.get(key, value))
            for key, value in profile.resources.items()
        }
        # Each input needs a private working copy and may also need a new local
        # immutable blob. Reserve both even when an eligible peer owns the data.
        input_bytes = sum(stage.artifacts.values_list("artifact__size", flat=True))
        resources["disk"] = resources.get("disk", 0) + 2 * input_bytes
        from api.execution.locality import prefer_peer

        if prefer_peer(stage, host, profile, input_bytes, now):
            stage.waiting_reason = "waiting_cpu"
            stage.save(update_fields=["waiting_reason"])
            continue
        result = admit(
            resources,
            active,
            host.budgets,
            host.telemetry,
            gpu=bool(profile.resources.get("vram")),
        )
        if not result.fits or not _gpu_profile_allowed(profile, active, host):
            stage.waiting_reason = result.reason or "unvalidated_gpu_combination"
            if profile.resources.get("vram", 0):
                # Planning/dependency time is not time spent eligible for a
                # GPU. Otherwise an old dependency graph drains the GPU after
                # each allocation and prevents validated pairs from running.
                stage.payload.setdefault("admission_wait_started_at", now.isoformat())
            stage.save(update_fields=["waiting_reason", "payload"])
            if (
                profile.resources.get("vram", 0)
                and (
                    now
                    - timezone.datetime.fromisoformat(stage.payload["admission_wait_started_at"])
                ).total_seconds()
                >= 600
                and resources.get("vram", 0)
                + max(2 * 1024**3, host.telemetry.get("vram_total", 0) // 10)
                <= host.telemetry.get("vram_total", 0)
            ):
                gpu_barrier = True
            continue
        attempt = ExecutionAttempt.objects.create(
            stage=stage,
            host=host,
            resources=resources,
            profile_fingerprint=profile.fingerprint,
            lease_until=now + timedelta(seconds=30),
        )
        stage.state = "running"
        stage.profile = profile
        stage.waiting_reason = ""
        stage.save(update_fields=["state", "waiting_reason", "profile"])
        Job.objects.filter(pk=stage.execution.job_id, start_time__isnull=True).update(
            start_time=now, status="Processing"
        )
        log.info(
            "Execution admitted",
            extra={
                "event": "execution.admitted",
                "attempt_id": str(attempt.pk),
                "executor_host": host.pk,
                "method_key": stage.method,
                **resources,
            },
        )
        return serialize_attempt(attempt)
    return None


def serialize_attempt(attempt):
    stage = attempt.stage
    profile = stage.profile
    artifacts = []
    for dependency in stage.artifacts.select_related("artifact"):
        artifacts.append(
            {
                "id": dependency.artifact_id,
                "checksum": dependency.artifact.checksum,
                "size": dependency.artifact.size,
                "path": dependency.path,
                "hosts": list(
                    dependency.artifact.locations.filter(ready=True).values_list(
                        "host_id", flat=True
                    )
                ),
            }
        )
    for output in stage.payload.get("outputs", []):
        artifact = (
            Artifact.objects.filter(pk=output["id"], locations__ready=True).distinct().first()
        )
        if artifact:
            artifacts.append(
                {
                    "id": artifact.pk,
                    "checksum": artifact.checksum,
                    "size": artifact.size,
                    "path": output["path"],
                    "hosts": list(
                        artifact.locations.filter(ready=True).values_list("host_id", flat=True)
                    ),
                }
            )
    return {
        "id": str(attempt.pk),
        "token": str(attempt.token),
        "stage_id": str(stage.pk),
        "kind": stage.kind,
        "method": stage.method,
        "target": stage.target,
        "public_id": stage.execution.job.public_id
        if stage.execution.job_id
        else f"embed-{stage.execution_id}",
        "specification": stage.execution.specification,
        "payload": stage.payload,
        "resources": attempt.resources,
        "profile_fingerprint": profile.fingerprint,
        "image": profile.image,
        "settings": profile.settings,
        "artifacts": artifacts,
        "lease_seconds": 30,
    }


def _owned(host_id, attempt_id, token):
    attempt = (
        ExecutionAttempt.objects.select_for_update(of=("self",))
        .select_related("stage__execution__job")
        .get(pk=attempt_id, host_id=host_id)
    )
    if str(attempt.token) != str(token):
        raise PermissionError("Attempt ownership mismatch")
    return attempt


def _stopped_at(attempt, data):
    from django.utils.dateparse import parse_datetime

    now = timezone.now()
    supplied = data.get("container_finished_at")
    ended = parse_datetime(supplied) if isinstance(supplied, str) else None
    if ended is None or timezone.is_naive(ended):
        ended = now
    return min(now, max(attempt.started_at, ended))


@transaction.atomic
def heartbeat(host_id, attempt_id, data):
    attempt = _owned(host_id, attempt_id, data["token"])
    now = timezone.now()
    if attempt.state not in {"leased", "running"} or attempt.lease_until <= now:
        raise ValueError("Attempt lease expired; executor must stop")
    attempt.heartbeat_at = now
    attempt.lease_until = now + timedelta(seconds=30)
    attempt.state = "running"
    attempt.container_id = data.get("container_id", attempt.container_id)
    attempt.save(update_fields=["heartbeat_at", "lease_until", "state", "container_id"])
    stage = attempt.stage
    stage.progress = max(stage.progress, int(data.get("progress", 0)))
    stage.save(update_fields=["progress"])
    from api.execution.progress import refresh

    refresh(stage.execution)
    return {"lease_seconds": 30}


@transaction.atomic
def checkpoint(host_id, attempt_id, data):
    """Publish a verified subset without completing the stage or releasing reservations."""
    attempt = _owned(host_id, attempt_id, data["token"])
    if attempt.state not in {"leased", "running"} or attempt.lease_until <= timezone.now():
        raise ValueError("Stale attempt cannot checkpoint")
    expected = {output["id"]: output for output in attempt.stage.payload.get("outputs", [])}
    for output in data.get("artifacts", []):
        declaration = expected.get(output["id"])
        if not declaration or declaration["identity"] != output["identity"]:
            raise ValueError("Undeclared checkpoint artifact")
        artifact = Artifact.objects.select_for_update().get(pk=output["id"], producer=attempt.stage)
        if (
            digest(output["identity"]) != artifact.pk
            or len(output["checksum"]) != 64
            or output["size"] < 0
        ):
            raise ValueError("Invalid checkpoint manifest")
        if artifact.checksum and (
            artifact.checksum != output["checksum"] or artifact.size != output["size"]
        ):
            raise ValueError("Checkpoint identity conflict")
        artifact.checksum, artifact.size = output["checksum"], output["size"]
        artifact.save(update_fields=["checksum", "size"])
        ArtifactLocation.objects.update_or_create(
            artifact=artifact,
            host_id=host_id,
            defaults={"ready": True, "verified_at": timezone.now()},
        )
    return {"checkpointed": len(data.get("artifacts", []))}


@transaction.atomic
def artifact_location(host_id, attempt_id, data):
    """Record verified transfers and confirmed 404/corruption for an owned dependency."""
    attempt = _owned(host_id, attempt_id, data["token"])
    if attempt.state not in {"leased", "running"} or attempt.lease_until <= timezone.now():
        raise ValueError("Stale attempt cannot change artifact availability")
    artifact_id = data["artifact_id"]
    if not attempt.stage.artifacts.filter(
        artifact_id=artifact_id
    ).exists() and artifact_id not in attempt.stage.payload.get("output_artifacts", []):
        raise PermissionError("Artifact is outside this attempt")
    artifact = Artifact.objects.select_for_update().get(pk=artifact_id)
    if artifact.checksum != data["checksum"]:
        raise ValueError("Artifact checksum changed")
    source_host = data.get("source_host", host_id)
    if data.get("ready") is True:
        if source_host != host_id or data.get("size") != artifact.size:
            raise PermissionError("Only this executor can verify its own copy")
        ArtifactLocation.objects.update_or_create(
            artifact=artifact,
            host_id=host_id,
            defaults={"ready": True, "verified_at": timezone.now()},
        )
    elif data.get("ready") is False:
        artifact.locations.filter(host_id=source_host).update(ready=False)
    else:
        raise ValueError("Artifact readiness must be boolean")
    return {"recorded": True}


@transaction.atomic
def finish(host_id, attempt_id, data):
    attempt = _owned(host_id, attempt_id, data["token"])
    if attempt.state == "completed":
        if digest(attempt.outcome) != digest(data):
            raise ValueError("Conflicting duplicate completion")
        return {"state": "completed"}
    if attempt.state not in {"leased", "running"} or attempt.lease_until <= timezone.now():
        raise ValueError("Stale attempt cannot commit")
    if not data.get("stopped", False):
        raise ValueError("Executor must confirm container termination")
    stage = attempt.stage
    outputs = data.get("artifacts", [])
    expected = set(stage.payload.get("output_artifacts", []))
    if expected != {output["id"] for output in outputs} or len(outputs) != len(expected):
        raise ValueError("Incomplete or undeclared output manifest")
    for output in outputs:
        identity = output["identity"]
        if digest(identity) != output["id"] or len(output["checksum"]) != 64 or output["size"] < 0:
            raise ValueError("Invalid artifact manifest")
        artifact, created = Artifact.objects.get_or_create(
            pk=output["id"],
            defaults={
                "identity": identity,
                "checksum": output["checksum"],
                "size": output["size"],
                "metadata": output.get("metadata", {}),
            },
        )
        if not created and artifact.checksum:
            if artifact.checksum != output["checksum"] or artifact.size != output["size"]:
                raise ValueError("Artifact identity conflict")
        elif not created:
            if artifact.producer_id != stage.pk:
                raise ValueError("Artifact producer ownership mismatch")
            artifact.checksum = output["checksum"]
            artifact.size = output["size"]
            artifact.metadata = {**artifact.metadata, **output.get("metadata", {})}
            artifact.save(update_fields=["checksum", "size", "metadata"])
        ArtifactLocation.objects.update_or_create(
            artifact=artifact,
            host_id=host_id,
            defaults={"ready": True, "verified_at": timezone.now()},
        )
    # A runner cannot declare an embedding complete without producing every required output.
    attempt.state = "completed"
    attempt.outcome = data
    attempt.finished_at = _stopped_at(attempt, data)
    attempt.save(update_fields=["state", "outcome", "finished_at"])
    stage.state = "completed"
    stage.checkpoint = data.get("checkpoint", {})
    stage.completed_at = timezone.now()
    stage.save(update_fields=["state", "checkpoint", "completed_at"])
    if stage.kind == "assemble" and stage.payload.get("phase") == "preliminary":
        for final in stage.execution.stages.filter(key="publish"):
            final.payload["assembly_checkpoint"] = stage.checkpoint
            final.save(update_fields=["payload"])
    if stage.kind == "assemble" and stage.payload.get("phase") in {"final", "publish"}:
        import os
        import shutil
        import tempfile
        from pathlib import Path

        from api.execution.artifacts import ArtifactStore, sha256_file
        from api.execution.quota import refund

        if host_id != "production" or len(outputs) != 1:
            raise ValueError("Final publication must occur on production")
        job = stage.execution.job
        result = outputs[0]
        source = ArtifactStore(Path(settings.EXECUTION_BLOB_ROOT)).path(result["checksum"])
        if not source.is_file() or sha256_file(source) != result["checksum"]:
            raise ValueError("Final result blob has not reached authoritative storage")
        directory = Path(settings.MEDIA_ROOT) / "jobs" / job.public_id
        directory.mkdir(parents=True, exist_ok=True)
        fd, temp = tempfile.mkstemp(prefix=".completed-", dir=directory)
        try:
            with os.fdopen(fd, "wb") as handle, source.open("rb") as incoming:
                shutil.copyfileobj(incoming, handle)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temp, directory / "output.csv")
        finally:
            if os.path.exists(temp):
                os.unlink(temp)
        completed = int(stage.checkpoint.get("completed_reactions", -1))
        if not 0 <= completed <= job.requested_rows:
            raise ValueError("Final reaction count is invalid")
        refund(job, job.requested_rows - completed)
        Job.objects.filter(pk=job.pk).update(
            status="Completed",
            completion_time=timezone.now(),
            output_file=f"jobs/{job.public_id}/output.csv",
        )
    ExecutionOutbox.objects.get_or_create(
        key=f"complete:{stage.pk}",
        defaults={
            "event": {
                "stage_id": str(stage.pk),
                "cache_writes": stage.checkpoint.get("cache_writes", []),
                "similarity_writes": stage.checkpoint.get("similarity_writes", []),
            }
        },
    )
    from api.execution.progress import refresh

    refresh(stage.execution)
    return {"state": "completed"}


@transaction.atomic
def fail(host_id, attempt_id, data):
    attempt = _owned(host_id, attempt_id, data["token"])
    if attempt.state == "completed":
        raise ValueError("Completed attempt cannot fail")
    if attempt.state == "stopped":
        return {"state": attempt.stage.state}
    if not data.get("stopped", False):
        attempt.state = "uncertain"
        attempt.save(update_fields=["state"])
        return {"state": "uncertain"}
    stage = WorkStage.objects.select_for_update().get(pk=attempt.stage_id)
    attempt.state = "stopped"
    attempt.finished_at = _stopped_at(attempt, data)
    attempt.outcome = data
    attempt.save(update_fields=["state", "finished_at", "outcome"])
    stage.failures += 1
    stage.waiting_reason = str(data.get("category", "unknown"))[:128]
    stage.available_at = timezone.now() + timedelta(
        seconds=min(300, 5 * 2 ** min(stage.failures, 6))
    )
    temporary = stage.waiting_reason in {
        "network",
        "disk",
        "gpu_unavailable",
        "waiting_ram",
        "waiting_vram",
        "waiting_disk",
        "waiting_cpu",
        "executor_reconciled",
        "interrupted",
        "canceled",
        "storage",
        "cpu",
        "host_pressure",
        "waiting_gpu_slot",
        "telemetry_unavailable",
    }
    adjusted = False
    if stage.waiting_reason == "oom" and stage.profile_id:
        profile = ResourceProfile.objects.select_for_update().get(pk=stage.profile_id)
        alternatives = list(
            ResourceProfile.objects.filter(
                stage_kind=profile.stage_kind,
                method=profile.method,
                validated=True,
                suspended=False,
            ).exclude(pk=profile.pk)
        )
        smaller = [
            p
            for p in alternatives
            if p.settings.get("step") == profile.settings.get("step")
            and stage.target in p.settings.get("targets", [stage.target])
            and p.settings.get("model_identity") == profile.settings.get("model_identity")
            and all(
                stage.payload.get("shape", {}).get(key, float("inf")) <= value
                for key, value in p.limits.items()
            )
            and p.settings.get("batch_size", 1) < profile.settings.get("batch_size", 1)
        ]
        if smaller:
            stage.profile = max(smaller, key=lambda p: p.settings.get("batch_size", 1))
            adjusted = True
        elif not profile.resources.get("vram"):
            measured_hosts = profile.evidence.get("runtime", {}).get("hosts", {})
            ceiling = max(
                (
                    host.get("budgets", {}).get("ram", 0)
                    for name, host in settings.EXECUTION_HOSTS.items()
                    if not measured_hosts or name in measured_hosts
                ),
                default=0,
            )
            previous = max(profile.resources["ram"], profile.runtime_floor.get("ram", 0))
            if previous < ceiling:
                profile.runtime_floor["ram"] = min(previous * 2, ceiling)
                profile.save(update_fields=["runtime_floor"])
                adjusted = True
    recent = list(stage.attempts.filter(state="stopped").order_by("-finished_at")[:3])
    repeated = len(recent) == 3 and all(
        previous.profile_fingerprint == attempt.profile_fingerprint
        and previous.resources == attempt.resources
        and previous.outcome.get("category") == stage.waiting_reason
        for previous in recent
    )
    stage.state = "held" if repeated and not temporary and not adjusted else "ready"
    if stage.waiting_reason == "missing_embedding" and stage.state != "held":
        stage.state = "recovering"
        ExecutionOutbox.objects.get_or_create(
            key=f"recover:{attempt.pk}",
            defaults={"event": {"recover_stage": str(stage.pk), "host_id": host_id}},
        )
    stage.save(update_fields=["failures", "waiting_reason", "available_at", "state", "profile"])
    if stage.state == "held" and stage.profile_id:
        ResourceProfile.objects.filter(pk=stage.profile_id).update(suspended=True)
    from api.execution.progress import refresh

    refresh(stage.execution)
    log.warning(
        "Execution deferred",
        extra={
            "event": "execution.deferred",
            "executor_host": host_id,
            "attempt_id": str(attempt.pk),
            "method_key": stage.method,
            "reason": stage.waiting_reason,
            "stage_state": stage.state,
        },
    )
    return {"state": stage.state}


@transaction.atomic
def reconcile(host_id, data):
    """Agent supplies a full running inventory; expiry alone never releases reservations."""
    host = ExecutorHost.objects.select_for_update().get(pk=host_id)
    running = set(data.get("running", []))
    for attempt in host.attempts.filter(state__in=LIVE_STATES):
        if str(attempt.pk) not in running:
            fail(
                host_id,
                attempt.pk,
                {
                    "token": str(attempt.token),
                    "stopped": True,
                    "category": "executor_reconciled",
                    "container_finished_at": data.get("stopped", {}).get(str(attempt.pk)),
                },
            )
        elif attempt.lease_until <= timezone.now():
            attempt.state = "uncertain"
            attempt.save(update_fields=["state"])
    host.reconciled = not host.attempts.filter(state="uncertain").exists()
    host.save(update_fields=["reconciled"])
    return {"reconciled": host.reconciled}
