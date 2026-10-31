from __future__ import annotations

import hashlib
import json
import logging
import uuid
from datetime import timedelta
from itertools import combinations

from django.conf import settings
from django.db import connection, transaction
from django.db.models import BooleanField, Case, Exists, OuterRef, Q, Value, When
from django.db.models.functions import Coalesce, Greatest
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
    from api.execution.background import background_for_job

    background = background_for_job(job)
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
        background=background,
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


def _after_stopping(active, removed, telemetry):
    removed = set(removed)
    retained = [row for row in active if row["id"] not in removed]
    projected = dict(telemetry)
    for resource, available, total in (
        ("ram", "ram_available", "ram_total"),
        ("vram", "vram_free", "vram_total"),
    ):
        if available in projected:
            projected[available] += sum(
                row["observed"].get(resource, 0) for row in active if row["id"] in removed
            )
            if total in projected:
                projected[available] = min(projected[available], projected[total])
    return retained, projected


def _preempt_background(stage, profile, resources, active, host):
    """Request the smallest useful background stop set; retain all reservations."""
    lower = sorted(
        [row for row in active if row["background"]],
        key=lambda row: (
            bool(row["resources"].get("vram")) == bool(resources.get("vram")),
            row["started_at"],
        ),
        reverse=True,
    )
    if not lower or stage.has_offline_input:
        return False
    if (
        stage.artifacts.filter(artifact__locations__ready=True).values("pk").distinct().count()
        != stage.artifacts.count()
    ):
        return False
    for size in range(1, len(lower) + 1):
        for subset in combinations(lower, size):
            ids = [row["id"] for row in subset]
            retained, projected = _after_stopping(active, ids, host.telemetry)
            if admit(
                resources, retained, host.budgets, projected, gpu=bool(resources.get("vram"))
            ).fits and _gpu_profile_allowed(profile, retained, host):
                ExecutionAttempt.objects.filter(pk__in=ids, state__in=LIVE_STATES).update(
                    stop_requested=True
                )
                log.info(
                    "Background work yielding resources",
                    extra={
                        "event": "execution.background_preemption",
                        "executor_host": host.pk,
                        "foreground_stage": str(stage.pk),
                        "attempts": len(ids),
                    },
                )
                return True
    return False


def _admission_wait(stage, profile, resources, host, reason, now):
    fields = []
    if stage.waiting_reason != reason:
        stage.waiting_reason = reason
        fields.append("waiting_reason")
    if profile.resources.get("vram", 0) and "admission_wait_started_at" not in stage.payload:
        stage.payload["admission_wait_started_at"] = now.isoformat()
        fields.append("payload")
    if fields:
        stage.save(update_fields=fields)
    return (
        bool(profile.resources.get("vram", 0))
        and not stage.has_offline_input
        and all(resources.get(key, 0) <= host.budgets[key] for key in ("ram", "cpu", "disk"))
        and (
            max(stage.priority, stage.execution.priority) > 0
            or (
                now - timezone.datetime.fromisoformat(stage.payload["admission_wait_started_at"])
            ).total_seconds()
            >= 600
        )
        and resources.get("vram", 0) + max(2 * 1024**3, host.telemetry.get("vram_total", 0) // 10)
        <= host.telemetry.get("vram_total", 0)
    )


@transaction.atomic
def claim(host_id: str):
    require_postgres()
    # This is a short control request, not an analytical query. JIT compilation
    # of the large compatibility predicate can itself exceed the HTTP deadline.
    with connection.cursor() as cursor:
        cursor.execute(
            "SET LOCAL jit = off; SET LOCAL max_parallel_workers_per_gather = 0; "
            "SET LOCAL statement_timeout = '3000ms'"
        )
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
    attempts = list(
        host.attempts.filter(state__in=LIVE_STATES)
        .select_related("stage__execution")
        .defer("stage__execution__specification", "stage__payload", "stage__checkpoint")
    )
    if any(a.stop_requested for a in attempts):
        return None  # Confirm termination before reallocating any reservation.
    from api.execution.background import foreground_dependencies

    needed = foreground_dependencies()
    observed = host.telemetry.get("attempts", {})
    active = [
        {
            "id": a.pk,
            "started_at": a.started_at,
            "background": a.stage.execution.background and a.stage_id not in needed,
            "resources": a.resources,
            "observed": observed.get(str(a.pk), {}),
            "profile_fingerprint": a.profile_fingerprint,
        }
        for a in attempts
    ]
    # Apply host eligibility and completed dependencies before the bounded
    # scan. Thousands of blocked or other-host stages must not hide fitting
    # work beyond the first page, or consume its row locks.
    advertised = list(
        ResourceProfile.objects.filter(
            validated=True,
            suspended=False,
            fingerprint__in=host.capabilities.get("profiles", []),
        )
    )
    eligible = Q(profile__fingerprint__in=host.capabilities.get("profiles", []))
    for kind, method in {(p.stage_kind, p.method) for p in advertised}:
        eligible |= Q(kind=kind, method=method)
    blocked = WorkStage.dependencies.through.objects.filter(
        from_workstage_id=OuterRef("pk")
    ).exclude(to_workstage__state="completed")
    reachable_hosts = [
        host_id,
        *ExecutorHost.objects.filter(heartbeat_at__gte=now - timedelta(seconds=10)).values_list(
            "pk", flat=True
        ),
    ]
    offline_locations = ArtifactLocation.objects.filter(ready=True).exclude(
        host_id__in=reachable_hosts
    )
    # Availability is a property of locations, not of every candidate/artifact
    # pair. The old nested correlated EXISTS ran millions of checks just to
    # order a backlog, including when every host was online.
    offline = Value(False, output_field=BooleanField())
    if offline_locations.exists():
        reachable_artifacts = ArtifactLocation.objects.filter(
            ready=True, host_id__in=reachable_hosts
        ).values("artifact_id")
        offline_artifacts = offline_locations.exclude(artifact_id__in=reachable_artifacts).values(
            "artifact_id"
        )
        offline = Exists(
            ArtifactDependency.objects.filter(
                stage_id=OuterRef("pk"), artifact_id__in=offline_artifacts
            )
        )
    active_executions = ExecutionAttempt.objects.filter(state__in=LIVE_STATES).values(
        "stage__execution_id"
    )
    candidates = (
        WorkStage.objects.filter(
            state="ready",
            execution__paused=False,
            available_at__lte=now,
            profile__validated=True,
            profile__suspended=False,
        )
        .exclude(execution__job__status__in=["Completed", "Failed"])
        .exclude(kind="prepare")
        .filter(eligible)
        .annotate(has_blocked_dependency=Exists(blocked))
        .filter(has_blocked_dependency=False)
        .annotate(
            effective_background=Case(
                When(Q(execution__background=True) & ~Q(pk__in=needed), then=Value(True)),
                default=Value(False),
            ),
            effective_priority=Greatest("priority", "execution__priority"),
            has_offline_input=offline,
            job_has_active=Case(
                When(execution_id__in=active_executions, then=Value(True)),
                default=Value(False),
            ),
        )
        .select_related("execution__job", "profile")
        .defer("execution__specification", "profile__evidence")
        .order_by(
            "effective_background",
            "-effective_priority",
            "has_offline_input",
            "job_has_active",
            Coalesce("execution__job__submission_time", "execution__created_at"),
            "created_at",
        )
    )
    # A page full of resource-blocked GPU work must not hide CPU work or a
    # compatible GPU pair beyond that page. Keep the oldest candidate of each
    # class for its waiting reason/fairness barrier, and preselect fitting
    # profiles before applying the bounded scan.
    cpu_stage = Q(profile__resources__vram__isnull=True) | Q(profile__resources__vram__lte=0)
    oldest = [
        candidates.filter(rule)
        .order_by(
            "effective_background",
            "-effective_priority",
            Coalesce("execution__job__submission_time", "execution__created_at"),
            "created_at",
        )
        .values_list("pk", flat=True)
        .first()
        for rule in (cpu_stage, Q(profile__resources__vram__gt=0))
    ]
    fitting = []
    foreground_fitting = []
    retained, projected = _after_stopping(
        active, [row["id"] for row in active if row["background"]], host.telemetry
    )
    for p in advertised:
        resources = {k: max(v, p.runtime_floor.get(k, v)) for k, v in p.resources.items()}
        if admit(
            resources, active, host.budgets, host.telemetry, gpu=bool(resources.get("vram"))
        ).fits and _gpu_profile_allowed(p, active, host):
            fitting.append(p)
        if admit(
            resources, retained, host.budgets, projected, gpu=bool(resources.get("vram"))
        ).fits and _gpu_profile_allowed(p, retained, host):
            foreground_fitting.append(p)

    def profile_capacity(profiles):
        capacity = Q(profile_id__in=[p.pk for p in profiles])
        for p in profiles:
            if not p.resources.get("vram"):
                identity = p.settings.get("model_identity")
                same_identity = Q(profile__settings__model_identity=identity)
                if identity is None:
                    same_identity |= Q(profile__settings__model_identity__isnull=True)
                capacity |= (
                    cpu_stage
                    & same_identity
                    & Q(kind=p.stage_kind, method=p.method)
                    & ~Q(profile__fingerprint__in=host.capabilities.get("profiles", []))
                )
        return capacity

    foreground_capacity = profile_capacity(foreground_fitting)
    normal_executions = list(
        ExecutionSpec.objects.filter(background=False)
        .exclude(job__status__in=["Completed", "Failed"])
        .values_list("pk", flat=True)
    )
    foreground = Q(execution_id__in=normal_executions) | Q(pk__in=needed)
    # Nonlocking demand check also sees normal work another host is claiming.
    # It must not let a background request take that capacity in the meantime.
    # Use indexed stage ownership/IDs here: filtering the ordering CASE forces
    # PostgreSQL to inspect the whole background backlog to prove no demand.
    foreground_demand = (
        candidates.filter(foreground, has_offline_input=False).filter(foreground_capacity).exists()
    )
    capacity = (
        Q(pk__in=[pk for pk in oldest if pk is not None])
        | profile_capacity(fitting)
        | (foreground & foreground_capacity)
    )
    candidates = candidates.filter(capacity).select_for_update(skip_locked=True, of=("self",))
    gpu_barrier = False
    # Rank/lock IDs only. Sorting joined payloads, checkpoint JSON and profile
    # manifests for the whole backlog spills large rows to disk. Fetch the full
    # context only for candidates we actually inspect under those row locks.
    candidate_rows = list(
        candidates.values_list("pk", "has_offline_input", "effective_background")[:512]
    )
    for ident, has_offline_input, background in candidate_rows:
        if background and foreground_demand:
            continue
        stage = (
            WorkStage.objects.select_related("execution__job", "profile")
            .defer("execution__specification", "profile__evidence")
            .get(pk=ident)
        )
        stage.has_offline_input = has_offline_input
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
        resources = {
            key: max(value, profile.runtime_floor.get(key, value))
            for key, value in profile.resources.items()
        }
        # Reject unavailable capacity and unvalidated combinations before
        # per-artifact queries. A saturated 5,000-job queue must not spend
        # thousands of queries holding the host lock and delay heartbeats.
        base = admit(
            resources, active, host.budgets, host.telemetry, gpu=bool(profile.resources.get("vram"))
        )
        if not base.fits or not _gpu_profile_allowed(profile, active, host):
            if not background and _preempt_background(stage, profile, resources, active, host):
                return None
            gpu_barrier |= _admission_wait(
                stage,
                profile,
                resources,
                host,
                base.reason or "unvalidated_gpu_combination",
                now,
            )
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
            if not background and _preempt_background(stage, profile, resources, active, host):
                return None
            gpu_barrier |= _admission_wait(
                stage,
                profile,
                resources,
                host,
                result.reason or "unvalidated_gpu_combination",
                now,
            )
            continue
        admitted_at = timezone.now()
        attempt = ExecutionAttempt.objects.create(
            stage=stage,
            host=host,
            resources=resources,
            profile_fingerprint=profile.fingerprint,
            lease_until=admitted_at + timedelta(seconds=30),
        )
        stage.state = "running"
        stage.profile = profile
        stage.waiting_reason = ""
        stage.save(update_fields=["state", "waiting_reason", "profile"])
        Job.objects.filter(pk=stage.execution.job_id, start_time__isnull=True).update(
            start_time=admitted_at, status="Processing"
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
    from django.db.models import Prefetch

    for dependency in stage.artifacts.select_related("artifact").prefetch_related(
        Prefetch(
            "artifact__locations",
            queryset=ArtifactLocation.objects.filter(ready=True),
            to_attr="verified_locations",
        )
    ):
        artifacts.append(
            {
                "id": dependency.artifact_id,
                "checksum": dependency.artifact.checksum,
                "size": dependency.artifact.size,
                "path": dependency.path,
                "hosts": [location.host_id for location in dependency.artifact.verified_locations],
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
        from api.execution.artifact_policy import retention_for

        artifact.retention = retention_for(artifact.identity)
        if artifact.retention == "transient" and attempt.stage.execution.job_id is None:
            artifact.retention = "protected"
        artifact.save(update_fields=["checksum", "size", "retention"])
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
        from api.execution.artifact_policy import retention_for

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
                "retention": retention_for(identity),
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
        retention = retention_for(identity) if artifact.regenerable else "permanent"
        if retention == "transient" and stage.execution.job_id is None:
            retention = "protected"
        if artifact.retention != retention:
            artifact.retention = retention
            artifact.save(update_fields=["retention"])
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
