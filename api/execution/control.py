from __future__ import annotations

import hashlib
import json
import logging
import time
import uuid
from datetime import timedelta
from itertools import combinations

from django.conf import settings
from django.db import connection, transaction
from django.db.models import BooleanField, Case, Exists, OuterRef, Q, Sum, Value, When
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
from api.execution.resources import LIVE_STATES, admit, cpu_capacity
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
    """Bind only explicit matching execution profiles; missing profiles remain waiting.

    Reads only the compared fields. Loading every unbound stage with its full payload
    and execution specification exceeded the controller's memory once one large
    submission published ~1,900 stages, so none of them could ever be claimed.
    """
    stages = list(
        WorkStage.objects.filter(
            Q(profile__isnull=True) | Q(profile__suspended=True) | Q(profile__validated=False),
            state="ready",
        )
        .exclude(kind__in=["prepare", "barrier"])
        .values_list(
            "pk",
            "profile_id",
            "kind",
            "method",
            "target",
            "execution_id",
            "payload__step",
            "payload__shape",
        )
    )
    if not stages:
        return
    specifications = {
        pk: (versions or {}, methods or {}, execution_versions or {})
        for pk, versions, methods, execution_versions in ExecutionSpec.objects.filter(
            pk__in={row[5] for row in stages}
        ).values_list(
            "pk",
            "specification__model_versions",
            "specification__methods",
            "specification__execution_versions",
        )
    }
    candidates = {}
    for profile in (
        ResourceProfile.objects.filter(validated=True, suspended=False)
        .defer("evidence")
        .order_by("id")
    ):
        candidates.setdefault((profile.stage_kind, profile.method), []).append(profile)
    for pk, profile_id, kind, method_key, target, execution_id, step, shape in stages:
        identity, methods, execution_versions = specifications[execution_id]
        shape = shape or {}
        for profile in candidates.get((kind, method_key), []):
            if target not in profile.settings.get("targets", [target]):
                continue
            if profile.settings.get("step") != step:
                continue
            # An envelope measured with other weights/tokenizers is not transferable.
            method = methods.get(target, method_key)
            expected = (
                identity.get(method)
                if kind in {"embedding", "predict"}
                else execution_versions.get(kind, identity)
            )
            if expected is None or profile.settings.get("model_identity") != expected:
                continue
            if all(shape.get(k, float("inf")) <= v for k, v in profile.limits.items()):
                WorkStage.objects.filter(pk=pk, profile_id=profile_id, state="ready").update(
                    profile=profile, waiting_reason=""
                )
                break
        else:
            WorkStage.objects.filter(pk=pk, state="ready").update(
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
    host, _ = ExecutorHost.objects.select_for_update(no_key=True).get_or_create(
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


def _preempt_background(stage, resources, active, host):
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
            ).fits:
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
        # Only this host's own GPU attempts can make room by finishing. Memory held
        # by other users' processes is not ours to drain: holding smaller GPU work
        # back for it idled the GPU for as long as those processes ran.
        and resources.get("vram", 0) + max(2 * 1024**3, host.telemetry.get("vram_total", 0) // 10)
        <= host.telemetry.get("vram_free", 0)
        + sum(value.get("vram", 0) for value in host.telemetry.get("attempts", {}).values())
    )


def _offline_input_stages(stages, offline_hosts, reachable_hosts):
    """IDs of ``stages`` with an input whose only ready copies are on offline hosts.

    Resolved from the runnable stages' own inputs. An offline GPU host holds
    millions of ready locations; deriving the stranded set from the catalogue
    took longer than the claim statement limit, so no host could claim at all.
    """
    inputs = list(
        ArtifactDependency.objects.filter(stage__in=stages.values("pk")).values_list(
            "stage_id", "artifact_id"
        )
    )
    stranded = set(
        ArtifactLocation.objects.filter(
            artifact_id__in={artifact for _, artifact in inputs},
            ready=True,
            host_id__in=offline_hosts,
        ).values_list("artifact_id", flat=True)
    )
    stranded -= set(
        ArtifactLocation.objects.filter(
            artifact_id__in=stranded, ready=True, host_id__in=reachable_hosts
        ).values_list("artifact_id", flat=True)
    )
    return {stage for stage, artifact in inputs if artifact in stranded}


@transaction.atomic
def claim(host_id: str):
    attempts = claim_batch(host_id, 1)
    return attempts[0] if attempts else None


# One ranked scan can admit several fitting stages. Re-ranking the whole
# backlog for every slot made admissions, not resources, the throughput limit.
CLAIM_BATCH_LIMIT = 8
CLAIM_BATCH_SECONDS = 2.0


@transaction.atomic
def claim_batch(host_id: str, limit: int = 1):
    """Admit up to ``limit`` stages under one host lock and one candidate ranking.

    Every admission repeats the exact per-stage checks against a reservation
    ledger that already includes the attempts admitted earlier in this batch.
    """
    require_postgres()
    limit = max(1, min(int(limit), CLAIM_BATCH_LIMIT))
    began = time.monotonic()
    # This is a short control request, not an analytical query. JIT compilation
    # of the large compatibility predicate can itself exceed the HTTP deadline.
    with connection.cursor() as cursor:
        cursor.execute(
            "SET LOCAL jit = off; SET LOCAL max_parallel_workers_per_gather = 0; "
            "SET LOCAL work_mem = '16MB'; SET LOCAL statement_timeout = '3000ms'"
        )
    host = ExecutorHost.objects.select_for_update(no_key=True).get(pk=host_id)
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
        return []
    attempts = list(
        host.attempts.filter(state__in=LIVE_STATES)
        .select_related("stage__execution")
        .defer("stage__execution__specification", "stage__payload", "stage__checkpoint")
    )
    if any(a.stop_requested for a in attempts):
        return []  # Confirm termination before reallocating any reservation.
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
    available_profiles = list(
        ResourceProfile.objects.filter(validated=True, suspended=False).defer("evidence")
    )
    if host.gpu_paused:
        available_profiles = [p for p in available_profiles if not p.resources.get("vram")]
    advertised_fingerprints = set(host.capabilities.get("profiles", []))
    advertised = [p for p in available_profiles if p.fingerprint in advertised_fingerprints]
    retained, projected = _after_stopping(
        active, [row["id"] for row in active if row["background"]], host.telemetry
    )
    # A busy host that cannot fit any advertised profile cannot admit work or
    # preempt for a foreground request. Avoid ranking its entire backlog just
    # to return an empty claim. The ordinary path still checks actual demand,
    # profile compatibility and ownership before preemption or admission.
    if attempts and not any(
        admit(resources, active, host.budgets, host.telemetry, gpu=gpu).fits
        or admit(resources, retained, host.budgets, projected, gpu=gpu).fits
        for p in advertised
        for resources in [{k: max(v, p.runtime_floor.get(k, v)) for k, v in p.resources.items()}]
        for gpu in [bool(resources.get("vram"))]
    ):
        return []
    methods = {(p.stage_kind, p.method) for p in advertised}
    eligible = Q(
        profile_id__in=[
            p.pk
            for p in available_profiles
            if p.fingerprint in advertised_fingerprints or (p.stage_kind, p.method) in methods
        ]
    )
    blocked = WorkStage.dependencies.through.objects.filter(
        from_workstage_id=OuterRef("pk")
    ).exclude(to_workstage__state="completed")
    reachable_hosts = [
        host_id,
        *ExecutorHost.objects.filter(heartbeat_at__gte=now - timedelta(seconds=10)).values_list(
            "pk", flat=True
        ),
    ]
    # Locations reference registered hosts. Resolve the tiny host set first;
    # a negative host predicate otherwise scans the entire location catalogue
    # just to prove that every host is online.
    offline_hosts = list(
        ExecutorHost.objects.exclude(pk__in=reachable_hosts).values_list("pk", flat=True)
    )
    runnable = (
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
    )
    # Availability is a property of locations, not of every candidate/artifact
    # pair. The old nested correlated EXISTS ran millions of checks just to
    # order a backlog, including when every host was online.
    offline_stages = set()
    if ArtifactLocation.objects.filter(ready=True, host_id__in=offline_hosts).exists():
        offline_stages = _offline_input_stages(runnable, offline_hosts, reachable_hosts)
    offline = Value(False, output_field=BooleanField())
    if offline_stages:
        offline = Case(When(pk__in=offline_stages, then=Value(True)), default=Value(False))
    active_executions = ExecutionAttempt.objects.filter(state__in=LIVE_STATES).values(
        "stage__execution_id"
    )
    candidates = (
        runnable.annotate(
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
            # Final assembly can only run on production. Serving it first there
            # turns finished computation into downloadable jobs; other hosts
            # can take the remaining CPU work.
            other_hosts_can_run=Case(
                When(kind="assemble", then=Value(host_id != "production")),
                default=Value(True),
            ),
        )
        .select_related("execution__job", "profile")
        .defer("execution__specification", "profile__evidence")
        .order_by(
            "effective_background",
            "-effective_priority",
            "other_hosts_can_run",
            "has_offline_input",
            "job_has_active",
            Coalesce(
                "oldest_consumer_at", "execution__job__submission_time", "execution__created_at"
            ),
            "created_at",
        )
    )
    # A page full of resource-blocked GPU work must not hide CPU work or a
    # compatible GPU pair beyond that page. Keep the oldest candidate of each
    # class for its waiting reason/fairness barrier, and preselect fitting
    # profiles before applying the bounded scan.
    cpu_stage = Q(profile__resources__vram__isnull=True) | Q(profile__resources__vram__lte=0)
    # One bounded top-1 per class. DISTINCT ON had to sort the whole eligible
    # backlog and was the most expensive statement of every claim.
    head_order = (
        "effective_background",
        "-effective_priority",
        Coalesce("oldest_consumer_at", "execution__job__submission_time", "execution__created_at"),
        "created_at",
    )
    oldest = [
        candidates.filter(allocation).order_by(*head_order).values_list("pk", flat=True).first()
        for allocation in (cpu_stage, Q(profile__resources__vram__gt=0))
    ]
    fitting = []
    foreground_fitting = []
    for p in advertised:
        resources = {k: max(v, p.runtime_floor.get(k, v)) for k, v in p.resources.items()}
        if admit(
            resources, active, host.budgets, host.telemetry, gpu=bool(resources.get("vram"))
        ).fits:
            fitting.append(p)
        if admit(
            resources, retained, host.budgets, projected, gpu=bool(resources.get("vram"))
        ).fits:
            foreground_fitting.append(p)

    # A large CPU-only backlog must not continually take the CPU capacity
    # needed to start GPU workers. Reserve this share only while queued GPU
    # work could fit after background allocations finish. Foreground requests
    # retain their full admission/preemption rights.
    gpu_waiting = oldest[1] is not None and any(p.resources.get("vram") for p in foreground_fitting)
    cpu_only_used = sum(
        row["resources"].get("cpu", 0) for row in active if not row["resources"].get("vram")
    )

    # Carve the GPU share from the CPUs actually reservable now. With overcommit
    # the nominal budget can exceed them, which let CPU-only work take the CPUs
    # GPU workers needed whenever other host load rose.
    cpu_only_limit = cpu_capacity(host.budgets, host.telemetry) - host.budgets.get(
        "gpu_cpu_reserve", 0
    )

    def background_cpu_fits(resources):
        return (
            not gpu_waiting
            or resources.get("vram")
            or cpu_only_used + resources.get("cpu", 0) <= cpu_only_limit
        )

    fitting = [
        p
        for p in fitting
        if background_cpu_fits(
            {key: max(value, p.runtime_floor.get(key, value)) for key, value in p.resources.items()}
        )
    ]

    def profile_capacity(profiles):
        # Resolve the small profile registry once, not large JSON identities for
        # each of tens of thousands of queued stages. Keep the exact admission
        # and compatibility checks below, after locking a candidate.
        ids = {p.pk for p in profiles}
        for p in profiles:
            if not p.resources.get("vram"):
                identity = p.settings.get("model_identity")
                ids.update(
                    source.pk
                    for source in available_profiles
                    if not source.resources.get("vram")
                    and source.fingerprint not in advertised_fingerprints
                    and source.stage_kind == p.stage_kind
                    and source.method == p.method
                    and source.settings.get("model_identity") == identity
                )
        return Q(profile_id__in=ids)

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
        candidates.values_list(
            "pk",
            "has_offline_input",
            "effective_background",
            "effective_priority",
            "other_hosts_can_run",
            "job_has_active",
            "execution_id",
            "profile__resources__vram",
        )[:512]
    )
    admitted, admitted_executions = [], set()

    def rank(index):
        # The database order, with ``job_has_active`` refreshed for requests that
        # received an attempt earlier in this batch, exactly as a new claim
        # would rank them. The original index preserves every later key.
        # Within a class and priority, GPU work is examined first: the device
        # is the scarce resource, and CPU work fills whatever capacity remains
        # instead of filling a batch before any waiting GPU stage is reached.
        _, offline, background, priority, other_hosts, has_active, execution, vram = candidate_rows[
            index
        ]
        return (
            background,
            -priority,
            other_hosts,
            not (isinstance(vram, (int, float)) and vram > 0),
            offline,
            has_active or execution in admitted_executions,
            index,
        )

    order, position = sorted(range(len(candidate_rows)), key=rank), 0
    while position < len(order):
        ident, has_offline_input, background, *_ = candidate_rows[order[position]]
        position += 1
        if len(admitted) >= limit or (admitted and time.monotonic() - began >= CLAIM_BATCH_SECONDS):
            break
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
        if stage.kind == "embedding" and not profile.resources.get("vram"):
            stage.waiting_reason = "profiling_required"
            stage.save(update_fields=["waiting_reason"])
            continue
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
            # Prefer the smallest reservation whose measured envelope covers
            # this input; an arbitrary first match could be one that cannot
            # admit the stage at all, or a wider envelope's larger reservation.
            shape = stage.payload.get("shape", {})
            profile = next(
                (
                    p
                    for p in sorted(
                        alternatives,
                        key=lambda p: (p.resources.get("ram", 0), p.resources.get("cpu", 0), p.pk),
                    )
                    if p.settings.get("step") == stage.payload.get("step")
                    and stage.target in p.settings.get("targets", [stage.target])
                    and p.settings.get("model_identity")
                    == stage.profile.settings.get("model_identity")
                    and (
                        not stage.failures
                        or p.settings.get("batch_size", 1)
                        <= stage.profile.settings.get("batch_size", 1)
                    )
                    and not any(shape.get(k, float("inf")) > v for k, v in p.limits.items())
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
        if background and not background_cpu_fits(resources):
            stage.waiting_reason = "waiting_cpu"
            stage.save(update_fields=["waiting_reason"])
            continue
        # Reject unavailable capacity before
        # per-artifact queries. A saturated 5,000-job queue must not spend
        # thousands of queries holding the host lock and delay heartbeats.
        base = admit(
            resources, active, host.budgets, host.telemetry, gpu=bool(profile.resources.get("vram"))
        )
        if not base.fits:
            if not background and admitted:
                # Never ask background work admitted moments ago to yield. The
                # next claim preempts against a ledger that includes this batch.
                break
            if not background and _preempt_background(stage, resources, active, host):
                return []
            gpu_barrier |= _admission_wait(
                stage,
                profile,
                resources,
                host,
                base.reason,
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
        cached_output_bytes = (
            Artifact.objects.filter(
                pk__in=stage.payload.get("output_artifacts", []),
            )
            .filter(Exists(ArtifactLocation.objects.filter(artifact_id=OuterRef("pk"), ready=True)))
            .exclude(pk__in=stage.artifacts.values("artifact_id"))
            .aggregate(size=Sum("size"))["size"]
            or 0
        )
        input_bytes += cached_output_bytes
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
        if not result.fits:
            if not background and admitted:
                break
            if not background and _preempt_background(stage, resources, active, host):
                return []
            gpu_barrier |= _admission_wait(
                stage,
                profile,
                resources,
                host,
                result.reason,
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
        shared = (
            stage.execution.specification.get("shared_protein_batch")
            if not stage.execution.job_id
            else None
        )
        if shared:
            Job.objects.filter(
                execution__pk__in=[row["id"] for row in shared["source_requests"]],
                status="Pending",
                start_time__isnull=True,
            ).update(start_time=admitted_at, status="Processing")
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
        admitted.append(serialize_attempt(attempt))
        # Later candidates in this batch see this reservation as fully unspent,
        # exactly as a subsequent separate claim would.
        active.append(
            {
                "id": attempt.pk,
                "started_at": admitted_at,
                "background": background,
                "resources": resources,
                "observed": {},
                "profile_fingerprint": profile.fingerprint,
            }
        )
        if not resources.get("vram"):
            cpu_only_used += resources.get("cpu", 0)
        if stage.execution_id not in admitted_executions:
            admitted_executions.add(stage.execution_id)
            order[position:] = sorted(order[position:], key=rank)
    return admitted


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
    outputs = stage.payload.get("outputs", [])
    ready_outputs = {
        artifact.pk: artifact
        for artifact in Artifact.objects.filter(pk__in=[o["id"] for o in outputs])
        .filter(Exists(ArtifactLocation.objects.filter(artifact_id=OuterRef("pk"), ready=True)))
        .only("id", "checksum", "size")
        .prefetch_related(
            Prefetch(
                "locations",
                queryset=ArtifactLocation.objects.filter(ready=True),
                to_attr="verified_locations",
            )
        )
    }
    for output in outputs:
        artifact = ready_outputs.get(output["id"])
        if artifact:
            artifacts.append(
                {
                    "id": artifact.pk,
                    "checksum": artifact.checksum,
                    "size": artifact.size,
                    "path": output["path"],
                    "hosts": [location.host_id for location in artifact.verified_locations],
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
        "artifact_protocol": 2,
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
    with connection.cursor() as cursor:
        cursor.execute("SET LOCAL jit = off; SET LOCAL max_parallel_workers_per_gather = 0")
    attempt = _owned(host_id, attempt_id, data["token"])
    now = timezone.now()
    if attempt.state not in {"leased", "running"} or attempt.lease_until <= now:
        raise ValueError("Attempt lease expired; executor must stop")
    first_running_heartbeat = attempt.state == "leased"
    attempt.heartbeat_at = now
    attempt.lease_until = now + timedelta(seconds=30)
    attempt.state = "running"
    attempt.container_id = data.get("container_id", attempt.container_id)
    attempt.save(update_fields=["heartbeat_at", "lease_until", "state", "container_id"])
    stage = attempt.stage
    progress = max(stage.progress, int(data.get("progress", 0)))
    advanced = progress > stage.progress
    if advanced:
        stage.progress = progress
        stage.save(update_fields=["progress"])
    if first_running_heartbeat or advanced:
        from api.execution.progress import request_refresh

        request_refresh(stage.execution)
    return {"lease_seconds": 30}


HEARTBEAT_BATCH_LIMIT = 256


def heartbeats(host_id, data):
    """Renew many leases in one request; each renewal keeps its own transaction.

    Per-attempt failures are reported with the status the single-attempt
    endpoint would have returned, so one stale lease never blocks the others.
    """
    from django.core.exceptions import ObjectDoesNotExist
    from django.db import OperationalError

    entries = data.get("attempts")
    if not isinstance(entries, list) or len(entries) > HEARTBEAT_BATCH_LIMIT:
        raise ValueError(f"Expected at most {HEARTBEAT_BATCH_LIMIT} heartbeats")
    results = {}
    for entry in entries:
        if not isinstance(entry, dict) or not isinstance(entry.get("id"), str):
            raise ValueError("Each heartbeat needs an attempt id")
        ident = entry["id"]
        try:
            results[ident] = heartbeat(host_id, uuid.UUID(ident), entry)
        except PermissionError:
            results[ident] = {"status": 403, "error": "ownership mismatch"}
        except ObjectDoesNotExist:
            results[ident] = {"status": 404, "error": "unknown execution record"}
        except OperationalError as exc:
            if getattr(exc.__cause__, "pgcode", None) != "57014":
                raise
            results[ident] = {"status": 503, "error": "control temporarily busy"}
        except (ValueError, KeyError, TypeError) as exc:
            results[ident] = {"status": 409, "error": str(exc)}
    return {"results": results}


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
        if (
            artifact.retention == "transient"
            and attempt.stage.execution.job_id is None
            and not attempt.stage.execution.specification.get("shared_protein_batch")
        ):
            artifact.retention = "protected"
        artifact.save(update_fields=["checksum", "size", "retention"])
        ArtifactLocation.objects.update_or_create(
            artifact=artifact,
            host_id=host_id,
            defaults={"ready": True, "verified_at": timezone.now()},
        )
    return {"checkpointed": len(data.get("artifacts", []))}


@transaction.atomic
def artifact_locations(host_id, attempt_id, data):
    """Validate a bounded location manifest once per attempt, then publish atomically."""
    attempt = _owned(host_id, attempt_id, data["token"])
    if attempt.state not in {"leased", "running"} or attempt.lease_until <= timezone.now():
        raise ValueError("Stale attempt cannot change artifact availability")
    rows = data.get("locations")
    if not isinstance(rows, list) or not 1 <= len(rows) <= 1024:
        raise ValueError("Expected 1..1024 artifact locations")
    ids = {row["artifact_id"] for row in rows}
    allowed = set(
        attempt.stage.artifacts.filter(artifact_id__in=ids).values_list("artifact_id", flat=True)
    )
    allowed.update(set(attempt.stage.payload.get("output_artifacts", [])) & ids)
    if ids != allowed:
        raise PermissionError("Artifact is outside this attempt")
    artifacts = {
        a.pk: a for a in Artifact.objects.select_for_update().filter(pk__in=ids).order_by("pk")
    }
    updates = {}
    for row in rows:
        artifact = artifacts[row["artifact_id"]]
        if artifact.checksum != row["checksum"]:
            raise ValueError("Artifact checksum changed")
        source_host = row.get("source_host", host_id)
        ready = row.get("ready")
        if ready is True:
            if source_host != host_id or row.get("size") != artifact.size:
                raise PermissionError("Only this executor can verify its own copy")
        elif ready is not False:
            raise ValueError("Artifact readiness must be boolean")
        updates[(artifact.pk, source_host)] = (artifact, ready)
    now = timezone.now()
    # All validation precedes writes. Confirmed absent remote locations are fenced
    # by the same ownership token as verified local arrivals.
    for source_host in {host for _, host in updates}:
        missing = [
            ident
            for (ident, host), (_, ready) in updates.items()
            if host == source_host and not ready
        ]
        if missing:
            ArtifactLocation.objects.filter(artifact_id__in=missing, host_id=source_host).update(
                ready=False
            )
    local = [
        ArtifactLocation(artifact=artifact, host_id=host_id, ready=True, verified_at=now)
        for (_, host), (artifact, ready) in updates.items()
        if ready and host == host_id
    ]
    if local:
        ArtifactLocation.objects.bulk_create(
            local,
            update_conflicts=True,
            unique_fields=["artifact", "host"],
            update_fields=["ready", "verified_at"],
            batch_size=1000,
        )
    return {"recorded": len(rows)}


def artifact_location(host_id, attempt_id, data):
    return artifact_locations(host_id, attempt_id, {"token": data["token"], "locations": [data]})


@transaction.atomic
def finish(host_id, attempt_id, data):
    with connection.cursor() as cursor:
        cursor.execute("SET LOCAL jit = off; SET LOCAL max_parallel_workers_per_gather = 0")
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
    from api.execution.artifact_policy import retention_for

    # Validate the complete manifest before writes. Fetch declared records once;
    # loading every large scientific identity again is unnecessary after hashing
    # the supplied identity against its declared immutable ID.
    for output in outputs:
        if (
            digest(output["identity"]) != output["id"]
            or len(output["checksum"]) != 64
            or output["size"] < 0
        ):
            raise ValueError("Invalid artifact manifest")
    existing = {
        artifact.pk: artifact
        for artifact in Artifact.objects.select_for_update()
        .filter(pk__in=expected)
        .defer("identity")
        .order_by("pk")
    }
    created, changed, locations = [], [], []
    standalone = stage.execution.job_id is None and not stage.execution.specification.get(
        "shared_protein_batch"
    )
    verified_at = timezone.now()
    for output in outputs:
        artifact = existing.get(output["id"])
        update = False
        if artifact is None:
            artifact = Artifact(
                id=output["id"],
                identity=output["identity"],
                checksum=output["checksum"],
                size=output["size"],
                metadata=output.get("metadata", {}),
            )
            created.append(artifact)
        elif artifact.checksum:
            if artifact.checksum != output["checksum"] or artifact.size != output["size"]:
                raise ValueError("Artifact identity conflict")
        else:
            if artifact.producer_id != stage.pk:
                raise ValueError("Artifact producer ownership mismatch")
            artifact.checksum, artifact.size = output["checksum"], output["size"]
            artifact.metadata = {**artifact.metadata, **output.get("metadata", {})}
            update = True
        retention = retention_for(output["identity"]) if artifact.regenerable else "permanent"
        if retention == "transient" and standalone:
            retention = "protected"
        if artifact.retention != retention:
            artifact.retention = retention
            update = True
        if artifact.pk in existing and update:
            changed.append(artifact)
        locations.append(
            ArtifactLocation(
                artifact=artifact,
                host_id=host_id,
                ready=True,
                verified_at=verified_at,
            )
        )
    if created:
        Artifact.objects.bulk_create(created, batch_size=256)
    if changed:
        Artifact.objects.bulk_update(
            changed, ["checksum", "size", "metadata", "retention"], batch_size=256
        )
    if locations:
        ArtifactLocation.objects.bulk_create(
            locations,
            batch_size=512,
            update_conflicts=True,
            unique_fields=["artifact", "host"],
            update_fields=["ready", "verified_at"],
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
    from api.execution.retention import release_consumed_residues

    release_consumed_residues(stage)
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
        from api.services.about_stats_service import mark_about_stats_cache_stale

        transaction.on_commit(mark_about_stats_cache_stale)
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
    from api.execution.progress import request_refresh

    request_refresh(stage.execution, force=stage.kind == "assemble")
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
        "missing_embedding",
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
    shared_unknown = stage.waiting_reason == "unknown" and stage.execution.specification.get(
        "shared_protein_batch"
    )
    if stage.payload.get("cohort_source_stages") and (
        stage.waiting_reason == "oom" or shared_unknown
    ):
        stage.state = "recovering"
        if shared_unknown:
            stage.payload["split_width"] = 1
            stage.save(update_fields=["payload"])
        if stage.waiting_reason == "oom" and stage.profile.settings.get("cohort_sources"):
            ResourceProfile.objects.filter(pk=stage.profile_id).update(suspended=True)
        ExecutionOutbox.objects.get_or_create(
            key=f"split-cohort:{attempt.pk}", defaults={"event": {"split_cohort": str(stage.pk)}}
        )
    if stage.waiting_reason == "missing_embedding" and stage.state != "held":
        stage.state = "recovering"
        ExecutionOutbox.objects.get_or_create(
            key=f"recover:{attempt.pk}",
            defaults={"event": {"recover_stage": str(stage.pk), "host_id": host_id}},
        )
    stage.save(update_fields=["failures", "waiting_reason", "available_at", "state", "profile"])
    if stage.state == "held" and stage.profile_id:
        ResourceProfile.objects.filter(pk=stage.profile_id).update(suspended=True)
    from api.execution.progress import request_refresh

    request_refresh(stage.execution)
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
    host = ExecutorHost.objects.select_for_update(no_key=True).get(pk=host_id)
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
