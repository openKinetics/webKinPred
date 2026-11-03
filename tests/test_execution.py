from __future__ import annotations

import io
from concurrent.futures import ThreadPoolExecutor
from datetime import timedelta

import pytest
from api.execution import control
from api.execution.artifacts import ArtifactStore
from api.execution.models import ExecutionAttempt, ExecutorHost, ResourceProfile, WorkStage
from api.execution.resources import GIB, admit
from api.models import Job
from django.db import close_old_connections
from django.test import override_settings
from django.utils import timezone


def test_reservation_accounts_for_models_that_have_not_loaded():
    active = [
        {
            "resources": {"ram": 3 * GIB, "vram": 8 * GIB, "cpu": 1, "disk": 0},
            "observed": {"ram": 3 * GIB, "vram": GIB},
        }
    ]
    result = admit(
        {"ram": GIB, "cpu": 1, "disk": 0, "vram": 8 * GIB},
        active,
        {"ram": 8 * GIB, "cpu": 4, "disk": GIB},
        {
            "ram_available": 8 * GIB,
            "disk_free": 8 * GIB,
            "gpu_uuid": "gpu",
            "vram_total": 20 * GIB,
            "vram_free": 16 * GIB,
        },
        gpu=True,
    )
    assert not result.fits
    assert result.reason == "waiting_vram"


@pytest.mark.parametrize("cpu_slots", [1, 0, -1, True, 1.5])
def test_canary_cpu_limit_does_not_consume_gpu_slots(cpu_slots):
    candidate = {"ram": GIB, "cpu": 1, "disk": GIB, "vram": 0}
    budgets = {"ram": 16 * GIB, "cpu": 8, "disk": 16 * GIB, "cpu_slots": cpu_slots}
    telemetry = {"ram_available": 32 * GIB, "disk_free": 32 * GIB}
    gpu_attempt = {"resources": {**candidate, "vram": GIB}, "observed": {}}
    cpu_attempt = {"resources": candidate, "observed": {}}
    result = admit(candidate, [gpu_attempt, cpu_attempt], budgets, telemetry, gpu=False)
    assert not result.fits
    assert result.reason == (
        "waiting_cpu" if cpu_slots == 1 and not isinstance(cpu_slots, bool) else "invalid_budget"
    )
    if cpu_slots == 1 and not isinstance(cpu_slots, bool):
        assert admit(candidate, [gpu_attempt], budgets, telemetry, gpu=False).fits
        assert admit(candidate, [], budgets, telemetry, gpu=False).fits


def test_artifact_partial_transfer_never_publishes(tmp_path):
    store = ArtifactStore(tmp_path)
    with pytest.raises(ValueError):
        store.publish(io.BytesIO(b"partial"), "f" * 64, 100)
    assert not store.path("f" * 64).exists()
    assert not list(tmp_path.rglob(".transfer-*"))
    with pytest.raises(ValueError):
        store.path("../bad")


@pytest.fixture
def ready_work(db):
    if control.connection.vendor != "postgresql":
        pytest.skip("Concurrency tests require PostgreSQL")
    host = ExecutorHost.objects.create(
        id="test",
        paused=False,
        reconciled=True,
        heartbeat_at=timezone.now(),
        budgets={"ram": 8 * GIB, "cpu": 4, "disk": 8 * GIB},
        telemetry={"ram_available": 16 * GIB, "disk_free": 32 * GIB},
        capabilities={"profiles": ["f" * 64]},
    )
    profile = ResourceProfile.objects.create(
        id="cpu",
        fingerprint="f" * 64,
        image="sha256:" + "a" * 64,
        stage_kind="predict",
        resources={"ram": 6 * GIB, "cpu": 2, "disk": GIB},
        validated=True,
    )
    stages = []
    for i in range(2):
        job = Job.objects.create(public_id=f"test{i}", prediction_type="kcat", status="Pending")
        execution = control.ingest(job.public_id, {"targets": ["kcat"]}, "a" * 64, paused=False)
        execution.stages.all().delete()
        stages.append(
            WorkStage.objects.create(
                execution=execution, key="cpu", kind="predict", profile=profile
            )
        )
    return host, stages


@pytest.mark.django_db(transaction=True)
def test_claim_is_atomic_across_concurrent_agents(ready_work):
    def claim():
        close_old_connections()
        try:
            return control.claim("test")
        finally:
            close_old_connections()

    with override_settings(EXECUTION_V2_ENABLED=True):
        with ThreadPoolExecutor(max_workers=2) as pool:
            claimed = list(pool.map(lambda _: claim(), range(2)))
    assert sum(x is not None for x in claimed) == 1
    assert ExecutionAttempt.objects.count() == 1


@pytest.mark.django_db(transaction=True)
def test_large_artifacts_prefer_ready_local_host_but_never_wait_indefinitely(ready_work):
    from api.execution.models import Artifact, ArtifactDependency, ArtifactLocation

    host, stages = ready_work
    stage = stages[0]
    stages[1].execution.paused = True
    stages[1].execution.save(update_fields=["paused"])
    peer = ExecutorHost.objects.create(
        id="peer",
        paused=False,
        reconciled=True,
        heartbeat_at=timezone.now(),
        budgets=host.budgets,
        telemetry=host.telemetry,
        capabilities=host.capabilities,
    )
    artifact = Artifact.objects.create(
        id="a" * 64, identity={}, checksum="b" * 64, size=64 * 1024**2
    )
    ArtifactDependency.objects.create(stage=stage, artifact=artifact, path="embedding.npy")
    ArtifactLocation.objects.create(artifact=artifact, host=peer, ready=True)
    with override_settings(EXECUTION_V2_ENABLED=True):
        assert control.claim(host.pk) is None
        WorkStage.objects.filter(pk=stage.pk).update(
            created_at=timezone.now() - timedelta(seconds=4)
        )
        assert control.claim(host.pk) is not None


@pytest.mark.django_db(transaction=True)
def test_claim_reserves_input_copies_and_transfers_before_admission(ready_work):
    from api.execution.models import Artifact, ArtifactDependency, ArtifactLocation

    host, stages = ready_work
    stages[1].execution.paused = True
    stages[1].execution.save(update_fields=["paused"])
    artifact = Artifact.objects.create(
        id="d" * 64,
        identity={"large_input": True},
        checksum="e" * 64,
        size=4 * GIB,
        regenerable=False,
    )
    ArtifactLocation.objects.create(artifact=artifact, host=host, ready=True)
    ArtifactDependency.objects.create(artifact=artifact, stage=stages[0], path="input.npy")
    with override_settings(EXECUTION_V2_ENABLED=True):
        assert control.claim(host.pk) is None
    stages[0].refresh_from_db()
    assert stages[0].waiting_reason == "waiting_disk"
    assert not ExecutionAttempt.objects.exists()


@pytest.mark.django_db(transaction=True)
def test_claim_never_uses_an_envelope_measured_for_another_target(ready_work):
    host, stages = ready_work
    profile = stages[0].profile
    profile.settings = {"targets": ["kcat"]}
    profile.save(update_fields=["settings"])
    for stage in stages:
        stage.target = "Km"
        stage.save(update_fields=["target"])
    with override_settings(EXECUTION_V2_ENABLED=True):
        assert control.claim(host.pk) is None
    assert not ExecutionAttempt.objects.exists()


@pytest.mark.django_db(transaction=True)
@pytest.mark.parametrize("recovery", ["peer_returns", "local_copy"])
def test_remote_artifact_outage_waits_without_launch_or_invalidating_data(ready_work, recovery):
    from api.execution.models import Artifact, ArtifactDependency, ArtifactLocation

    host, stages = ready_work
    stages[1].execution.paused = True
    stages[1].execution.save(update_fields=["paused"])
    peer = ExecutorHost.objects.create(
        id="peer", heartbeat_at=timezone.now() - timedelta(minutes=5)
    )
    artifact = Artifact.objects.create(
        id="9" * 64, identity={}, checksum="8" * 64, size=12, regenerable=False
    )
    ArtifactDependency.objects.create(stage=stages[0], artifact=artifact, path="input.csv")
    location = ArtifactLocation.objects.create(artifact=artifact, host=peer, ready=True)
    partial = ArtifactLocation.objects.create(artifact=artifact, host=host, ready=False)
    with override_settings(EXECUTION_V2_ENABLED=True):
        assert control.claim(host.pk) is None
        stages[0].refresh_from_db()
        location.refresh_from_db()
        assert stages[0].state == "ready"
        assert stages[0].waiting_reason == "network"
        assert stages[0].failures == 0
        assert location.ready
        assert not ExecutionAttempt.objects.exists()
        if recovery == "peer_returns":
            peer.heartbeat_at = timezone.now()
            peer.save(update_fields=["heartbeat_at"])
        else:
            partial.ready = True
            partial.save(update_fields=["ready"])
        assert control.claim(host.pk)["stage_id"] == str(stages[0].pk)


@pytest.mark.django_db(transaction=True)
def test_offline_artifact_backlog_does_not_hide_reachable_work(ready_work):
    from api.execution.models import Artifact, ArtifactDependency, ArtifactLocation

    host, available = ready_work
    job = Job.objects.create(
        public_id="offlinepg",
        prediction_type="kcat",
        status="Pending",
        submission_time=timezone.now() - timedelta(days=1),
    )
    execution = control.ingest(job.public_id, {}, "b" * 64, paused=False)
    peer = ExecutorHost.objects.create(id="offline")
    artifact = Artifact.objects.create(
        id="6" * 64, identity={}, checksum="7" * 64, size=12, regenerable=False
    )
    ArtifactLocation.objects.create(artifact=artifact, host=peer, ready=True)
    stages = WorkStage.objects.bulk_create(
        [
            WorkStage(
                execution=execution,
                key=f"offline:{i}",
                kind="predict",
                profile=available[0].profile,
            )
            for i in range(513)
        ]
    )
    ArtifactDependency.objects.bulk_create(
        [ArtifactDependency(stage=stage, artifact=artifact, path="input.csv") for stage in stages]
    )
    with override_settings(EXECUTION_V2_ENABLED=True):
        claimed = control.claim(host.pk)
    assert claimed["stage_id"] == str(available[0].pk)
    assert WorkStage.objects.filter(execution=execution, state="ready").count() == 514


@pytest.mark.django_db(transaction=True)
def test_retired_profiles_do_not_keep_attention_alert_active(ready_work):
    from api.execution.metrics import render

    profile = ready_work[1][0].profile
    profile.suspended = True
    profile.save(update_fields=["suspended"])
    assert "webkinpred_execution_suspended_profiles 1" in render()
    WorkStage.objects.filter(profile=profile).update(state="completed")
    assert "webkinpred_execution_suspended_profiles 0" in render()
    WorkStage.objects.filter(pk=ready_work[1][0].pk).update(state="held")
    assert "webkinpred_execution_suspended_profiles 1" in render()


@pytest.mark.django_db(transaction=True)
def test_executor_recovery_clock_is_durable_and_cannot_be_supplied_by_agent(ready_work):
    host, _ = ready_work
    config = {host.pk: {"budgets": host.budgets}}
    with override_settings(EXECUTION_HOSTS=config):
        first = control.register(host.pk, {"telemetry": {"control_recovered_at": "spoof"}})
        stamp = first.telemetry["control_recovered_at"]
        assert stamp != "spoof"
        assert (
            control.register(host.pk, {"telemetry": {}}).telemetry["control_recovered_at"] == stamp
        )
        ExecutorHost.objects.filter(pk=host.pk).update(
            heartbeat_at=timezone.now() - timedelta(seconds=31)
        )
        recovered = control.register(host.pk, {"telemetry": {"control_recovered_at": stamp}})
        assert recovered.telemetry["control_recovered_at"] > stamp


@pytest.mark.django_db(transaction=True)
def test_expired_lease_time_is_resource_wait_after_host_disconnect(ready_work):
    from api.execution.models import ExecutionSpec
    from api.execution.status import execution_status

    host, stages = ready_work
    now = timezone.now()
    execution = ExecutionSpec.objects.create(
        job=Job.objects.create(public_id="leasegap", prediction_type="kcat", status="Processing"),
        specification={},
        checksum="a" * 64,
        input_checksum="b" * 64,
        created_at=now - timedelta(seconds=120),
        paused=False,
    )
    stage = WorkStage.objects.create(execution=execution, key="predict", kind="predict")
    ExecutionAttempt.objects.create(
        stage=stage,
        host=host,
        state="stopped",
        started_at=now - timedelta(seconds=120),
        lease_until=now - timedelta(seconds=90),
        finished_at=now - timedelta(seconds=10),
    )
    with override_settings(EXECUTION_V2_ENABLED=True):
        state = execution_status(execution.job)
    assert state["active_execution_seconds"] == 30
    assert 89 <= state["resource_wait_seconds"] <= 91


@pytest.mark.django_db(transaction=True)
def test_oldest_submission_precedes_specification_import_order(ready_work):
    older = ready_work[1][1]
    Job.objects.filter(pk=older.execution.job_id).update(
        submission_time=timezone.now() - timedelta(days=1)
    )
    with override_settings(EXECUTION_V2_ENABLED=True):
        assert control.claim("test")["stage_id"] == str(older.pk)


@pytest.mark.django_db(transaction=True)
def test_verified_artifact_locations_are_fenced_by_attempt_ownership(ready_work):
    from api.execution.models import Artifact, ArtifactDependency, ArtifactLocation

    host, stages = ready_work
    peer = ExecutorHost.objects.create(id="peer", heartbeat_at=timezone.now())
    artifact = Artifact.objects.create(
        id="d" * 64, identity={"input": "test"}, checksum="e" * 64, size=12, regenerable=False
    )
    ArtifactLocation.objects.create(artifact=artifact, host=peer, ready=True)
    ArtifactDependency.objects.create(artifact=artifact, stage=stages[0], path="input.csv")
    with override_settings(EXECUTION_V2_ENABLED=True):
        attempt = control.claim(host.pk)
    data = {
        "token": attempt["token"],
        "artifact_id": artifact.pk,
        "checksum": artifact.checksum,
        "size": artifact.size,
        "ready": True,
    }
    control.artifact_location(host.pk, attempt["id"], data)
    assert artifact.locations.filter(host=host, ready=True).exists()
    with pytest.raises(PermissionError, match="own copy"):
        control.artifact_location(host.pk, attempt["id"], {**data, "source_host": peer.pk})
    with pytest.raises(PermissionError, match="outside this attempt"):
        control.artifact_location(host.pk, attempt["id"], {**data, "artifact_id": "f" * 64})
    ExecutionAttempt.objects.filter(pk=attempt["id"]).update(
        lease_until=timezone.now() - timedelta(seconds=1)
    )
    with pytest.raises(ValueError, match="Stale attempt"):
        control.artifact_location(host.pk, attempt["id"], {**data, "ready": False})
    assert artifact.locations.filter(host=host, ready=True).exists()


@pytest.mark.django_db(transaction=True)
def test_unavailable_dependency_returns_to_recovery_before_model_launch(ready_work):
    from api.execution.models import Artifact, ArtifactDependency, ExecutionOutbox

    host, stages = ready_work
    artifact = Artifact.objects.create(
        id="d" * 64, identity={"input": "test"}, checksum="e" * 64, size=12, regenerable=False
    )
    ArtifactDependency.objects.create(artifact=artifact, stage=stages[0], path="input.csv")
    with override_settings(EXECUTION_V2_ENABLED=True):
        other = control.claim(host.pk)
    assert other["stage_id"] == str(stages[1].pk)
    stages[0].refresh_from_db()
    assert stages[0].state == "recovering"
    assert not stages[0].attempts.exists()
    assert ExecutionOutbox.objects.filter(event__recover_stage=str(stages[0].pk)).exists()


@pytest.mark.django_db(transaction=True)
def test_expiry_does_not_free_reservations_and_stale_owner_cannot_commit(ready_work):
    with override_settings(EXECUTION_V2_ENABLED=True):
        attempt = control.claim("test")
        ExecutionAttempt.objects.filter(pk=attempt["id"]).update(
            lease_until=timezone.now() - timedelta(seconds=1)
        )
        assert control.claim("test") is None
        with pytest.raises(ValueError, match="Stale"):
            control.finish("test", attempt["id"], {"token": attempt["token"], "stopped": True})
        control.reconcile("test", {"running": []})
        assert ExecutionAttempt.objects.get(pk=attempt["id"]).state == "stopped"
        assert control.claim("test") is not None


@pytest.mark.django_db(transaction=True)
def test_unchanged_heartbeats_renew_ownership_without_reprojecting_job_progress(
    ready_work, monkeypatch
):
    from api.execution import progress
    from api.models import JobProgressStage

    host, stages = ready_work
    stage = stages[0]
    stage.target = "kcat"
    stage.payload = {"rows": 10, "shape": {"units": 10}}
    stage.save(update_fields=["target", "payload"])
    public = JobProgressStage.objects.create(
        job=stage.execution.job, stage_index=1, target="kcat", method_key="test"
    )
    calls = []
    refresh = progress.refresh

    def project(execution):
        calls.append(execution.pk)
        refresh(execution)

    monkeypatch.setattr(progress, "refresh", project)
    with override_settings(EXECUTION_V2_ENABLED=True):
        claimed = control.claim(host.pk)
    assert claimed["stage_id"] == str(stage.pk)
    for value in [0, 0, 4, 4, 2]:
        # An old near-expiry lease must be renewed even when the counter is unchanged.
        ExecutionAttempt.objects.filter(pk=claimed["id"]).update(
            lease_until=timezone.now() + timedelta(seconds=3)
        )
        control.heartbeat(host.pk, claimed["id"], {"token": claimed["token"], "progress": value})
        attempt = ExecutionAttempt.objects.get(pk=claimed["id"])
        assert attempt.state == "running" and attempt.lease_until > timezone.now() + timedelta(
            seconds=20
        )
    stage.refresh_from_db()
    public.refresh_from_db()
    assert len(calls) == 2
    assert stage.progress == public.predictions_made == 4
    assert public.status == "running" and public.predictions_total == 10


@pytest.mark.django_db(transaction=True)
def test_resource_failures_hold_instead_of_failing_jobs(ready_work):
    stage = ready_work[1][0]
    with override_settings(EXECUTION_V2_ENABLED=True):
        for _ in range(3):
            stage.available_at = timezone.now()
            stage.save(update_fields=["available_at"])
            attempt = control.claim("test")
            assert attempt["stage_id"] == str(stage.pk)
            control.fail(
                "test",
                attempt["id"],
                {"token": attempt["token"], "stopped": True, "category": "oom"},
            )
    stage.refresh_from_db()
    assert stage.state == "held"
    assert stage.execution.job.status != "Failed"
    stage.profile.refresh_from_db()
    assert stage.profile.suspended


@pytest.mark.django_db(transaction=True)
def test_complete_is_idempotent_and_requires_full_outputs(ready_work):
    with override_settings(EXECUTION_V2_ENABLED=True):
        attempt = control.claim("test")
    stage = WorkStage.objects.get(pk=attempt["stage_id"])
    stage.payload = {"output_artifacts": ["a" * 64]}
    stage.save()
    body = {"token": attempt["token"], "stopped": True, "artifacts": []}
    with pytest.raises(ValueError, match="Incomplete"):
        control.finish("test", attempt["id"], body)
    stage.payload = {}
    stage.save()
    assert control.finish("test", attempt["id"], body) == {"state": "completed"}
    assert control.finish("test", attempt["id"], body) == {"state": "completed"}


@pytest.mark.django_db(transaction=True)
def test_partial_checkpoint_survives_retry_without_releasing_live_reservation(ready_work):
    from api.execution.models import Artifact

    stage = ready_work[1][0]
    identity = {"sequence": "MAAA", "representation": "pinned-test"}
    artifact = Artifact.objects.create(
        pk=control.digest(identity), identity=identity, checksum="", size=0, producer=stage
    )
    stage.payload = {
        "outputs": [{"id": artifact.pk, "identity": identity, "path": "sequence_info/test.npy"}],
        "output_artifacts": [artifact.pk],
    }
    stage.save()
    with override_settings(EXECUTION_V2_ENABLED=True):
        attempt = control.claim("test")
        data = {
            "token": attempt["token"],
            "artifacts": [
                {"id": artifact.pk, "identity": identity, "checksum": "a" * 64, "size": 16}
            ],
        }
        assert control.checkpoint("test", attempt["id"], data)["checkpointed"] == 1
        assert control.claim("test") is None
        control.fail(
            "test",
            attempt["id"],
            {"token": attempt["token"], "stopped": True, "category": "interrupted"},
        )
        stage.available_at = timezone.now()
        stage.save(update_fields=["available_at"])
        retry = control.claim("test")
        assert retry["stage_id"] == str(stage.pk)
        assert retry["artifacts"][0]["checksum"] == "a" * 64
        with pytest.raises(ValueError, match="Stale"):
            control.checkpoint("test", attempt["id"], data)


@pytest.mark.django_db(transaction=True)
@pytest.mark.parametrize("blocked_by_dependency", [True, False])
def test_fitting_work_is_not_hidden_by_a_full_page_of_ineligible_stages(
    ready_work, blocked_by_dependency
):
    host, available = ready_work
    older = Job.objects.create(
        public_id="olderpage",
        prediction_type="kcat",
        status="Pending",
        submission_time=timezone.now() - timedelta(days=1),
    )
    execution = control.ingest(older.public_id, {"targets": ["kcat"]}, "b" * 64, paused=False)
    dependency = execution.stages.get(key="prepare")
    profile = available[0].profile
    if not blocked_by_dependency:
        profile = ResourceProfile.objects.create(
            id="unadvertised",
            fingerprint="u" * 64,
            image=profile.image,
            stage_kind="embedding",
            method="other-host",
            resources=profile.resources,
            validated=True,
        )
    stages = [
        WorkStage(
            execution=execution,
            key=f"blocked:{i}",
            kind=profile.stage_kind,
            method=profile.method,
            profile=profile,
        )
        for i in range(513)
    ]
    WorkStage.objects.bulk_create(stages)
    if blocked_by_dependency:
        WorkStage.dependencies.through.objects.bulk_create(
            [
                WorkStage.dependencies.through(
                    from_workstage_id=stage.pk, to_workstage_id=dependency.pk
                )
                for stage in stages
            ]
        )
    with override_settings(EXECUTION_V2_ENABLED=True):
        result = control.claim(host.pk)
    assert result is not None
    assert ExecutionAttempt.objects.get(pk=result["id"]).stage_id in {s.pk for s in available}


@pytest.mark.django_db(transaction=True)
def test_controller_bootstrap_creates_offline_hosts_without_resuming_them():
    budgets = {"ram": 8 * GIB, "cpu": 3, "disk": 8 * GIB}
    with override_settings(
        EXECUTION_HOSTS={"production": {"budgets": budgets}, "gpu": {"budgets": budgets}}
    ):
        control.ensure_hosts()
        control.ensure_hosts()
    assert ExecutorHost.objects.count() == 2
    for host in ExecutorHost.objects.all():
        assert host.paused and not host.reconciled and host.heartbeat_at is None
        assert host.budgets == budgets
    ExecutorHost.objects.filter(pk="production").update(paused=False)
    with override_settings(EXECUTION_HOSTS={"production": {"budgets": budgets}}):
        control.ensure_hosts()
    assert not ExecutorHost.objects.get(pk="production").paused


def test_cpu_prediction_can_overlap_gpu_execution_without_a_gpu_combination_profile():
    from types import SimpleNamespace

    cpu = SimpleNamespace(resources={"cpu": 1, "ram": GIB, "vram": 0}, fingerprint="cpu")
    gpu = SimpleNamespace(resources={"cpu": 1, "ram": GIB, "vram": GIB}, fingerprint="gpu2")
    active = [{"resources": {"vram": GIB}, "profile_fingerprint": "gpu1"}]
    host = SimpleNamespace(capabilities={"validated_gpu_combinations": []})
    assert control._gpu_profile_allowed(cpu, active, host)
    assert not control._gpu_profile_allowed(gpu, active, host)


@pytest.mark.django_db(transaction=True)
@pytest.mark.parametrize("already_waited", [False, True])
def test_gpu_drain_fairness_counts_eligible_wait_not_dependency_age(ready_work, already_waited):
    host, stages = ready_work
    older, fitting = stages
    now = timezone.now()
    common = dict(
        stage_kind="predict",
        validated=True,
        resources={
            "ram": GIB,
            "cpu": 1,
            "disk": GIB,
            "vram": 2 * GIB,
        },
    )
    ResourceProfile.objects.filter(pk=older.profile_id).update(**common)
    compatible = ResourceProfile.objects.create(
        id="compatible-gpu", fingerprint="g" * 64, image=older.profile.image, **common
    )
    fitting.profile = compatible
    fitting.save()
    host.budgets.update(ram=8 * GIB, gpu_slots=2)
    host.telemetry.update(gpu_uuid="gpu", vram_total=20 * GIB, vram_free=16 * GIB)
    host.capabilities.update(
        profiles=["f" * 64, "g" * 64], validated_gpu_combinations=[["g" * 64] * 2]
    )
    host.save()
    WorkStage.objects.filter(pk=older.pk).update(created_at=now - timedelta(hours=2))
    if already_waited:
        older.payload["admission_wait_started_at"] = (now - timedelta(minutes=11)).isoformat()
        older.save(update_fields=["payload"])
    parent = WorkStage.objects.create(
        execution=fitting.execution, key="active-gpu", state="running"
    )
    ExecutionAttempt.objects.create(
        stage=parent,
        host=host,
        resources=common["resources"],
        profile_fingerprint="g" * 64,
        lease_until=now + timedelta(seconds=30),
    )
    with override_settings(EXECUTION_V2_ENABLED=True):
        result = control.claim(host.pk)
    older.refresh_from_db()
    assert older.waiting_reason == "unvalidated_gpu_combination"
    assert "admission_wait_started_at" in older.payload
    if already_waited:
        assert result is None
    else:
        assert result["stage_id"] == str(fitting.pk)


@pytest.mark.django_db(transaction=True)
def test_saturated_claim_has_bounded_queries_and_does_not_rewrite_waiting_stages(ready_work):
    from django.db import connection
    from django.test.utils import CaptureQueriesContext

    host, stages = ready_work
    WorkStage.objects.bulk_create(
        [
            WorkStage(
                execution=stages[1].execution,
                key="waiting-" + str(i),
                kind="predict",
                profile=stages[1].profile,
            )
            for i in range(128)
        ]
    )
    with override_settings(EXECUTION_V2_ENABLED=True):
        assert control.claim(host.pk) is not None
        assert control.claim(host.pk) is None  # Publish the initial waiting reason once.
        with CaptureQueriesContext(connection) as captured:
            assert control.claim(host.pk) is None
    # Fixed availability lookups and fetching only inspected candidate contexts
    # avoid expensive correlated sorting; query count stays independent of backlog.
    assert len(captured) <= 16
    assert not any(query["sql"].startswith("UPDATE") for query in captured)


@pytest.mark.django_db(transaction=True)
@pytest.mark.parametrize("blocked_gpu", [False, True])
def test_513_resource_blocked_stages_cannot_hide_fitting_cpu_work(ready_work, blocked_gpu):
    host, stages = ready_work
    blocked, small = stages
    profile = blocked.profile
    profile.resources = {
        "cpu": 1,
        "ram": GIB if blocked_gpu else 20 * GIB,
        "disk": GIB,
        "vram": 2 * GIB if blocked_gpu else 0,
    }
    profile.save()
    fitting = ResourceProfile.objects.create(
        id="small",
        fingerprint="s" * 64,
        image=profile.image,
        stage_kind="predict",
        validated=True,
        resources={"cpu": 1, "ram": GIB, "disk": GIB, "vram": 0},
    )
    small.profile = fitting
    small.save()
    WorkStage.objects.bulk_create(
        [
            WorkStage(
                execution=blocked.execution,
                key="blocked-" + str(i),
                kind="predict",
                profile=profile,
            )
            for i in range(513)
        ]
    )
    host.budgets.update(ram=8 * GIB, cpu=8, disk=16 * GIB, gpu_slots=2)
    host.telemetry.update(gpu_uuid="gpu", vram_total=20 * GIB, vram_free=16 * GIB)
    host.capabilities["profiles"] += [fitting.fingerprint]
    host.save()
    if blocked_gpu:
        from api.execution.models import ExecutionSpec

        owner = ExecutionSpec.objects.create(
            specification={}, checksum="a" * 64, input_checksum="b" * 64, paused=False
        )
        active = WorkStage.objects.create(execution=owner, key="active", state="running")
        ExecutionAttempt.objects.create(
            stage=active,
            host=host,
            resources={"cpu": 1, "ram": GIB, "disk": GIB, "vram": 2 * GIB},
            profile_fingerprint="g" * 64,
            lease_until=timezone.now() + timedelta(seconds=30),
        )
    with override_settings(EXECUTION_V2_ENABLED=True):
        result = control.claim(host.pk)
    assert result["stage_id"] == str(small.pk)


@pytest.mark.django_db(transaction=True)
def test_attempt_input_manifest_prefetches_verified_locations(ready_work):
    from api.execution.models import Artifact, ArtifactDependency, ArtifactLocation
    from django.db import connection
    from django.test.utils import CaptureQueriesContext

    host, stages = ready_work
    for i in range(32):
        artifact = Artifact.objects.create(
            id=f"{i:064x}", identity={"i": i}, checksum="b" * 64, size=1
        )
        ArtifactDependency.objects.create(stage=stages[0], artifact=artifact, path=f"input{i}.npy")
        ArtifactLocation.objects.create(artifact=artifact, host=host, ready=True)
    with override_settings(EXECUTION_V2_ENABLED=True):
        request = control.claim(host.pk)
    attempt = ExecutionAttempt.objects.select_related(
        "stage__profile", "stage__execution__job"
    ).get(pk=request["id"])
    with CaptureQueriesContext(connection) as captured:
        manifest = control.serialize_attempt(attempt)
    assert len(manifest["artifacts"]) == 32
    assert all(row["hosts"] == [host.pk] for row in manifest["artifacts"])
    assert len(captured) <= 3


@pytest.mark.django_db(transaction=True)
def test_many_small_artifacts_prefer_live_local_peer_and_backfill_after_grace(ready_work):
    from api.execution.models import Artifact, ArtifactDependency, ArtifactLocation

    host, stages = ready_work
    stages[1].execution.paused = True
    stages[1].execution.save(update_fields=["paused"])
    peer = ExecutorHost.objects.create(
        id="peer",
        paused=False,
        reconciled=True,
        heartbeat_at=timezone.now(),
        budgets=host.budgets,
        telemetry=host.telemetry,
        capabilities=host.capabilities,
    )
    artifacts = [
        Artifact(id=f"{i:064x}", identity={}, checksum="b" * 64, size=1024) for i in range(200)
    ]
    Artifact.objects.bulk_create(artifacts)
    ArtifactDependency.objects.bulk_create(
        [
            ArtifactDependency(stage=stages[0], artifact=a, path=f"{i}.npy")
            for i, a in enumerate(artifacts)
        ]
    )
    ArtifactLocation.objects.bulk_create(
        [ArtifactLocation(artifact=a, host=peer, ready=True) for a in artifacts]
    )
    with override_settings(EXECUTION_V2_ENABLED=True):
        assert control.claim(host.pk) is None
        WorkStage.objects.filter(pk=stages[0].pk).update(
            created_at=timezone.now() - timedelta(seconds=31)
        )
        assert control.claim(host.pk) is not None


@pytest.mark.django_db(transaction=True)
def test_bulk_location_commit_is_atomic_bounded_and_owned(ready_work, django_assert_num_queries):
    from api.execution.models import Artifact, ArtifactDependency, ArtifactLocation

    host, stages = ready_work
    artifacts = [
        Artifact(id=f"{i:064x}", identity={}, checksum="b" * 64, size=12) for i in range(200)
    ]
    Artifact.objects.bulk_create(artifacts)
    ArtifactDependency.objects.bulk_create(
        [
            ArtifactDependency(stage=stages[0], artifact=a, path=f"{i}.npy")
            for i, a in enumerate(artifacts)
        ]
    )
    ArtifactLocation.objects.bulk_create(
        [ArtifactLocation(artifact=a, host=host, ready=True) for a in artifacts]
    )
    with override_settings(EXECUTION_V2_ENABLED=True):
        attempt = control.claim(host.pk)
    rows = [
        {"artifact_id": a.pk, "checksum": a.checksum, "size": 12, "ready": True} for a in artifacts
    ]
    ArtifactLocation.objects.all().update(ready=False)
    with django_assert_num_queries(9, exact=False):
        control.artifact_locations(
            host.pk, attempt["id"], {"token": attempt["token"], "locations": rows}
        )
    assert ArtifactLocation.objects.filter(ready=True).count() == 200
    rows[-1] = {**rows[-1], "checksum": "f" * 64}
    ArtifactLocation.objects.all().update(ready=False)
    with pytest.raises(ValueError, match="checksum"):
        control.artifact_locations(
            host.pk, attempt["id"], {"token": attempt["token"], "locations": rows}
        )
    assert not ArtifactLocation.objects.filter(ready=True).exists()


@pytest.mark.django_db(transaction=True)
def test_repeated_dependency_loss_never_suspends_cpu_profile(ready_work):
    from api.execution.models import ExecutionOutbox

    host, stages = ready_work
    stage = stages[0]
    for _ in range(3):
        stage.state = "ready"
        stage.available_at = timezone.now()
        stage.save(update_fields=["state", "available_at"])
        with override_settings(EXECUTION_V2_ENABLED=True):
            attempt = control.claim(host.pk)
        control.fail(
            host.pk,
            attempt["id"],
            {"token": attempt["token"], "stopped": True, "category": "missing_embedding"},
        )
        stage.refresh_from_db()
        assert stage.state == "recovering"
        stage.profile.refresh_from_db()
        assert not stage.profile.suspended
    assert ExecutionOutbox.objects.filter(event__has_key="recover_stage").count() == 3


@pytest.mark.django_db(transaction=True)
def test_repair_dependency_holds_is_idempotent_and_leaves_oom_holds(ready_work):
    from django.core.management import call_command

    host, stages = ready_work
    stage = stages[0]
    stage.state = "held"
    stage.waiting_reason = "missing_embedding"
    stage.save(update_fields=["state", "waiting_reason"])
    stage.profile.suspended = True
    stage.profile.save(update_fields=["suspended"])
    ExecutionAttempt.objects.create(
        stage=stage,
        host=host,
        state="completed",
        lease_until=timezone.now(),
        profile_fingerprint=stage.profile.fingerprint,
    )
    call_command("execution_repair_dependencies", "--dry-run", stdout=io.StringIO())
    stage.profile.refresh_from_db()
    stage.refresh_from_db()
    assert stage.profile.suspended and stage.state == "held"
    call_command("execution_repair_dependencies", stdout=io.StringIO())
    call_command("execution_repair_dependencies", stdout=io.StringIO())
    stage.profile.refresh_from_db()
    stage.refresh_from_db()
    assert not stage.profile.suspended and stage.state == "recovering"
    stage.state = "held"
    stage.waiting_reason = "oom"
    stage.save(update_fields=["state", "waiting_reason"])
    stage.profile.suspended = True
    stage.profile.save(update_fields=["suspended"])
    call_command("execution_repair_dependencies", stdout=io.StringIO())
    stage.profile.refresh_from_db()
    stage.refresh_from_db()
    assert stage.profile.suspended and stage.state == "held"


@pytest.mark.django_db(transaction=True)
def test_gpu_drain_leaves_cpu_admission_enabled_and_survives_registration(ready_work):
    from django.core.management import call_command

    host, stages = ready_work
    call_command("execution_admin", "drain-gpu", "--host", host.pk, stdout=io.StringIO())
    with override_settings(
        EXECUTION_V2_ENABLED=True, EXECUTION_HOSTS={host.pk: {"budgets": host.budgets}}
    ):
        control.register(host.pk, {"capabilities": host.capabilities, "telemetry": host.telemetry})
        host.refresh_from_db()
        assert host.gpu_paused
        assert control.claim(host.pk) is not None
    call_command("execution_admin", "resume-gpu", "--host", host.pk, stdout=io.StringIO())
    host.refresh_from_db()
    assert not host.gpu_paused


def test_validated_gpu_mixtures_allow_startup_subsets_and_preserve_multiplicity():
    from types import SimpleNamespace

    a, b, c = ["a" * 64, "b" * 64, "c" * 64]
    host = SimpleNamespace(capabilities={"validated_gpu_combinations": [[a, b, c]]})
    profile = SimpleNamespace(resources={"vram": 1}, fingerprint=b)
    active = [{"resources": {"vram": 1}, "profile_fingerprint": a}]
    assert control._gpu_profile_allowed(profile, active, host)
    active.append({"resources": {"vram": 1}, "profile_fingerprint": b})
    assert not control._gpu_profile_allowed(profile, active, host)
    host.capabilities["validated_gpu_combinations"] = [[a, b], [b, c]]
    profile.fingerprint = c
    assert not control._gpu_profile_allowed(profile, active, host)


@pytest.mark.django_db(transaction=True)
def test_serializing_many_checkpointed_outputs_uses_bounded_queries(
    ready_work, django_assert_num_queries
):
    from api.execution.models import Artifact, ArtifactLocation

    host, stages = ready_work
    with override_settings(EXECUTION_V2_ENABLED=True):
        record = control.claim(host.pk)
    stage = stages[0]
    artifacts = [
        Artifact(id=f"{i:064x}", identity={}, checksum="b" * 64, size=12) for i in range(200)
    ]
    Artifact.objects.bulk_create(artifacts)
    ArtifactLocation.objects.bulk_create(
        [ArtifactLocation(artifact=a, host=host, ready=True) for a in artifacts]
    )
    stage.payload = {"outputs": [{"id": a.pk, "path": f"{i}.npy"} for i, a in enumerate(artifacts)]}
    stage.save(update_fields=["payload"])
    attempt = ExecutionAttempt.objects.select_related(
        "stage__execution__job", "stage__profile"
    ).get(pk=record["id"])
    with django_assert_num_queries(4, exact=False):
        request = control.serialize_attempt(attempt)
    assert len(request["artifacts"]) == 200


@pytest.mark.django_db(transaction=True)
def test_claim_reserves_disk_for_verified_retry_outputs_too(ready_work):
    from api.execution.models import Artifact, ArtifactLocation

    host, stages = ready_work
    stages[1].execution.paused = True
    stages[1].execution.save(update_fields=["paused"])
    artifact = Artifact.objects.create(
        id="e" * 64, identity={}, checksum="f" * 64, size=4 * GIB, producer=stages[0]
    )
    ArtifactLocation.objects.create(artifact=artifact, host=host, ready=True)
    stages[0].payload = {
        "output_artifacts": [artifact.pk],
        "outputs": [{"id": artifact.pk, "path": "cached.npy"}],
    }
    stages[0].save(update_fields=["payload"])
    with override_settings(EXECUTION_V2_ENABLED=True):
        assert control.claim(host.pk) is None
    stages[0].refresh_from_db()
    assert stages[0].waiting_reason == "waiting_disk"
    assert not ExecutionAttempt.objects.exists()


@pytest.mark.django_db(transaction=True)
def test_cached_output_reservation_counts_ready_artifact_once_across_hosts(ready_work):
    from api.execution.models import Artifact, ArtifactLocation
    from django.db import connection
    from django.test.utils import CaptureQueriesContext

    host, stages = ready_work
    stages[1].execution.paused = True
    stages[1].execution.save(update_fields=["paused"])
    peer = ExecutorHost.objects.create(id="peer", heartbeat_at=timezone.now())
    ready = Artifact.objects.create(
        id="e" * 64,
        identity={"large_private_identity": "x" * 100000},
        checksum="a" * 64,
        size=2 * GIB,
        producer=stages[0],
    )
    absent = Artifact.objects.create(
        id="d" * 64, identity={}, checksum="b" * 64, size=10 * GIB, producer=stages[0]
    )
    for h in [host, peer]:
        ArtifactLocation.objects.create(artifact=ready, host=h, ready=True)
    ArtifactLocation.objects.create(artifact=absent, host=host, ready=False)
    stages[0].payload = {
        "output_artifacts": [ready.pk, absent.pk],
        "outputs": [{"id": ready.pk, "path": "ready.npy"}, {"id": absent.pk, "path": "absent.npy"}],
    }
    stages[0].save(update_fields=["payload"])
    with (
        override_settings(EXECUTION_V2_ENABLED=True),
        CaptureQueriesContext(connection) as captured,
    ):
        request = control.claim(host.pk)
    assert request and request["resources"]["disk"] == 5 * GIB
    assert len(request["artifacts"]) == 1
    assert request["artifacts"][0]["id"] == ready.pk
    assert set(request["artifacts"][0]["hosts"]) == {host.pk, peer.pk}
    sums = [q["sql"] for q in captured if "SUM(" in q["sql"] and "api_artifact" in q["sql"]]
    assert sums and all('"identity"' not in sql and '"metadata"' not in sql for sql in sums)
    selected = [q["sql"] for q in captured if q["sql"].startswith('SELECT "api_artifact".')]
    assert selected and all('"identity"' not in sql and '"metadata"' not in sql for sql in selected)


@pytest.mark.django_db(transaction=True)
def test_online_hosts_do_not_require_negative_location_catalogue_scan(ready_work):
    from django.db import connection
    from django.test.utils import CaptureQueriesContext

    host, _ = ready_work
    with (
        override_settings(EXECUTION_V2_ENABLED=True),
        CaptureQueriesContext(connection) as captured,
    ):
        assert control.claim(host.pk)
    assert not any(
        'FROM "api_artifactlocation"' in q["sql"]
        and "NOT (" in q["sql"]
        and '"host_id" IN' in q["sql"]
        for q in captured
    )


@pytest.mark.django_db(transaction=True)
def test_oldest_gpu_wait_does_not_hide_cpu_in_single_ranking_pass(ready_work):
    from api.execution.models import ExecutionSpec
    from django.db import connection
    from django.test.utils import CaptureQueriesContext

    host, stages = ready_work
    gpu_profile = ResourceProfile.objects.create(
        id="gpu-wait",
        fingerprint="g" * 64,
        image=stages[0].profile.image,
        stage_kind="embedding",
        validated=True,
        resources={"cpu": 1, "ram": GIB, "disk": GIB, "vram": 2 * GIB},
    )
    owner = ExecutionSpec.objects.create(
        specification={}, checksum="c" * 64, input_checksum="d" * 64, paused=False, priority=100
    )
    gpu = WorkStage.objects.create(
        execution=owner, key="oldest-gpu", kind="embedding", profile=gpu_profile
    )
    host.capabilities["profiles"].append(gpu_profile.fingerprint)
    host.save(update_fields=["capabilities"])
    with (
        override_settings(EXECUTION_V2_ENABLED=True),
        CaptureQueriesContext(connection) as captured,
    ):
        admitted = control.claim(host.pk)
    assert admitted["stage_id"] == str(stages[0].pk)
    gpu.refresh_from_db()
    assert gpu.state == "ready" and gpu.waiting_reason == "telemetry_unavailable"
    rankings = [
        q["sql"] for q in captured if "DISTINCT ON" in q["sql"] and '"allocation_gpu"' in q["sql"]
    ]
    assert len(rankings) == 1


def test_extra_gpu_headroom_reserves_capacity_for_controlled_external_probes():
    candidate = {"cpu": 2, "ram": GIB, "disk": GIB, "vram": 6 * GIB}
    budgets = {"cpu": 16, "ram": 96 * GIB, "disk": 128 * GIB, "gpu_slots": 5}
    telemetry = {
        "ram_available": 120 * GIB,
        "disk_free": 200 * GIB,
        "gpu_uuid": "gpu",
        "vram_total": 20 * GIB,
        "vram_free": 14 * GIB,
    }
    assert admit(candidate, [], budgets, telemetry, gpu=True).fits
    assert not admit(candidate, [], {**budgets, "vram_headroom": 9 * GIB}, telemetry, gpu=True).fits
    assert not admit(
        candidate, [], {**budgets, "vram_headroom": float("nan")}, telemetry, gpu=True
    ).fits
