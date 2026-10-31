import io
from types import SimpleNamespace

import pytest
from api.execution import control
from api.execution.artifact_policy import retention_for
from api.execution.models import (
    Artifact,
    ArtifactDependency,
    ArtifactLocation,
    ExecutorHost,
    WorkStage,
)
from api.execution.retention import claim_cleanup, finish_cleanup
from api.models import Job
from django.core.management import call_command
from django.utils import timezone
from tools.executor.agent import Agent


@pytest.mark.parametrize(
    "family,policy",
    [
        ("prot_t5_last/residue_vecs", "transient"),
        ("esm2_layer_26/residue_vecs", "transient"),
        ("esmc_layer_32/residue_vecs", "transient"),
        ("esm1v", "transient"),
        ("omniesi_esm2", "transient"),
        ("iecata_prot_t5_residues", "transient"),
        ("prot_t5_last/mean_vecs", "permanent"),
        ("esmc_layer_32/weighted_vecs", "permanent"),
        ("catpred_esm2/kcat/checkpoint", "permanent"),
        ("catrange_esmc", "permanent"),
        ("molecular/CataPro", "permanent"),
        ("unknown", "protected"),
    ],
)
def test_policy_requires_concurrency_provenance_and_preserves_vectors(family, policy):
    identity = {"path": "sequence_info/" + family + "/seq.npy"}
    assert retention_for(identity) == "protected"
    identity["representation_version"] = "execution-v2"
    assert retention_for(identity) == policy


def context():
    host = ExecutorHost.objects.create(
        id="gpu",
        reconciled=True,
        telemetry={"disk_free": 100000},
        budgets={"cache_free_target": 0},
        heartbeat_at=timezone.now(),
    )
    job = Job.objects.create(public_id="residue1", prediction_type="kcat", status="Processing")
    spec = control.ingest(job.public_id, {}, "a" * 64, paused=False)
    producer = WorkStage.objects.create(
        execution=spec, key="residue", kind="embedding", state="completed"
    )
    return host, spec, producer


def blob(host, producer, ident, policy="transient", family="prot_t5_last/residue_vecs"):
    artifact = Artifact.objects.create(
        id=ident * 64,
        identity={
            "representation_version": "execution-v2",
            "path": "sequence_info/" + family + "/seq.npy",
        },
        checksum=ident * 64,
        size=100,
        producer=producer,
        retention=policy,
    )
    ArtifactLocation.objects.create(artifact=artifact, host=host, ready=True)
    return artifact


@pytest.mark.django_db(transaction=True)
def test_residue_cleanup_runs_without_disk_pressure_after_last_shared_consumer():
    from datetime import timedelta

    from api.execution.models import ExecutionAttempt

    host, spec, producer = context()
    residue = blob(host, producer, "a")
    first = WorkStage.objects.create(execution=spec, key="first", kind="predict", state="completed")
    second = WorkStage.objects.create(execution=spec, key="second", kind="predict", state="ready")
    for stage in [first, second]:
        ArtifactDependency.objects.create(artifact=residue, stage=stage, path="residue.npy")
    active = ExecutionAttempt.objects.create(
        stage=second,
        host=host,
        resources={"cpu": 1},
        profile_fingerprint="f" * 64,
        lease_until=timezone.now() + timedelta(seconds=30),
        state="running",
    )
    assert claim_cleanup(host.pk) is None
    second.state = "completed"
    second.save(update_fields=["state"])
    assert claim_cleanup(host.pk) is None  # A live/uncertain reader still protects it.
    active.state = "completed"
    active.save(update_fields=["state"])
    unrelated = WorkStage.objects.create(
        execution=spec, key="unrelated", kind="predict", state="running"
    )
    ExecutionAttempt.objects.create(
        stage=unrelated,
        host=host,
        resources={"cpu": 1},
        profile_fingerprint="f" * 64,
        lease_until=timezone.now() + timedelta(seconds=30),
        state="running",
    )
    cleanup = claim_cleanup(host.pk)
    assert cleanup["blobs"] == [{"checksum": "a" * 64, "size": 100}]
    assert not residue.locations.get().ready
    assert claim_cleanup(host.pk) == cleanup
    finish_cleanup(host.pk, cleanup)
    assert claim_cleanup(host.pk) is None


@pytest.mark.django_db(transaction=True)
def test_old_project_residues_vectors_and_unknown_aliases_survive_pressure():
    host, _, producer = context()
    host.budgets = {"cache_free_target": 1000000, "cache_min_age_seconds": 0}
    host.save(update_fields=["budgets"])
    old = blob(host, producer, "a", "protected")
    vector = blob(host, producer, "b", "permanent", "prot_t5_last/mean_vecs")
    transient = blob(host, producer, "c")
    Artifact.objects.create(
        id="d" * 64, identity={"project": "legacy"}, checksum=transient.checksum, size=100
    )
    assert claim_cleanup(host.pk) is None
    assert all(a.locations.get().ready for a in [old, vector, transient])


@pytest.mark.django_db(transaction=True)
def test_backfill_uses_verified_platform_provenance_and_retains_standalone_requests():
    host, spec, producer = context()
    residue = blob(host, producer, "a", "protected")
    pooled = blob(host, producer, "b", "protected", "prot_t5_last/weighted_vecs")
    legacy = Artifact.objects.create(
        id="c" * 64,
        identity={"path": "sequence_info/prot_t5_last/residue_vecs/old.npy"},
        checksum="c" * 64,
        size=100,
    )
    ArtifactLocation.objects.create(artifact=legacy, host=host, ready=True)
    # A separate standalone executor request belongs to no platform prediction.
    from api.execution.models import ExecutionSpec

    standalone = ExecutionSpec.objects.create(
        specification={}, checksum="d" * 64, input_checksum="e" * 64
    )
    external = WorkStage.objects.create(
        execution=standalone, key="external", kind="embedding", state="completed"
    )
    external_blob = blob(host, external, "d", "protected")
    call_command("execution_artifact_retention", "--apply", stdout=io.StringIO())
    for a, expected in [
        (residue, "transient"),
        (pooled, "permanent"),
        (legacy, "protected"),
        (external_blob, "protected"),
    ]:
        a.refresh_from_db()
        assert a.retention == expected
    assert spec.job_id is not None


def test_executor_collects_only_approved_managed_blobs_while_unrelated_attempt_runs(tmp_path):
    agent = Agent({"root": str(tmp_path / "executor"), "host_id": "test"})
    agent.jobs = {"active": SimpleNamespace()}
    path = agent.store.path("a" * 64)
    path.parent.mkdir(parents=True)
    path.write_bytes(b"transient")
    project = tmp_path / "old-project-residue.npy"
    project.write_bytes(b"preserve")
    calls = []

    def request(operation, data=None):
        calls.append(operation)
        return (
            {"cleanup": {"id": "test", "token": "test", "blobs": [{"checksum": "a" * 64}]}}
            if operation == "collect-artifacts"
            else {}
        )

    agent.request = request
    agent.collect_artifacts()
    assert not path.exists()
    assert project.read_bytes() == b"preserve"
    assert calls == ["collect-artifacts", "collected-artifacts"]


@pytest.mark.django_db(transaction=True)
def test_private_scratch_is_released_only_after_completed_stage_and_ack():
    from api.execution.models import ExecutionAttempt

    host, spec, completed = context()
    host.capabilities = {"artifact_cleanup_version": 2}
    host.save(update_fields=["capabilities"])
    old = ExecutionAttempt.objects.create(
        stage=completed,
        host=host,
        state="stopped",
        resources={},
        profile_fingerprint="f" * 64,
        lease_until=timezone.now(),
        finished_at=timezone.now(),
    )
    held = WorkStage.objects.create(
        execution=spec, key="held-checkpoint", kind="embedding", state="held"
    )
    checkpoint = ExecutionAttempt.objects.create(
        stage=held,
        host=host,
        state="stopped",
        resources={},
        profile_fingerprint="f" * 64,
        lease_until=timezone.now(),
        finished_at=timezone.now(),
    )
    cleanup = claim_cleanup(host.pk, version=2)
    assert cleanup["attempt_ids"] == [str(old.pk)]
    assert claim_cleanup(host.pk) is None  # Protocol 1 cannot acknowledge scratch.
    old.refresh_from_db()
    assert old.scratch_released_at is None
    finish_cleanup(host.pk, cleanup)
    old.refresh_from_db()
    checkpoint.refresh_from_db()
    assert old.scratch_released_at is not None
    assert checkpoint.scratch_released_at is None
    assert claim_cleanup(host.pk, version=2) is None


def test_completed_scratch_cleanup_keeps_journals_and_external_project_files(tmp_path):
    agent = Agent({"root": str(tmp_path / "executor"), "host_id": "test"})
    work = agent.root / "attempts" / "old"
    (work / "tmp").mkdir(parents=True)
    (work / "tmp" / "residue.npy").write_bytes(b"scratch")
    (work / "request.json").write_text("{}")
    project = tmp_path / "project"
    project.mkdir()
    (project / "old-residue.npy").write_bytes(b"keep")
    (work / "media").symlink_to(project, target_is_directory=True)

    def request(operation, data=None):
        if operation == "collect-artifacts":
            return {"cleanup": {"id": "test", "token": "test", "blobs": [], "attempt_ids": ["old"]}}
        return {}

    agent.request = request
    agent.collect_artifacts()
    assert (work / "request.json").exists()
    assert not (work / "tmp").exists()
    assert not (work / "media").is_symlink()
    assert (project / "old-residue.npy").read_bytes() == b"keep"


@pytest.mark.django_db(transaction=True)
def test_cleanup_advances_past_a_page_of_shared_residues_and_honors_release_hints():
    from api.execution.models import ArtifactSweep
    from api.execution.retention import release_consumed_residues
    from django.db import transaction

    host, spec, producer = context()
    consumer = WorkStage.objects.create(execution=spec, key="shared-pending", kind="predict")
    artifacts = [
        Artifact(
            id=f"{i:064x}",
            checksum=f"{i:064x}",
            size=100,
            identity={},
            retention="transient",
            producer=producer,
        )
        for i in range(1, 258)
    ]
    Artifact.objects.bulk_create(artifacts)
    ArtifactLocation.objects.bulk_create(
        [ArtifactLocation(artifact=a, host=host, ready=True) for a in artifacts]
    )
    ArtifactDependency.objects.bulk_create(
        [
            ArtifactDependency(artifact=a, stage=consumer, path=f"{i}.npy")
            for i, a in enumerate(artifacts[:256])
        ]
    )
    assert claim_cleanup(host.pk) is None
    assert ArtifactSweep.objects.get(host=host).cursor > 0
    assert claim_cleanup(host.pk) is None
    cleanup = claim_cleanup(host.pk)
    assert cleanup["blobs"] == [{"checksum": f"{257:064x}", "size": 100}]
    finish_cleanup(host.pk, cleanup)
    # A completion hint bypasses the shared-residue scan immediately.
    consumer.state = "completed"
    consumer.save(update_fields=["state"])
    with transaction.atomic():
        release_consumed_residues(consumer)
    assert ArtifactSweep.objects.get(host=host).pending
    assert claim_cleanup(host.pk)["blobs"]
