from datetime import timedelta

import pytest
from api.execution import control
from api.execution.models import (
    Artifact,
    ArtifactDependency,
    ArtifactLocation,
    ExecutorHost,
    WorkStage,
)
from api.execution.retention import claim_cleanup, finish_cleanup
from api.models import Job
from django.db import transaction
from django.utils import timezone


@pytest.mark.django_db(transaction=True)
def test_cleanup_protects_unfinished_aliases_inputs_results_and_live_reservations():
    host = ExecutorHost.objects.create(
        id="gpu",
        reconciled=True,
        budgets={"cache_free_target": 10000, "cache_min_age_seconds": 0},
        telemetry={"disk_free": 1000},
    )
    job = Job.objects.create(public_id="gc-test", status="Pending", prediction_type="kcat")
    execution = control.ingest(job.public_id, {"test": True}, "a" * 64)
    done = WorkStage.objects.create(
        execution=execution, key="done", kind="predict", state="completed"
    )
    pending = WorkStage.objects.create(
        execution=execution, key="pending", kind="predict", state="held"
    )

    def blob(ident, checksum, regenerable=True):
        artifact = Artifact.objects.create(
            id=ident * 64,
            checksum=checksum * 64,
            size=100,
            identity={"id": ident},
            regenerable=regenerable,
            retention="cache",
        )
        ArtifactLocation.objects.create(
            artifact=artifact,
            host=host,
            ready=True,
            verified_at=timezone.now() - timedelta(hours=2),
        )
        return artifact

    original = blob("1", "a", False)
    aliased_original = blob("2", "a")
    shared = blob("3", "b")
    alias = blob("4", "b")
    ArtifactDependency.objects.create(artifact=shared, stage=done, path="done.npy")
    ArtifactDependency.objects.create(artifact=alias, stage=pending, path="pending.npy")
    # Mixed finished/unfinished references must retain the same bytes.
    ArtifactDependency.objects.create(artifact=alias, stage=done, path="alias.npy")
    eligible = blob("5", "c")
    ArtifactDependency.objects.create(artifact=eligible, stage=done, path="eligible.npy")
    cleanup = claim_cleanup(host.pk)
    assert cleanup["blobs"] == [{"checksum": "c" * 64, "size": 100}]
    assert not eligible.locations.get().ready
    assert all(
        artifact.locations.get().ready for artifact in [original, aliased_original, shared, alias]
    )
    assert claim_cleanup(host.pk) == cleanup  # Lost response resumes the exact same deletion.
    with pytest.raises(PermissionError):
        finish_cleanup(host.pk, {**cleanup, "token": "stale"})
    assert finish_cleanup(host.pk, cleanup) == {"completed": True}
    assert finish_cleanup(host.pk, cleanup) == {"completed": True}
    assert claim_cleanup(host.pk) is None


@pytest.mark.django_db(transaction=True)
def test_pending_cleanup_fences_new_claims_and_preparation(tmp_path):
    from api.execution.artifacts import ArtifactStore
    from api.execution.models import ArtifactCleanup
    from api.execution.planning import publish_file
    from django.test import override_settings

    host = ExecutorHost.objects.create(
        id="production", paused=False, reconciled=True, heartbeat_at=timezone.now()
    )
    ArtifactCleanup.objects.create(host=host, blobs=[])
    with override_settings(EXECUTION_V2_ENABLED=True):
        assert control.claim(host.pk) is None
    path = tmp_path / "input.csv"
    path.write_text("Protein Sequence,Substrate\nMAAA,CC\n")
    with pytest.raises(BlockingIOError, match="cleanup"):
        publish_file(path, {"input": "test"}, host.pk, ArtifactStore(tmp_path / "blobs"))
    assert not (tmp_path / "blobs").exists()


def residue(host, producer, ident, checksum):
    artifact = Artifact.objects.create(
        id=ident * 64,
        identity={"id": ident},
        checksum=checksum * 64,
        size=100,
        producer=producer,
        retention="transient",
    )
    return artifact, ArtifactLocation.objects.create(artifact=artifact, host=host, ready=True)


def residue_context():
    host = ExecutorHost.objects.create(
        id="gpu", reconciled=True, budgets={"cache_free_target": 0}, telemetry={"disk_free": 10}
    )
    job = Job.objects.create(public_id="sweep-test", status="Processing", prediction_type="kcat")
    execution = control.ingest(job.public_id, {}, "a" * 64, paused=False)
    producer = WorkStage.objects.create(
        execution=execution, key="embed", kind="embedding", state="completed"
    )
    return host, execution, producer


@pytest.mark.django_db(transaction=True)
def test_last_consumer_hints_copies_registered_under_aliases():
    from api.execution.models import ArtifactSweep
    from api.execution.retention import release_consumed_residues

    host, execution, producer = residue_context()
    copied, copied_location = residue(host, producer, "1", "a")
    alias, alias_location = residue(host, producer, "2", "a")
    early = WorkStage.objects.create(execution=execution, key="early", kind="predict")
    late = WorkStage.objects.create(execution=execution, key="late", kind="predict")
    ArtifactDependency.objects.create(artifact=copied, stage=early, path="r.npy")
    ArtifactDependency.objects.create(artifact=alias, stage=late, path="r.npy")
    # The last reader of the same bytes depends on the alias, not on the copy.
    with transaction.atomic():
        release_consumed_residues(late)
    assert set(ArtifactSweep.objects.get(host=host).pending) == {
        copied_location.pk,
        alias_location.pk,
    }


@pytest.mark.django_db(transaction=True)
def test_sweep_skips_pages_without_candidates_within_one_call(monkeypatch):
    from api.execution import retention
    from api.execution.models import ArtifactSweep

    monkeypatch.setattr(retention, "SWEEP_PAGE", 2)
    host, execution, producer = residue_context()
    for ident in "123456":
        artifact = Artifact.objects.create(
            id=ident * 64,
            identity={"id": ident},
            checksum=ident * 64,
            size=1,
            producer=producer,
            retention="permanent",
        )
        ArtifactLocation.objects.create(artifact=artifact, host=host, ready=True)
    _, location = residue(host, producer, "7", "7")
    done = WorkStage.objects.create(execution=execution, key="d", kind="predict", state="completed")
    ArtifactDependency.objects.create(artifact=location.artifact, stage=done, path="r.npy")
    cleanup = claim_cleanup(host.pk)
    assert cleanup["blobs"] == [{"checksum": "7" * 64, "size": 100}]
    assert ArtifactSweep.objects.get(host=host).cursor == location.pk
    finish_cleanup(host.pk, cleanup)
    assert claim_cleanup(host.pk) is None
    assert ArtifactSweep.objects.get(host=host).cursor == 0  # The pass wrapped and ended.


@pytest.mark.django_db(transaction=True)
def test_large_backlogs_are_collected_in_bounded_batches(monkeypatch):
    from api.execution import retention
    from api.execution.models import ArtifactSweep

    monkeypatch.setattr(retention, "BLOBS_PER_CLEANUP", 2)
    host, execution, producer = residue_context()
    done = WorkStage.objects.create(execution=execution, key="d", kind="predict", state="completed")
    for ident in "12345":
        artifact, _ = residue(host, producer, ident, ident)
        ArtifactDependency.objects.create(artifact=artifact, stage=done, path=ident)
    collected = []
    while cleanup := claim_cleanup(host.pk):
        assert len(cleanup["blobs"]) <= 2
        collected += [blob["checksum"][0] for blob in cleanup["blobs"]]
        finish_cleanup(host.pk, cleanup)
    assert sorted(collected) == list("12345")
    assert ArtifactSweep.objects.get(host=host).pending == []
