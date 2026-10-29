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
