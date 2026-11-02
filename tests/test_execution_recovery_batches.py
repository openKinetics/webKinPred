from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from datetime import timedelta
from threading import Barrier

import pytest
from api.execution import control
from api.execution.models import (
    Artifact,
    ArtifactLocation,
    ExecutionAttempt,
    ExecutorHost,
    RecoveryBatch,
    WorkStage,
)
from api.execution.planning import attach
from api.execution.recovery import missing_dependencies, reconcile_pending
from api.models import Job
from django.core.management import call_command
from django.db import close_old_connections
from django.utils import timezone

from test_execution_pipeline import specification

pytestmark = pytest.mark.django_db(transaction=True)


def fixture(count=16, method="EITLEM"):
    old_job = Job.objects.create(public_id="recoverold", status="Completed", prediction_type="kcat")
    old = control.ingest(old_job.public_id, specification(method), "a" * 64)
    mapping = {f"seq{i:02}": "MA" + "A" * i for i in range(count)}
    source = WorkStage.objects.create(
        execution=old,
        key="original",
        kind="embedding",
        method=method,
        target="kcat",
        state="completed",
    )
    artifacts, outputs = [], []
    for sid in mapping:
        path = f"sequence_info/esm1v/{sid}.npy"
        identity = {"method": method, "path": path}
        artifact = Artifact.objects.create(
            id=control.digest(identity),
            identity=identity,
            checksum="b" * 64,
            size=12,
            producer=source,
            retention="transient",
        )
        artifacts.append(artifact)
        outputs.append({"id": artifact.pk, "identity": identity, "path": path})
    source.payload = {
        "step": "eitlem_esm1v",
        "seq_id_to_seq": mapping,
        "outputs": outputs,
        "output_artifacts": [a.pk for a in artifacts],
        "shape": {"sequences": count, "max_length": max(map(len, mapping.values()))},
    }
    source.save()
    return source, artifacts


def consumer(name, artifacts):
    j = Job.objects.create(
        public_id=control.digest(name)[:10], status="Processing", prediction_type="kcat"
    )
    e = control.ingest(j.public_id, specification("EITLEM"), "a" * 64)
    e.paused = False
    e.save(update_fields=["paused"])
    s = WorkStage.objects.create(
        execution=e,
        key="prediction",
        kind="predict",
        method="EITLEM",
        target="kcat",
        state="recovering",
    )
    for a in artifacts:
        attach(s, a, a.identity["path"])
    return s


def test_missing_sixteen_outputs_create_one_complete_batch_and_preserve_history():
    original, artifacts = fixture()
    before = deepcopy(original.payload)
    head = consumer("recovery-new", artifacts)
    missing_dependencies(head.pk, None)
    batch = RecoveryBatch.objects.get().stage
    assert len(batch.payload["outputs"]) == len(batch.payload["seq_id_to_seq"]) == 16
    assert batch.payload["shape"]["sequences"] == 16
    assert head.dependencies.filter(pk=batch.pk).exists()
    assert Artifact.objects.filter(pk__in=[a.pk for a in artifacts], producer=batch).count() == 16
    original.refresh_from_db()
    assert original.state == "completed" and original.payload == before
    assert original.execution.job.status == "Completed"


def test_partial_protein_batch_is_narrowed_and_other_consumer_coalesces():
    _, artifacts = fixture()
    first = consumer("recovery-first", artifacts[:3])
    second = consumer("recovery-second", artifacts[3:5])
    missing_dependencies(first.pk, None)
    batch = RecoveryBatch.objects.get().stage
    assert len(batch.payload["seq_id_to_seq"]) == 3
    missing_dependencies(second.pk, None)
    batch.refresh_from_db()
    assert len(batch.payload["seq_id_to_seq"]) == len(batch.payload["outputs"]) == 5
    assert batch.payload["shape"] == {"sequences": 5, "max_length": 6}
    assert (
        first.dependencies.filter(pk=batch.pk).exists()
        and second.dependencies.filter(pk=batch.pk).exists()
    )
    assert RecoveryBatch.objects.count() == 1


def test_claimed_batch_is_immutable_and_following_work_gets_another_generation():
    _, artifacts = fixture()
    first = consumer("running-first", artifacts[:3])
    second = consumer("running-second", artifacts[3:5])
    missing_dependencies(first.pk, None)
    batch = RecoveryBatch.objects.get().stage
    before = deepcopy(batch.payload)
    host = ExecutorHost.objects.create(id="gpu")
    attempt = ExecutionAttempt.objects.create(
        stage=batch, host=host, state="leased", lease_until=timezone.now() + timedelta(seconds=30)
    )
    missing_dependencies(second.pk, None)
    batch.refresh_from_db()
    latest = RecoveryBatch.objects.get()
    assert batch.payload == before and latest.stage_id != batch.pk and latest.generation == 2
    assert len(latest.stage.payload["seq_id_to_seq"]) == 2
    attempt.refresh_from_db()
    assert attempt.state == "leased" and not attempt.stop_requested


def test_verified_artifact_is_not_invalidated_by_unspecified_worker_error():
    _, artifacts = fixture(2)
    host = ExecutorHost.objects.create(id="gpu")
    ArtifactLocation.objects.create(artifact=artifacts[0], host=host, ready=True)
    head = consumer("one-missing", artifacts)
    missing_dependencies(head.pk, host.pk)
    batch = RecoveryBatch.objects.get().stage
    assert [o["id"] for o in batch.payload["outputs"]] == [artifacts[1].pk]
    assert ArtifactLocation.objects.get(artifact=artifacts[0], host=host).ready
    artifacts[0].refresh_from_db()
    assert artifacts[0].checksum == "b" * 64


def test_molecular_recovery_preserves_all_original_positions_and_rows():
    original, artifacts = fixture(4, "UniKP")
    original.method = "molecular-UniKP"
    rows = [
        {"input": "CC", "legacy_position": p, "path": a.identity["path"], "unit_index": i}
        for i, (a, p) in enumerate(zip(artifacts, [0, 99, 100, 128], strict=True))
    ]
    original.payload = {
        "step": "molecular",
        "molecular_method": "UniKP",
        "rows": rows,
        "outputs": original.payload["outputs"],
        "output_artifacts": [a.pk for a in artifacts],
        "shape": {"sequences": 4, "max_length": 2},
    }
    original.save()
    head = consumer("molecular-partial", [artifacts[-1]])
    missing_dependencies(head.pk, None)
    batch = RecoveryBatch.objects.get().stage
    assert batch.payload["rows"] == rows and batch.payload["shape"]["sequences"] == 4
    assert len(batch.payload["outputs"]) == 1


def legacy(artifacts, original, execution):
    stages = []
    for a in artifacts:
        p = deepcopy(original.payload)
        p.update(
            outputs=[next(o for o in p["outputs"] if o["id"] == a.pk)], output_artifacts=[a.pk]
        )
        s = WorkStage.objects.create(
            execution=execution,
            key="restore:" + a.pk,
            method=original.method,
            target=original.target,
            kind=original.kind,
            payload=p,
        )
        a.producer = s
        a.save(update_fields=["producer"])
        stages.append(s)
    return stages


def test_reconcile_legacy_backlog_keeps_active_attempt_and_all_consumers():
    original, artifacts = fixture()
    head = consumer("legacy-owner", artifacts)
    stages = legacy(artifacts, original, head.execution)
    second = consumer("legacy-reader", artifacts)
    head.dependencies.add(*stages)
    host = ExecutorHost.objects.create(id="gpu")
    active = ExecutionAttempt.objects.create(
        stage=stages[0],
        host=host,
        state="leased",
        lease_until=timezone.now() + timedelta(seconds=30),
    )
    before = deepcopy(stages[0].payload)
    result = reconcile_pending()
    assert result["retired"] == 15 and result["batch_stages"] == 1
    batch = RecoveryBatch.objects.get().stage
    assert len(batch.payload["seq_id_to_seq"]) == len(batch.payload["outputs"]) == 15
    assert (
        head.dependencies.filter(pk=batch.pk).exists()
        and second.dependencies.filter(pk=batch.pk).exists()
    )
    assert head.dependencies.filter(pk=stages[0].pk).exists()
    assert not head.dependencies.filter(state="canceled").exists()
    stages[0].refresh_from_db()
    active.refresh_from_db()
    assert stages[0].payload == before and active.state == "leased" and not active.stop_requested
    assert reconcile_pending()["retired"] == 0


def test_dry_run_rolls_back_and_reuses_already_verified_output(capsys):
    original, artifacts = fixture(2)
    head = consumer("dry-run-owner", artifacts)
    stages = legacy(artifacts, original, head.execution)
    head.dependencies.add(*stages)
    call_command("execution_reconcile_recovery", dry_run=True)
    assert RecoveryBatch.objects.count() == 0
    assert all(s.state == "ready" for s in WorkStage.objects.filter(pk__in=[s.pk for s in stages]))
    host = ExecutorHost.objects.create(id="gpu")
    ArtifactLocation.objects.create(artifact=artifacts[0], host=host, ready=True)
    result = reconcile_pending()
    assert result["reused"] == 1 and result["retired"] == 1
    stages[0].refresh_from_db()
    assert stages[0].state == "completed" and stages[0].checkpoint["reused_verified_outputs"]


def test_concurrent_consumers_get_one_owner_for_shared_artifacts():
    _, artifacts = fixture(8)
    first = consumer("race-first", artifacts)
    second = consumer("race-second", artifacts)
    barrier = Barrier(2)

    def recover(pk):
        close_old_connections()
        try:
            barrier.wait(timeout=5)
            missing_dependencies(pk, None)
        finally:
            close_old_connections()

    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = [pool.submit(recover, s.pk) for s in [first, second]]
        for future in futures:
            future.result(timeout=15)
    batch = RecoveryBatch.objects.get().stage
    assert len(batch.payload["outputs"]) == 8
    assert Artifact.objects.filter(pk__in=[a.pk for a in artifacts], producer=batch).count() == 8
    assert all(s.dependencies.filter(pk=batch.pk).exists() for s in [first, second])


def test_held_recovery_stays_held_and_later_loss_uses_new_generation():
    _, artifacts = fixture(4)
    first = consumer("held-first", artifacts)
    missing_dependencies(first.pk, None)
    batch = RecoveryBatch.objects.get().stage
    batch.state = "held"
    batch.save(update_fields=["state"])
    second = consumer("held-second", [artifacts[0]])
    missing_dependencies(second.pk, None)
    batch.refresh_from_db()
    assert batch.state == "held" and RecoveryBatch.objects.get().generation == 1
    batch.state = "completed"
    batch.save(update_fields=["state"])
    first.execution.job.status = "Completed"
    first.execution.job.save(update_fields=["status"])
    third = consumer("later-loss", [artifacts[0]])
    missing_dependencies(third.pk, None)
    new = RecoveryBatch.objects.get()
    assert new.generation == 2 and new.stage.execution_id == third.execution_id
    assert len(new.stage.payload["seq_id_to_seq"]) == 1
    batch.refresh_from_db()
    assert batch.state == "completed"


def test_stopped_attempt_keeps_its_retry_payload_and_profile():
    _, artifacts = fixture(4)
    first = consumer("retry-first", artifacts[:2])
    second = consumer("retry-next", artifacts[2:])
    missing_dependencies(first.pk, None)
    batch = RecoveryBatch.objects.get().stage
    before = deepcopy(batch.payload)
    host = ExecutorHost.objects.create(id="gpu")
    ExecutionAttempt.objects.create(
        stage=batch,
        host=host,
        state="stopped",
        lease_until=timezone.now(),
        finished_at=timezone.now(),
        outcome={"category": "oom"},
    )
    missing_dependencies(second.pk, None)
    batch.refresh_from_db()
    assert batch.payload == before
    assert RecoveryBatch.objects.get().generation == 2
