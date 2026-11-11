"""Pruning removes only bookkeeping that finished requests no longer use."""

import io
import json
from datetime import timedelta

import pytest
from api.execution import control
from api.execution.models import (
    Artifact,
    ArtifactDependency,
    ArtifactLocation,
    ExecutionOutbox,
    ExecutorHost,
    WorkStage,
)
from api.models import Job
from django.core.management import call_command
from django.utils import timezone

pytestmark = pytest.mark.django_db(transaction=True)


def request(public_id, status, finished_hours_ago=None):
    job = Job.objects.create(public_id=public_id, prediction_type="kcat", status=status)
    if finished_hours_ago is not None:
        job.completion_time = timezone.now() - timedelta(hours=finished_hours_ago)
        job.save(update_fields=["completion_time"])
    execution = control.ingest(public_id, {}, "a" * 64, paused=False)
    execution.stages.all().delete()
    return execution


def stage(execution, key, kind, state="completed", method=""):
    return WorkStage.objects.create(
        execution=execution, key=key, kind=kind, state=state, method=method
    )


def artifact(ident, retention, producer, host=None, ready=False):
    row = Artifact.objects.create(
        id=ident * 64,
        identity={"id": ident},
        checksum=ident * 64,
        size=1,
        retention=retention,
        producer=producer,
    )
    if host is not None:
        ArtifactLocation.objects.create(artifact=row, host=host, ready=ready)
    return row


def link(consumer, row, path):
    ArtifactDependency.objects.create(stage=consumer, artifact=row, path=path)


def prune(*arguments):
    output = io.StringIO()
    call_command(
        "execution_prune_history",
        *arguments,
        "--pause",
        "0",
        "--max-io-pressure",
        "100",
        stdout=output,
    )
    return json.loads(output.getvalue())


def scenario():
    host = ExecutorHost.objects.create(id="gpu")
    old = request("oldjob", "Completed", finished_hours_ago=48)
    embedding = stage(old, "embed", "embedding")
    prediction = stage(old, "predict", "predict")
    pool = stage(old, "pool", "predict", method="KinForm-pool")
    rows = {
        "input": artifact("1", "permanent", None, host, ready=True),
        "residue": artifact("2", "transient", embedding, host, ready=False),
        "vector": artifact("3", "permanent", embedding, host, ready=True),
        "pooled_input": artifact("4", "transient", embedding, host, ready=False),
        "on_disk": artifact("5", "transient", embedding, host, ready=True),
        "live_input": artifact("6", "transient", embedding, host, ready=False),
        "recent_input": artifact("7", "transient", embedding, host, ready=False),
    }
    link(embedding, rows["input"], "input.csv")
    link(prediction, rows["residue"], "r.npy")
    link(prediction, rows["vector"], "v.npy")
    link(pool, rows["pooled_input"], "p.npy")
    running = request("livejob", "Processing")
    link(stage(running, "predict", "predict", state="ready"), rows["live_input"], "l.npy")
    recent = request("newjob", "Completed", finished_hours_ago=1)
    link(stage(recent, "predict", "predict"), rows["recent_input"], "n.npy")
    return rows, embedding, prediction, pool


def test_dry_run_reports_without_deleting():
    scenario()
    before = (ArtifactDependency.objects.count(), Artifact.objects.count())
    summary = prune()
    assert summary["input_links"] == 2 and summary["apply"] is False
    assert (ArtifactDependency.objects.count(), Artifact.objects.count()) == before


def test_only_finished_requests_unused_bookkeeping_is_deleted():
    rows, embedding, prediction, pool = scenario()
    summary = prune("--apply")
    assert summary["input_links"] == 2
    assert not prediction.artifacts.exists()
    # Recovery reads the inputs of embedding and KinForm pool/binding producers.
    assert embedding.artifacts.count() == 1 and pool.artifacts.count() == 1
    # Only the residue matrix that lost its last link and has no copy goes.
    assert summary == {**summary, "residue_artifacts": 1, "residue_locations": 1}
    assert not Artifact.objects.filter(pk=rows["residue"].pk).exists()
    for kept in ("input", "vector", "pooled_input", "on_disk", "live_input", "recent_input"):
        assert Artifact.objects.filter(pk=rows[kept].pk).exists(), kept
    assert ArtifactDependency.objects.filter(stage__execution__job__public_id="livejob").exists()
    assert ArtifactDependency.objects.filter(stage__execution__job__public_id="newjob").exists()


def test_requests_with_pending_progress_projection_are_kept():
    _, _, prediction, _ = scenario()
    ExecutionOutbox.objects.create(
        key="progress:old", event={"progress_execution_ids": [str(prediction.execution_id)]}
    )
    assert prune("--apply")["input_links"] == 0
    assert prediction.artifacts.count() == 2


def test_paging_keeps_aliases_of_one_checksum_together():
    host = ExecutorHost.objects.create(id="gpu")
    old = request("aliasjob", "Completed", finished_hours_ago=48)
    producer = stage(old, "embed", "embedding")
    for ident in "89":
        Artifact.objects.create(
            id=ident * 64,
            identity={"id": ident},
            checksum="c" * 64,
            size=1,
            retention="transient",
            producer=producer,
        )
        ArtifactLocation.objects.create(artifact_id=ident * 64, host=host, ready=False)
    assert prune("--apply", "--batch", "1")["residue_artifacts"] == 2
    assert not Artifact.objects.filter(checksum="c" * 64).exists()


def test_pruning_waits_while_host_io_pressure_would_refuse_launches(monkeypatch):
    from api.management.commands import execution_prune_history as module

    readings = iter(
        ["full avg10=15.00 avg60=1 avg300=1 total=1", "full avg10=1.00 avg60=1 avg300=1 total=1"]
    )
    waits = []

    class Pressure:
        def __init__(self, path):
            assert str(path) == "/proc/pressure/io"

        def read_text(self):
            return "some avg10=1 avg60=1 avg300=1 total=1\n" + next(readings)

    monkeypatch.setattr(module, "Path", Pressure)
    monkeypatch.setattr(module.time, "sleep", waits.append)
    command = module.Command()
    command.pause, command.max_io_pressure, command.deadline = 0.2, 3.0, None
    command.yield_to_load()
    assert waits == [0.2, 5]
