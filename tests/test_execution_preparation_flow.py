"""Large preparations release finished chunks without waiting for the whole request."""

import threading

import pytest
from api.execution import control, planning
from api.execution.artifacts import sha256_file
from api.execution.models import ArtifactCleanup, ExecutorHost
from api.models import Job
from django.db import close_old_connections
from django.test import override_settings
from django.utils import timezone

pytestmark = pytest.mark.django_db(transaction=True, databases=["default", "seqmap"])

SPEC = {
    "targets": ["kcat"],
    "methods": {"kcat": "DLKcat"},
    "handle_long_sequences": "truncate",
    "canonicalize_substrates": True,
    "include_similarity_columns": False,
    "model_versions": {"DLKcat": {"weights": "test"}},
}


def request(tmp_path, rows):
    media = tmp_path / "media"
    data = media / "jobs" / "large" / "input.csv"
    data.parent.mkdir(parents=True)
    data.write_text("Protein Sequence,Substrate\n" + "MAAA,CC\n" * rows)
    Job.objects.create(public_id="large", prediction_type="kcat", status="Pending")
    execution = control.ingest("large", SPEC, sha256_file(data), paused=False)
    return execution, media, tmp_path / "blobs"


def test_each_committed_chunk_is_bound_before_the_next_is_planned(tmp_path, monkeypatch):
    ExecutorHost.objects.create(id="production")
    execution, media, blobs = request(tmp_path, 129)
    seen = []
    monkeypatch.setattr(
        planning,
        "bind_profiles",
        lambda: seen.append(execution.stages.filter(kind="predict").count()),
    )
    with override_settings(EXECUTION_V2_ENABLED=True):
        planning.prepare(execution, media, blobs)
    assert seen == [1, 2]
    assert execution.stages.get(key="prepare").state == "completed"


def test_chunks_leave_the_graph_lock_to_a_starved_collector(tmp_path, monkeypatch):
    from api.execution.models import ArtifactSweep

    host = ExecutorHost.objects.create(id="production")
    execution, media, blobs = request(tmp_path, 129)
    ArtifactSweep.objects.create(host=host, pending=[1, 2, 3])
    monkeypatch.setattr(planning, "bind_profiles", lambda: None)
    waits = []

    def collector_turn(seconds):
        # The collector takes the free lock and creates a cleanup meanwhile.
        waits.append(seconds)
        ArtifactCleanup.objects.create(host=host, blobs=[], completed_at=timezone.now())

    monkeypatch.setattr(planning.time, "sleep", collector_turn)
    with override_settings(EXECUTION_V2_ENABLED=True):
        planning.prepare(execution, media, blobs)
    # The first chunk yields once; the second finds a recent cleanup.
    assert waits == [1]
    assert execution.stages.get(key="prepare").state == "completed"


def test_chunks_do_not_wait_without_pending_deletions_or_past_the_bound(monkeypatch):
    from api.execution.models import ArtifactSweep

    clock = [0.0]
    monkeypatch.setattr(planning.time, "monotonic", lambda: clock[0])
    waits = []

    def sleep(seconds):
        waits.append(seconds)
        clock[0] += seconds

    monkeypatch.setattr(planning.time, "sleep", sleep)
    host = ExecutorHost.objects.create(id="production")
    sweep = ArtifactSweep.objects.create(host=host, pending=[])
    planning._yield_to_cleanup()
    assert waits == []
    sweep.pending = [1]
    sweep.save(update_fields=["pending"])
    planning._yield_to_cleanup()  # The collector never gets a turn.
    assert len(waits) == planning.CLEANUP_YIELD_SECONDS


def test_chunk_publication_waits_for_a_pending_cleanup_instead_of_failing(tmp_path, monkeypatch):
    host = ExecutorHost.objects.create(id="production")
    execution, media, blobs = request(tmp_path, 3)
    cleanup = ArtifactCleanup.objects.create(host=host, blobs=[])
    monkeypatch.setattr(planning, "bind_profiles", lambda: None)

    def finish_cleanup():
        close_old_connections()
        try:
            ArtifactCleanup.objects.filter(pk=cleanup.pk).update(completed_at=timezone.now())
        finally:
            close_old_connections()

    timer = threading.Timer(1.5, finish_cleanup)
    timer.start()
    try:
        with override_settings(EXECUTION_V2_ENABLED=True):
            planning.prepare(execution, media, blobs)
    finally:
        timer.cancel()
    assert execution.stages.get(key="prepare").state == "completed"


def test_preparation_candidates_are_ordered_once_without_sorting_specifications():
    from api.execution.preparation import preparation_candidates
    from django.db import connection
    from django.db.models import Q
    from django.test.utils import CaptureQueriesContext

    executions = []
    for public_id in ("first", "second"):
        Job.objects.create(public_id=public_id, prediction_type="kcat", status="Pending")
        executions.append(control.ingest(public_id, SPEC, "a" * 64, paused=False))
    with CaptureQueriesContext(connection) as queries:
        found = list(preparation_candidates(Q(), timezone.now()))
    assert [e.pk for e in found] == [e.pk for e in executions]
    assert "DISTINCT" not in queries[0]["sql"]
