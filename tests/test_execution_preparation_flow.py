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
