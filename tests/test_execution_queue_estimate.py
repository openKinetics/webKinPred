"""User-facing queue hints remain honest under concurrent resource scheduling."""

from datetime import timedelta
from unittest.mock import patch

import pytest
from api.execution import status
from api.execution.models import ExecutionSpec, WorkStage
from api.models import Job
from django.db import connection
from django.test import RequestFactory, override_settings
from django.test.utils import CaptureQueriesContext
from django.utils import timezone

pytestmark = pytest.mark.django_db(transaction=True)


@pytest.fixture(autouse=True)
def fresh_snapshot(monkeypatch):
    monkeypatch.setattr(status, "_queue_snapshot", None)


def queued(name, submitted, *, paused=False, public_status="Pending", started=False):
    job = Job.objects.create(
        public_id=name,
        prediction_type="kcat",
        status=public_status,
        submission_time=submitted,
        start_time=submitted if started else None,
    )
    execution = ExecutionSpec.objects.create(
        job=job, specification={}, checksum="a" * 64, input_checksum="b" * 64, paused=paused
    )
    return job, execution


def test_estimate_counts_older_waiting_jobs_excludes_started_paused_and_held():
    now = timezone.now()
    queued("older", now - timedelta(days=2))
    queued("running", now - timedelta(days=3), public_status="Processing", started=True)
    queued("complete", now - timedelta(days=3), public_status="Completed", started=True)
    queued("paused", now - timedelta(days=3), paused=True)
    _, held = queued("held", now - timedelta(days=3))
    WorkStage.objects.create(execution=held, key="held", state="held")
    _, unprofiled = queued("profile", now - timedelta(days=3))
    WorkStage.objects.create(
        execution=unprofiled, key="predict", waiting_reason="profiling_required"
    )
    Job.objects.create(
        public_id="unowned",
        prediction_type="kcat",
        status="Pending",
        submission_time=now - timedelta(days=4),
    )
    job, _ = queued("current", now)
    queued("younger", now + timedelta(seconds=1))
    with override_settings(EXECUTION_V2_ENABLED=True):
        data = status.execution_status(job)
    assert data["queue_estimate"]["position"] == 2
    assert data["queue_estimate"]["jobs_ahead"] == 1
    assert data["queue_estimate"]["waiting_jobs"] == 3
    assert data["queue_estimate"]["approximate"] is True
    assert data["queue_estimate"]["basis"] == "submission_order"


def test_tied_submission_times_use_job_identity_not_import_order():
    now = timezone.now()
    first, _ = queued("first", now)
    second, _ = queued("second", now)
    assert status.queue_estimate(first, "waiting_gpu")["position"] == 1
    assert status.queue_estimate(second, "waiting_gpu")["position"] == 2


def test_cached_snapshot_is_bounded_and_counts_new_submissions_until_refresh():
    now = timezone.now()
    older, _ = queued("old", now - timedelta(seconds=1))
    current, _ = queued("current", now)
    assert status.queue_estimate(current, "waiting_cpu")["position"] == 2
    with CaptureQueriesContext(connection) as queries:
        assert status.queue_estimate(current, "waiting_cpu")["position"] == 2
    assert len(queries) == 0
    added, _ = queued("new", now + timedelta(seconds=1))
    new = status.queue_estimate(added, "waiting_cpu")
    assert new["position"] == 3 and new["waiting_jobs"] == 3
    Job.objects.filter(pk=older.pk).update(status="Processing", start_time=now)
    status._queue_snapshot["expires"] = 0
    refreshed = status.queue_estimate(current, "waiting_cpu")
    assert refreshed["position"] == 1 and refreshed["waiting_jobs"] == 2


@pytest.mark.parametrize(
    "public_status,state,started",
    [
        ("Pending", "paused", False),
        ("Pending", "needs_attention", False),
        ("Pending", "retrying", False),
        ("Pending", "waiting_cpu", True),
        ("Processing", "waiting_gpu", True),
        ("Completed", "completed", True),
        ("Failed", "failed", True),
    ],
)
def test_not_started_queue_rank_is_absent_for_ineligible_states(public_status, state, started):
    job, _ = queued("state", timezone.now(), public_status=public_status, started=started)
    with patch.object(
        status, "waiting_queue_snapshot", side_effect=AssertionError("No queue scan needed")
    ):
        assert status.queue_estimate(job, state) is None


def test_ui_status_adds_approximation_without_changing_strict_queue_contract():
    import json

    from api.utils.job_utils import get_queue_position
    from api.views.job_views import job_status

    now = timezone.now()
    queued("older", now - timedelta(seconds=1))
    job, _ = queued("current", now)
    with override_settings(EXECUTION_V2_ENABLED=True):
        response = job_status(RequestFactory().get("/api/job-status/current/"), job.public_id)
        data = json.loads(response.content)
        assert get_queue_position(job) is None
    assert data["queue_position"] is None
    assert data["queue_estimate"]["position"] == 2


def test_very_large_backlogs_use_bounded_memory_and_database_rank_for_the_tail(monkeypatch):
    monkeypatch.setattr(status, "_QUEUE_SNAPSHOT_LIMIT", 2)
    now = timezone.now()
    jobs = [queued("large" + str(i), now)[0] for i in range(4)]
    estimate = status.queue_estimate(jobs[-1], "waiting_gpu")
    assert len(status._queue_snapshot["jobs"]) == 2
    assert estimate["position"] == 4 and estimate["waiting_jobs"] == 4
    new, _ = queued("newtail", now + timedelta(seconds=1))
    estimate = status.queue_estimate(new, "waiting_gpu")
    assert estimate["position"] == 5 and estimate["waiting_jobs"] == 5
