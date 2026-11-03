import json
import time
from unittest.mock import Mock

import pytest
from api.execution.progress import stage_summaries
from django.db import connection
from django.test.utils import CaptureQueriesContext
from tests.test_execution import ready_work as ready_work
from tools.executor.agent import Agent


@pytest.mark.django_db(transaction=True)
def test_progress_reads_counters_without_loading_input_manifests(ready_work):
    _, stages = ready_work
    stage = stages[0]
    stage.payload = {
        "rows": [{"large_private_manifest": "x" * 100000}],
        "shape": {"units": 23},
        "outputs": [{}, {}],
    }
    stage.progress = 5
    stage.save()
    with CaptureQueriesContext(connection) as queries:
        rows = stage_summaries(stage.execution)
    row = next(r for r in rows if r["output_count"] == 2)
    assert row["units"] == 23 and row["has_rows"] and row["progress"] == 5
    assert len(queries) == 1 and '"api_workstage"."payload"' not in queries[0]["sql"]
    assert "large_private_manifest" not in json.dumps(rows)


@pytest.mark.django_db(transaction=True)
def test_shared_progress_is_durable_bounded_and_refreshes_again_after_later_events(
    ready_work, monkeypatch
):
    from api.execution import progress
    from api.execution.models import ExecutionOutbox, ExecutionSpec

    _, stages = ready_work
    owners = [s.execution for s in stages]
    shared = ExecutionSpec.objects.create(
        specification={"shared_protein_batch": {"source_requests": [{"id": x.pk} for x in owners]}},
        checksum="a" * 64,
        input_checksum="b" * 64,
    )
    refresh = Mock()
    monkeypatch.setattr(progress, "refresh", refresh)
    progress.request_refresh(shared)
    progress.request_refresh(shared)
    refresh.assert_not_called()
    event = ExecutionOutbox.objects.get(key=f"progress:{shared.pk}")
    assert len(event.event["progress_execution_ids"]) == 2
    assert progress.project_pending(limit=1) == 1
    assert refresh.call_count == 1
    # Completion while the initial projection is partially drained must not
    # lose the final state update for an owner already visited.
    progress.request_refresh(shared)
    assert progress.project_pending(limit=1) == 1
    assert progress.project_pending(limit=1) == 1
    event.refresh_from_db()
    assert event.published_at is not None
    progress.request_refresh(shared)
    assert progress.project_pending(limit=4) == 2
    assert ExecutionOutbox.objects.filter(key=f"progress:{shared.pk}").count() == 1


@pytest.mark.django_db(transaction=True)
def test_busy_optional_projection_preserves_pending_owner_and_retries(ready_work, monkeypatch):
    from api.execution import progress
    from api.execution.models import ExecutionOutbox, ExecutionSpec
    from django.db import OperationalError

    owner = ready_work[1][0].execution
    shared = ExecutionSpec.objects.create(
        specification={"shared_protein_batch": {"source_requests": [{"id": owner.pk}]}},
        checksum="a" * 64,
        input_checksum="b" * 64,
    )
    progress.request_refresh(shared)
    refresh = progress.refresh
    monkeypatch.setattr(progress, "refresh", Mock(side_effect=OperationalError("busy")))
    assert progress.project_pending() == 0
    event = ExecutionOutbox.objects.get(key=f"progress:{shared.pk}")
    assert event.published_at is None and event.event["progress_execution_ids"] == [owner.pk]
    monkeypatch.setattr(progress, "refresh", refresh)
    assert progress.project_pending() == 1


@pytest.mark.django_db(transaction=True)
def test_large_job_progress_coalesces_without_postponing_refresh_on_each_update(
    ready_work, monkeypatch
):
    from datetime import timedelta

    from api.execution import progress
    from api.execution.models import ExecutionOutbox
    from django.utils import timezone

    owner = ready_work[1][0].execution
    owner.job.requested_rows = 17010
    owner.job.save(update_fields=["requested_rows"])
    refresh = Mock()
    monkeypatch.setattr(progress, "refresh", refresh)
    now = timezone.now()
    monkeypatch.setattr(progress.timezone, "now", lambda: now)
    progress.request_refresh(owner)
    refresh.assert_not_called()
    assert progress.project_pending() == 1
    now += timedelta(seconds=5)
    progress.request_refresh(owner)
    event = ExecutionOutbox.objects.get(key=f"progress:{owner.pk}")
    due = event.event["not_before"]
    assert progress.project_pending() == 0
    now += timedelta(seconds=5)
    progress.request_refresh(owner)
    event.refresh_from_db()
    assert event.event["not_before"] == due
    now += timedelta(seconds=5)
    assert progress.project_pending() == 1
    progress.request_refresh(owner, force=True)
    assert progress.project_pending() == 1


@pytest.mark.django_db(transaction=True)
def test_progress_update_arriving_during_scan_is_not_lost(ready_work, monkeypatch):
    from api.execution import progress
    from api.execution.models import ExecutionOutbox, ExecutionSpec

    owner = ready_work[1][0].execution
    shared = ExecutionSpec.objects.create(
        specification={"shared_protein_batch": {"source_requests": [{"id": owner.pk}]}},
        checksum="a" * 64,
        input_checksum="b" * 64,
    )
    progress.request_refresh(shared)
    refresh = Mock(side_effect=lambda *args, **kwargs: progress.request_refresh(shared))
    monkeypatch.setattr(progress, "refresh", refresh)
    assert progress.project_pending(limit=1) == 1
    event = ExecutionOutbox.objects.get(key=f"progress:{shared.pk}")
    assert event.published_at is None
    assert event.event["progress_execution_ids"] == [owner.pk]
    monkeypatch.setattr(progress, "refresh", Mock())
    assert progress.project_pending() == 1
    event.refresh_from_db()
    assert event.published_at is not None


@pytest.mark.django_db(transaction=True)
def test_background_snapshot_does_not_overwrite_newer_public_progress(ready_work, monkeypatch):
    from api.execution import progress
    from api.models import JobProgressStage
    from django.utils import timezone

    owner = ready_work[1][0].execution
    summary = JobProgressStage.objects.create(
        job=owner.job, target="kcat", stage_index=0, predictions_made=7
    )

    def newer_summary(execution):
        JobProgressStage.objects.filter(pk=summary.pk).update(
            predictions_made=8, updated_at=timezone.now()
        )
        return []

    monkeypatch.setattr(progress, "stage_summaries", newer_summary)
    monkeypatch.setattr(progress, "shared_embedding_summaries", lambda execution: [])
    progress.refresh(owner, scan_before_lock=True)
    summary.refresh_from_db()
    assert summary.predictions_made == 8


def test_running_heartbeat_cadence_preserves_fencing(tmp_path, monkeypatch):
    agent = Agent({"root": str(tmp_path), "host_id": "test", "budgets": {"disk_headroom": 0}})
    agent.docker = Mock(return_value=json.dumps([{"State": {"Running": True}}]))
    agent.request = Mock()
    job = {
        "request": {"token": "test", "resources": {"disk": 1024}},
        "container": "owned",
        "renewed": 100,
    }
    monkeypatch.setattr(time, "monotonic", lambda: 104)
    try:
        agent.tick_job("a", job)
        agent.request.assert_not_called()
        monkeypatch.setattr(time, "monotonic", lambda: 105)
        agent.tick_job("a", job)
        assert agent.request.call_count == 1 and job["renewed"] == 105
        monkeypatch.setattr(time, "monotonic", lambda: 125)
        agent.tick_job("a", job)
        assert agent.docker.call_args.args == ("stop", "-t", "5", "owned")
    finally:
        agent.transfers.shutdown(wait=True)


def test_control_calibration_ceilings_can_be_restored_without_restart(tmp_path):
    agent = Agent(
        {
            "root": str(tmp_path),
            "host_id": "test",
            "budgets": {"cpu": 16, "ram": 96, "cpu_headroom": 2},
        }
    )
    try:
        agent.apply_control_budgets(
            {"budgets": {"cpu": 8, "ram": 48, "cpu_headroom": 1, "vram_headroom": 9}}
        )
        assert agent.config["budgets"] == {
            "cpu": 8,
            "ram": 48,
            "cpu_headroom": 2,
            "vram_headroom": 9,
        }
        agent.apply_control_budgets({"budgets": {"cpu": 16, "ram": 96, "cpu_headroom": 2}})
        assert agent.config["budgets"] == {"cpu": 16, "ram": 96, "cpu_headroom": 2}
    finally:
        agent.transfers.shutdown(wait=True)


@pytest.mark.django_db(transaction=True)
def test_public_status_never_loads_stage_manifests_or_attempt_logs(ready_work):
    from datetime import timedelta

    from api.execution.models import ExecutionAttempt
    from api.execution.status import execution_status
    from django.test import override_settings
    from django.utils import timezone

    host, stages = ready_work
    stage = stages[0]
    stage.payload = {"private_manifest": "x" * 100000}
    stage.checkpoint = {"large_result": "y" * 100000}
    stage.state = "running"
    stage.save()
    job = stage.execution.job
    job.status = "Processing"
    job.start_time = timezone.now()
    job.save()
    ExecutionAttempt.objects.create(
        stage=stage,
        host=host,
        state="running",
        lease_until=timezone.now() + timedelta(seconds=30),
        outcome={"log": "z" * 100000},
    )
    with override_settings(EXECUTION_V2_ENABLED=True), CaptureQueriesContext(connection) as queries:
        result = execution_status(job)
    assert result["execution_state"] == "running"
    assert len(queries) == 4
    sql = "\n".join(row["sql"] for row in queries)
    assert '"api_workstage"."payload"' not in sql
    assert '"api_workstage"."checkpoint"' not in sql
    assert '"api_executionattempt"."outcome"' not in sql
    assert '"api_executionspec"."specification"' not in sql
