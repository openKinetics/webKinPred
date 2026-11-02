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
