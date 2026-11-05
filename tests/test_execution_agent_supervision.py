"""Supervision work per tick stays bounded as the number of attempts grows."""

import json
import threading
import time
import urllib.error
from concurrent.futures import Future
from unittest.mock import Mock

import pytest
from api.execution.artifacts import sha256_file
from tools.executor import agent as module
from tools.executor.agent import Agent


@pytest.fixture
def agent(tmp_path):
    executor = Agent({"root": str(tmp_path), "host_id": "test", "budgets": {"disk_headroom": 0}})
    try:
        yield executor
    finally:
        executor.transfers.shutdown(wait=True)
        executor.control.shutdown(wait=True)


def done(value=None):
    future = Future()
    future.set_result(value)
    return future


def wait_until(condition, timeout=5):
    deadline = time.monotonic() + timeout
    while not condition():
        if time.monotonic() > deadline:
            pytest.fail("Condition was not reached")
        time.sleep(0.01)


def test_due_leases_are_renewed_with_one_request(agent):
    now = time.monotonic()
    agent.jobs = {
        "due": {"request": {"token": "a"}, "container": "c1", "renewed": now - 5},
        "fresh": {"request": {"token": "b"}, "container": "c2", "renewed": now},
        "launching": {
            "request": {"token": "c"},
            "container": None,
            "transfer": done({}),
            "renewed": now,
        },
        "committing": {
            "request": {"token": "d"},
            "container": "c4",
            "renewed": now - 9,
            "completion": Future(),
        },
        # Committed but not yet finalized: a renewal would be rejected and
        # divert it from ordinary cleanup.
        "committed": {
            "request": {"token": "e"},
            "container": "c5",
            "renewed": now - 9,
            "completion": done({"state": "completed"}),
        },
    }
    progress = agent.root / "attempts/due/progress.json"
    progress.parent.mkdir(parents=True)
    progress.write_text(json.dumps({"progress": 7}))
    agent.request = Mock(
        return_value={"results": {"due": {"lease_seconds": 30}, "launching": {"lease_seconds": 30}}}
    )
    agent.renew_leases()
    agent.request.assert_called_once()
    operation, body = agent.request.call_args.args
    assert operation == "heartbeats"
    assert sorted(entry["id"] for entry in body["attempts"]) == ["due", "launching"]
    due = next(entry for entry in body["attempts"] if entry["id"] == "due")
    assert due == {"id": "due", "token": "a", "container_id": "c1", "progress": 7}
    assert agent.jobs["due"]["renewed"] > now - 5
    assert agent.jobs["launching"]["launch_renewed"] is True
    assert agent.jobs["committing"]["renewed"] == now - 9
    assert agent.jobs["committed"]["renewed"] == now - 9


def test_launch_uses_the_batched_post_transfer_renewal(agent):
    job = {
        "request": {"id": "x", "token": "t", "kind": "predict", "payload": {}},
        "container": None,
        "transfer": done({}),
        "renewed": time.monotonic(),
    }
    agent.jobs = {"x": job}
    agent.request = Mock(return_value={"results": {"x": {"lease_seconds": 30}}})
    launched = []
    agent.launch = lambda request: launched.append(request["id"])
    agent.inspect_many = lambda containers: {}
    agent.tick_jobs()
    assert launched == ["x"]
    assert [call.args[0] for call in agent.request.call_args_list] == ["heartbeats"]


def test_older_controller_falls_back_to_individual_heartbeats(agent):
    agent.jobs = {"a": {"request": {"token": "t"}, "container": None, "renewed": 0}}
    agent.request = Mock(side_effect=urllib.error.HTTPError("url", 404, "missing", {}, None))
    agent.renew_leases()
    assert agent.batch_heartbeats is False
    agent.request.reset_mock()
    agent.renew_leases()
    agent.request.assert_not_called()


@pytest.mark.parametrize("running", [True, False])
def test_rejected_batched_renewal_fences_like_a_single_heartbeat(agent, running):
    job = {"request": {"id": "a", "token": "t"}, "container": "c1", "renewed": 0}
    agent.jobs = {"a": job}
    calls = []

    def request(operation, data=None, attempt=None):
        calls.append(operation)
        if operation == "heartbeats":
            return {"results": {"a": {"status": 409, "error": "Attempt lease expired"}}}
        return {}

    agent.request = request
    state = {"Id": "c1", "State": {"Running": running}}
    agent.inspect_many = lambda containers: {"c1": state}
    agent.docker = Mock(return_value=json.dumps([state]))
    agent.tick_jobs()
    if running:
        # Compute is fenced; the exit path later preserves checkpoints.
        agent.docker.assert_called_once_with("stop", "-t", "5", "c1")
        assert "a" in agent.jobs
    else:
        assert calls == ["heartbeats", "reconcile"] and not agent.jobs


def test_completion_commit_runs_beside_supervision_and_is_resubmitted(agent):
    work = agent.root / "attempts/a"
    work.mkdir(parents=True)
    (work / "outcome.json").write_text(json.dumps({"artifacts": [], "checkpoint": {"n": 1}}))
    job = {
        "request": {"id": "a", "token": "t", "kind": "predict", "payload": {}},
        "container": "c1",
        "renewed": time.monotonic(),
    }
    agent.jobs = {"a": job}
    exited = {"State": {"Running": False, "ExitCode": 0, "FinishedAt": "2026-11-04T00:00:00Z"}}
    agent.docker = Mock(
        side_effect=lambda *args: json.dumps([exited]) if args[0] == "inspect" else ""
    )
    release, attempts = threading.Event(), []

    def request(operation, data=None, attempt=None):
        assert operation == "complete" and data["checkpoint"] == {"n": 1}
        attempts.append(data)
        if len(attempts) == 1:
            raise TimeoutError("reply lost")
        release.wait(5)
        return {"state": "completed"}

    agent.request = request
    wait_until(lambda: agent.tick_job("a", job) or "completion" in job)
    wait_until(lambda: job["completion"].done())
    agent.tick_job("a", job)  # A lost reply is retried, never reported as an error.
    assert "completion" not in job and agent.jobs
    agent.tick_job("a", job)
    agent.tick_job("a", job)
    assert agent.jobs and not release.is_set()  # Supervision continues meanwhile.
    release.set()
    wait_until(lambda: job["completion"].done())
    # A committed attempt is finalized, never renewed, even with a stale lease
    # time (the request double rejects any operation other than completion).
    job["renewed"] = time.monotonic() - 10
    agent.tick_job("a", job)
    assert not agent.jobs and len(attempts) == 2
    assert agent.docker.call_args.args == ("rm", "c1")


@pytest.mark.parametrize("code", [500, 503, 409])
def test_server_error_on_commit_is_resubmitted_but_lease_rejection_is_not(agent, code):
    work = agent.root / "attempts/a"
    work.mkdir(parents=True)
    (work / "outcome.json").write_text(json.dumps({"artifacts": [], "checkpoint": {}}))
    exited = {"State": {"Running": False, "ExitCode": 0, "FinishedAt": "2026-11-04T00:00:00Z"}}
    agent.docker = Mock(return_value=json.dumps([exited]))
    job = {
        "request": {"id": "a", "token": "t", "kind": "predict", "payload": {}},
        "container": "c1",
        "renewed": time.monotonic(),
        "publication": done({"artifacts": [], "checkpoint": {}}),
        "completion": Future(),
    }
    job["completion"].set_exception(urllib.error.HTTPError("url", code, "error", {}, None))
    agent.jobs = {"a": job}
    if code == 409:
        with pytest.raises(urllib.error.HTTPError):
            agent.tick_job("a", job)
    else:
        # A deadlock victim or overloaded server must not stall every later tick.
        agent.tick_job("a", job)
        assert "completion" not in job and agent.jobs


def test_final_publication_reproduces_the_runner_outcome_without_a_container(agent):
    work = agent.root / "attempts/p"
    result = work / "media/jobs/JOB1/output.csv"
    result.parent.mkdir(parents=True)
    result.write_text("Substrate,Prediction\nCCO,1.0\n")
    checksum, size = sha256_file(result), result.stat().st_size
    output = {"id": "o" * 64, "path": "jobs/JOB1/output.csv", "identity": {"job": "JOB1"}}
    request = {
        "id": "p",
        "token": "t",
        "kind": "assemble",
        "image": "sha256:" + "a" * 64,
        "payload": {
            "phase": "publish",
            "outputs": [output],
            "assembly_checkpoint": {"completed_reactions": 1},
        },
    }
    job = {"request": request, "container": None, "transfer": done({}), "renewed": 0}
    agent.jobs = {"p": job}
    agent.docker = Mock(side_effect=AssertionError("publication must not start a container"))
    committed = []
    agent.request = lambda operation, data=None, attempt=None: (
        committed.append((operation, data)) or {"lease_seconds": 30}
    )
    agent.tick_job("p", job)
    outcome = json.loads((work / "outcome.json").read_text())
    assert outcome == {
        "artifacts": [
            {
                "id": output["id"],
                "identity": output["identity"],
                "checksum": checksum,
                "size": size,
                "metadata": {},
                "source": "media/jobs/JOB1/output.csv",
            }
        ],
        "checkpoint": {"completed_reactions": 1},
    }
    wait_until(lambda: agent.tick_job("p", job) or "completion" in job)
    wait_until(lambda: job["completion"].done())
    agent.tick_job("p", job)
    operation, data = committed[-1]
    assert operation == "complete" and data["checkpoint"] == {"completed_reactions": 1}
    assert data["artifacts"][0]["checksum"] == checksum
    assert "source" not in data["artifacts"][0]
    assert agent.store.path(checksum).exists()
    assert not agent.jobs


def test_missing_publication_input_is_reported_like_a_failed_launch(agent):
    request = {
        "id": "p",
        "token": "t",
        "kind": "assemble",
        "payload": {"phase": "publish", "outputs": [{"id": "o", "path": "jobs/X/output.csv"}]},
    }
    job = {"request": request, "container": None, "transfer": done({}), "renewed": 0}
    agent.jobs = {"p": job}
    agent.docker = Mock(return_value="")
    reports = []
    agent.request = lambda operation, data=None, attempt=None: reports.append((operation, data))
    agent.tick_job("p", job)
    assert reports[-1][0] == "error" and reports[-1][1]["stopped"] is True
    assert not agent.jobs


@pytest.mark.parametrize("container_created", [False, True])
def test_refused_launch_releases_materialized_inputs_unless_a_container_exists(
    agent, container_created
):
    from api.execution.errors import RecoverableExecutionError

    work = agent.root / "attempts/r"
    (work / "media/sequence_info").mkdir(parents=True)
    (work / "media/sequence_info/residues.npy").write_bytes(b"x" * 1024)
    (work / "request.json").write_text("{}")
    request = {"id": "r", "token": "t", "kind": "predict", "payload": {}}
    job = {"request": request, "container": None, "transfer": done({}), "renewed": 0}
    agent.jobs = {"r": job}

    def refuse(request):
        error = RecoverableExecutionError("Capacity changed during artifact transfer")
        error.category = "waiting_cpu"
        raise error

    agent.launch = refuse
    agent.docker = Mock(return_value="ambiguous-id" if container_created else "")
    agent.request = Mock(return_value={"lease_seconds": 30})
    agent.tick_job("r", job)
    assert not agent.jobs
    assert (work / "request.json").exists()  # The journal is always retained.
    assert (work / "media").exists() is container_created


def test_one_inspect_call_serves_every_container(agent, monkeypatch):
    calls = []

    def run(args, **kwargs):
        calls.append(args)
        return Mock(stdout=json.dumps([{"Id": "c1"}, {"Id": "c2"}]), returncode=1)

    monkeypatch.setattr(module.subprocess, "run", run)
    assert set(agent.inspect_many(["c1", "c2", "gone"])) == {"c1", "c2"}
    assert calls == [["docker", "inspect", "c1", "c2", "gone"]]
    assert agent.inspect_many([]) == {}


def test_loop_summary_reports_mean_phase_costs_once_a_minute(agent, monkeypatch, capsys):
    clock = [1000.0]
    monkeypatch.setattr(module.time, "monotonic", lambda: clock[0])
    agent.timings_since = clock[0]
    for _ in range(2):
        with agent.timed("claim"):
            clock[0] += 1.5
        agent.count("claimed", 3)
        agent.iterations += 1
        agent.report_timings()
    assert capsys.readouterr().out == ""
    clock[0] += 60
    agent.report_timings()
    summary = json.loads(capsys.readouterr().out)
    assert summary["event"] == "executor.loop" and summary["iterations"] == 2
    assert summary["mean_phase_seconds"] == {"claim": 1.5} and summary["claimed"] == 6
    assert agent.iterations == 0 and agent.counts == {}


def test_external_cpu_reports_the_recent_floor_not_churn_spikes(agent, monkeypatch):
    clock = [500.0]
    monkeypatch.setattr(module.time, "monotonic", lambda: clock[0])
    used = iter([5.0, 1.0, 9.0, 8.0])
    monkeypatch.setattr(
        module, "snapshot", lambda root, gpu: {"cpu_used": next(used), "ram_available": 1}
    )
    monkeypatch.setattr(module, "container_observations", lambda containers: {})
    agent.docker = Mock(return_value="")
    reported = []
    for step in [0, 5, 5, 25]:
        clock[0] += step
        data = agent.observe()
        reported.append((data["cpu_external"], data["cpu_external_raw"]))
    # The 1.0 sample ages out of the 20-second window before the last reading.
    assert reported == [(5.0, 5.0), (1.0, 1.0), (1.0, 9.0), (8.0, 8.0)]


def test_only_finished_containers_release_compute_reservations(agent, monkeypatch):
    monkeypatch.setattr(module, "snapshot", lambda root, gpu: {"cpu_used": 0.0})
    monkeypatch.setattr(
        module, "container_observations", lambda containers: {k: {"ram": 1} for k in containers}
    )
    agent.docker = Mock(return_value="")
    agent.jobs = {
        "running": {"request": {}, "container": "c1", "renewed": 0},
        "created": {"request": {}, "container": "c2", "renewed": 0},
        "exited": {"request": {}, "container": "c3", "renewed": 0},
        "publishing": {"request": {}, "container": "c4", "renewed": 0, "publication": Future()},
        "inline": {
            "request": {},
            "container": None,
            "renewed": 0,
            "inline": {"State": {"Running": False, "ExitCode": 0}},
        },
        "transfer": {"request": {}, "container": None, "renewed": 0, "transfer": Future()},
    }
    agent.inspected = {
        "c1": {"State": {"Running": True, "Status": "running"}},
        "c2": {"State": {"Running": False, "Status": "created"}},
        "c3": {"State": {"Running": False, "Status": "exited"}},
        "c4": {"State": {"Running": False, "Status": "exited"}},
    }
    attempts = agent.observe()["attempts"]
    assert {k for k, v in attempts.items() if v.get("released")} == {
        "exited",
        "publishing",
        "inline",
    }
    assert attempts["running"] == {"ram": 1} and attempts["created"] == {"ram": 1}
    assert "transfer" not in attempts


def test_launch_rechecks_can_share_a_fresh_host_snapshot(agent, monkeypatch):
    snapshots = []
    monkeypatch.setattr(
        module,
        "snapshot",
        lambda root, gpu: snapshots.append(1) or {"cpu_used": 0.0, "ram_available": 1},
    )
    monkeypatch.setattr(module, "container_observations", lambda containers: {})
    agent.docker = Mock(return_value="")
    assert agent.recent_observation(2.0) is None
    measured = agent.observe()
    assert agent.recent_observation(2.0) == measured
    assert len(snapshots) == 1 and agent.docker.call_count == 1
    agent.observe()
    # A new measurement, but the legacy-consumer scan stays rate limited.
    assert len(snapshots) == 2 and agent.docker.call_count == 1
