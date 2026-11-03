from copy import deepcopy
from unittest.mock import Mock

import pytest
from api.execution.resources import GIB
from tools.executor import agent as module


@pytest.fixture
def executor(tmp_path, monkeypatch):
    clock = [100.0]
    monkeypatch.setattr(module.time, "monotonic", lambda: clock[0])
    agent = module.Agent(
        {
            "root": str(tmp_path),
            "host_id": "test",
            "budgets": {"cpu": 16, "ram": 64 * GIB, "disk": 128 * GIB, "gpu_slots": 5},
        }
    )
    resources = {"cpu": 1, "ram": GIB, "disk": GIB, "vram": 0}
    calls = []

    def request(operation, data=None, ident=None):
        calls.append((operation, data, ident))
        if operation == "claim":
            ident = str(sum(c[0] == "claim" for c in calls))
            return {"attempt": {"id": ident, "token": "test", "resources": deepcopy(resources)}}
        return {}

    def stage(request):
        agent.jobs[request["id"]] = {
            "request": request,
            "container": None,
            "renewed": clock[0],
        }

    agent.request = request
    agent.stage = stage
    telemetry = {"ram_available": 128 * GIB, "disk_free": 256 * GIB, "attempts": {}}
    try:
        yield agent, resources, calls, telemetry, clock
    finally:
        agent.transfers.shutdown(wait=True)


def test_burst_stages_at_most_four_attempts_and_bounds_pending_transfers(executor):
    agent, _, calls, telemetry, _ = executor
    agent.claim_available(telemetry)
    assert len(agent.jobs) == 4 and len(calls) == 4
    agent.claim_available(telemetry)
    assert len(calls) == 4


@pytest.mark.parametrize("resource", ["ram", "vram"])
def test_unspent_reservations_protect_simultaneous_model_loading(executor, resource):
    agent, resources, calls, telemetry, _ = executor
    if resource == "ram":
        agent.config["budgets"]["ram"] = GIB
    else:
        resources["vram"] = 8 * GIB
        telemetry.update(gpu_uuid="test-gpu", vram_free=16 * GIB, vram_total=20 * GIB)
    agent.claim_available(telemetry)
    assert len(agent.jobs) == 1
    assert calls[-1][0] == "error" and calls[-1][1]["stopped"]
    assert calls[-1][1]["category"] == "waiting_" + resource
    assert sum(c[0] == "claim" for c in calls) == 2


def test_burst_yields_before_heartbeat_supervision_is_due(executor):
    agent, resources, calls, telemetry, clock = executor
    agent.jobs["existing"] = {
        "request": {"resources": resources},
        "container": "running",
        "renewed": clock[0] - 5,
    }
    agent.claim_available(telemetry)
    assert calls == []


def test_burst_has_a_wall_time_budget(executor):
    agent, _, calls, telemetry, clock = executor
    request = agent.request

    def slow(*args):
        clock[0] += 2.1
        agent.last_contact = clock[0]
        return request(*args)

    agent.request = slow
    agent.claim_available(telemetry)
    assert len(agent.jobs) == 2
    assert sum(c[0] == "claim" for c in calls) == 2


def test_paused_agent_does_not_admit_work(executor):
    agent, _, calls, telemetry, _ = executor
    agent.claim_available(telemetry, paused=True)
    assert calls == []


def test_saturated_running_host_still_polls_for_foreground_preemption(executor):
    agent, resources, _, telemetry, clock = executor
    agent.config["budgets"]["cpu"] = 1
    agent.jobs["busy"] = {
        "request": {"resources": resources},
        "container": "running",
        "renewed": clock[0],
    }
    agent.request = Mock(return_value={"attempt": None})
    agent.claim_available(telemetry)
    agent.request.assert_called_once_with("claim")


def test_empty_poll_backoff_is_bounded_and_completion_wakes_admission(executor):
    agent, _, _, telemetry, clock = executor
    agent.request = Mock(return_value={"attempt": None})
    for delay in [1, 2, 4, 5, 5]:
        agent.last_contact = clock[0]
        before = agent.request.call_count
        agent.claim_available(telemetry)
        assert agent.request.call_count == before + 1
        assert agent.next_claim_at == clock[0] + delay
        clock[0] += delay - 0.1
        agent.claim_available(telemetry)
        assert agent.request.call_count == before + 1
        clock[0] += 0.1
    # A locally completed allocation changes ownership and clears an idle delay.
    agent.claim_owners = frozenset({"finished"})
    agent.next_claim_at = clock[0] + 5
    agent.last_contact = clock[0]
    before = agent.request.call_count
    agent.claim_available(telemetry)
    assert agent.request.call_count == before + 1
    assert agent.next_claim_at == clock[0] + 1


def test_empty_poll_delay_never_delays_lease_supervision_or_resumption(executor):
    agent, _, _, telemetry, clock = executor
    agent.request = Mock(return_value={"attempt": None})
    agent.next_claim_at = clock[0] + 5
    agent.claim_available(telemetry, paused=True)
    agent.claim_available(telemetry, paused=False)
    agent.request.assert_called_once_with("claim")
