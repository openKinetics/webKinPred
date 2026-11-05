"""On a shared GPU host attempts run at low CPU weight; GPU work keeps a CPU floor."""

import time

import pytest
from api.execution import telemetry
from api.execution.resources import GIB, admit
from tools.executor import agent as module
from tools.executor.agent import Agent

TELEMETRY = {
    "ram_available": 64 * GIB,
    "disk_free": 64 * GIB,
    "cpu_count": 20,
    "cpu_external": 1,
    "gpu_uuid": "gpu",
    "vram_total": 20 * GIB,
    "vram_free": 18 * GIB,
}
BUDGETS = {
    "ram": 48 * GIB,
    "cpu": 40,
    "disk": 48 * GIB,
    "cpu_headroom": 1,
    "cpu_slots": 40,
    "gpu_slots": 4,
}
CPU = {"ram": GIB, "cpu": 2, "disk": GIB, "vram": 0}
GPU = {"ram": GIB, "cpu": 2, "disk": GIB, "vram": 2 * GIB}
SATURATED = {**TELEMETRY, "cpu_external": 19}  # 20 CPUs - 1 headroom - 19 external


def running(resources, count):
    return [{"resources": resources, "observed": {}}] * count


def test_gpu_work_keeps_its_cpu_floor_behind_saturating_external_work():
    assert admit(GPU, [], BUDGETS, SATURATED, gpu=True).reason == "waiting_cpu"
    budgets = {**BUDGETS, "gpu_cpu_floor": 4}
    assert admit(CPU, [], budgets, SATURATED, gpu=False).reason == "waiting_cpu"
    assert admit(GPU, running(GPU, 1), budgets, SATURATED, gpu=True).fits
    assert admit(GPU, running(GPU, 2), budgets, SATURATED, gpu=True).reason == "waiting_cpu"
    # CPU-only attempts never consume the GPU floor; spare CPUs still admit as before.
    assert admit(GPU, running(CPU, 3), budgets, SATURATED, gpu=True).fits
    assert admit(CPU, running(GPU, 2), budgets, TELEMETRY, gpu=False).fits


def test_sustained_external_gpu_use_stops_only_gpu_admission():
    budgets = {**BUDGETS, "gpu_external_limit": 40}
    assert admit(GPU, [], budgets, {**TELEMETRY, "gpu_external": 39}, gpu=True).fits
    busy = {**TELEMETRY, "gpu_external": 40}
    assert admit(GPU, [], budgets, busy, gpu=True).reason == "gpu_unavailable"
    assert admit(CPU, [], budgets, busy, gpu=False).fits
    assert admit(GPU, [], BUDGETS, busy, gpu=True).fits  # No limit configured.


@pytest.mark.parametrize(
    "budget",
    [
        {"gpu_cpu_floor": -1},
        {"gpu_cpu_floor": True},
        {"gpu_cpu_floor": float("inf")},
        {"gpu_external_limit": float("nan")},
        {"gpu_external_limit": "40"},
    ],
)
def test_invalid_shared_host_budgets_admit_nothing(budget):
    assert admit(CPU, [], {**BUDGETS, **budget}, TELEMETRY, gpu=False).reason == "invalid_budget"


def test_external_gpu_use_excludes_processes_in_the_attempt_slice(tmp_path, monkeypatch):
    groups = {
        "101": "0::/webkinpred.slice/docker-a.scope\n",
        "102": "0::/user.slice/user-1000.slice/session-7.scope\n",
        "103": "0::/system.slice/docker-b.scope\n",
    }
    for pid, group in groups.items():
        (tmp_path / pid).mkdir()
        (tmp_path / pid / "cgroup").write_text(group)
    real = telemetry.Path
    monkeypatch.setattr(
        telemetry, "Path", lambda path: real(str(path).replace("/proc", str(tmp_path), 1))
    )
    sample = """# gpu         pid   type     sm    mem    enc    dec    jpg    ofa    command
# Idx           #    C/G      %      %      %      %      %      %    name
    0       2360     G      -      -      -      -      -      -    Xorg
    0        101     C     60      9      -      -      -      -    python
    0        102     C     25      3      -      -      -      -    python
    0        103     C      5      1      -      -      -      -    python
    0        999     C     50      1      -      -      -      -    gone
"""
    calls = []
    monkeypatch.setattr(
        telemetry.subprocess,
        "check_output",
        lambda args, **kwargs: calls.append(args) or sample,
    )
    assert telemetry.external_gpu_use("GPU-1", "webkinpred.slice") == 30
    assert calls == [["nvidia-smi", "pmon", "-i", "GPU-1", "-c", "1", "-s", "u"]]


@pytest.fixture
def agent(tmp_path):
    executor = Agent({"root": str(tmp_path), "host_id": "gpu", "budgets": {"disk_headroom": 0}})
    try:
        yield executor
    finally:
        executor.transfers.shutdown(wait=True)
        executor.control.shutdown(wait=True)
        executor.verifier.shutdown(wait=True)


class Created(Exception):
    pass


def created_args(agent):
    image = "sha256:" + "a" * 64
    agent.config.update(images=[image], budgets={"ram": 8 * GIB, "cpu": 4, "disk": 8 * GIB})
    request = {
        "id": "a",
        "image": image,
        "kind": "predict",
        "resources": {"ram": GIB, "cpu": 1, "disk": GIB},
    }
    agent.jobs["a"] = {"request": request, "container": None}
    agent.observe = lambda: {"ram_available": 16 * GIB, "disk_free": 16 * GIB, "attempts": {}}

    def docker(*args):
        if args[0] == "create":
            raise Created(list(args))
        return ""

    agent.docker = docker
    with pytest.raises(Created) as created:
        agent.launch(request)
    return created.value.args[0]


def test_attempts_join_the_configured_low_weight_slice(agent):
    assert "--cgroup-parent" not in created_args(agent)
    agent.config.update(attempt_cgroup_parent="webkinpred.slice", attempt_oom_score_adj=500)
    args = created_args(agent)
    assert args[args.index("--cgroup-parent") + 1] == "webkinpred.slice"
    assert args[args.index("--oom-score-adj") + 1] == "500"


def test_external_gpu_samples_reach_telemetry_until_they_go_stale(agent, monkeypatch):
    monkeypatch.setattr(module, "external_gpu_use", lambda uuid, owned: 70)
    monkeypatch.setattr(module, "snapshot", lambda root, gpu: {"cpu_used": 0.0})
    monkeypatch.setattr(module, "container_observations", lambda containers: {})
    agent.docker = lambda *args: ""
    agent.config.update(gpu_uuid="GPU-1", attempt_cgroup_parent="webkinpred.slice")
    agent.sample_gpu_external(iterations=1)
    assert agent.observe()["gpu_external"] == 70
    agent.gpu_external = (time.monotonic() - module.GPU_EXTERNAL_WINDOW_SECONDS - 1, 70)
    assert "gpu_external" not in agent.observe()
