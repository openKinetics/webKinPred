import json

from api.execution.resources import GIB
from tools.executor import compare_concurrency as module
from tools.executor.profiling_suite import dataset_manifest


def test_comparison_waits_for_cuda_context_release_before_concurrent_launch(tmp_path, monkeypatch):
    dataset = tmp_path / "dataset"
    dataset.mkdir()
    (dataset / "input").write_text("copied probe")
    reference = tmp_path / "reference.json"
    reference.write_text(
        json.dumps(
            {
                "exit_code": 0,
                "oom_killed": False,
                "peaks": {"ram": 1024**2, "pids": 2, "vram": 2 * GIB, "cpu": 1, "disk": 0},
                "configuration": {
                    "dataset_checksum": dataset_manifest(dataset)["checksum"],
                    "image": "sha256:" + "a" * 64,
                    "gpu_uuid": "gpu",
                },
            }
        )
    )
    samples = []
    waits = []
    launches = []

    def sample(*args):
        samples.append(1)
        return {
            "ram_available": 32 * GIB,
            "disk_free": 32 * GIB,
            "cpu_count": 8,
            "cpu_used": 0,
            "gpu_uuid": "gpu",
            "vram_total": 20 * GIB,
            "vram_free": 6 * GIB if len(samples) == 3 else 10 * GIB,
        }

    def probe(runtime, path):
        launches.append(runtime["scratch"])
        return {"exit_code": 0, "oom_killed": False}

    monkeypatch.setattr(module, "snapshot", sample)
    monkeypatch.setattr(module, "run_probe", probe)
    monkeypatch.setattr(module.time, "sleep", lambda n: waits.append(n))
    case = {"dataset": str(dataset), "reference_report": str(reference)}
    result = module.compare(
        {
            "probes": [dict(case), dict(case)],
            "repetitions": 1,
            "budgets": {"cpu": 8, "ram": 32 * GIB, "disk": 32 * GIB, "gpu_slots": 2},
        },
        tmp_path / "results",
    )
    assert waits == [1] and len(launches) == 4 and result["repetitions"] == 1


def test_profile_scratch_limit_stops_only_its_isolated_container(tmp_path):
    import os
    import subprocess

    import pytest
    from tools.executor.profile import run_probe

    image = os.environ.get("WEBKINPRED_TEST_EXECUTOR_IMAGE")
    if not image:
        pytest.skip("Pinned Docker rehearsal image required")
    scratch = tmp_path / "scratch"
    scratch.mkdir()
    report = tmp_path / "report.json"
    config = {
        "isolated": True,
        "dataset_checksum": "a" * 64,
        "image": image,
        "scratch": str(scratch),
        "memory_limit": 64 * 1024**2,
        "cpu": 1,
        "scratch_limit": 1024,
        "command": [
            "python",
            "-c",
            "from pathlib import Path; import time; "
            "Path('/work/probe').write_bytes(b'x'*4096); time.sleep(10)",
        ],
    }
    try:
        with pytest.raises(RuntimeError, match="scratch limit"):
            run_probe(config, report)
        evidence = json.loads(report.read_text())
        assert evidence["peaks"]["disk"] >= 4096 and evidence["exit_code"] == -1
        container = evidence["container"]
        assert (
            subprocess.check_output(
                ["docker", "inspect", container, "--format", "{{.State.Running}}"], text=True
            ).strip()
            == "false"
        )
    finally:
        if report.exists():
            container = json.loads(report.read_text()).get("container", "")
            if container.startswith("wk-profile-"):
                subprocess.run(["docker", "rm", "-f", container], check=True, capture_output=True)
