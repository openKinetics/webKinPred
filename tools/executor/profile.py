"""Profile approved container probes; never invokes production job submission APIs."""

from __future__ import annotations

import argparse
import json
import subprocess
import time
import uuid
from pathlib import Path

import psutil
from api.execution.artifacts import atomic_json
from api.execution.resources import reservation


def cgroup(pid):
    relative = next(
        line.split(":", 2)[2]
        for line in Path(f"/proc/{pid}/cgroup").read_text().splitlines()
        if line.startswith("0::")
    )
    return Path("/sys/fs/cgroup") / relative.lstrip("/")


def read_int(path):
    try:
        return int(path.read_text())
    except (OSError, ValueError):
        return 0


def scratch_bytes(root):
    """An atomic output publication can rename a file between walk and stat."""
    total = 0
    for path in root.rglob("*"):
        try:
            if path.is_file():
                total += path.stat().st_size
        except FileNotFoundError:
            continue
    return total


def check_availability(config):
    """Optional host-side guard gives existing platform jobs priority over probes."""
    if config.get("availability_file"):
        state = json.loads(Path(config["availability_file"]).read_text())
        age = time.time() - state.get("observed_at", 0)
        if state.get("safe") is not True or not 0 <= age <= 2:
            raise RuntimeError("Profiling paused: host guard is busy or stale")


def run_probe(config, output: Path) -> dict:
    if not config.get("isolated") or not config.get("dataset_checksum"):
        raise ValueError("Profiling requires isolated copied inputs and a dataset checksum")
    if "sha256:" not in config["image"]:
        raise ValueError("Profiling requires an immutable image")
    check_availability(config)
    scratch = Path(config["scratch"]).resolve()
    scratch.mkdir(parents=True, exist_ok=True)
    (scratch / "media").mkdir(exist_ok=True)
    ident = "wk-profile-" + uuid.uuid4().hex
    command = [
        "docker",
        "run",
        "-d",
        "--name",
        ident,
        "--label",
        "org.webkinpred.profiling=true",
        "--network",
        "none",
        "--read-only",
        "--memory",
        str(config["memory_limit"]),
        "--memory-swap",
        str(config["memory_limit"]),
        "--cpus",
        str(config["cpu"]),
        "--pids-limit",
        str(config.get("pids_limit", 256)),
        "--shm-size",
        str(min(config["memory_limit"] // 4, 1024**3)),
        "--mount",
        f"type=bind,src={scratch},dst=/work",
        "--mount",
        f"type=bind,src={scratch / 'media'},dst=/app/media",
        "--env",
        "TMPDIR=/work",
        "--env",
        "WEBKINPRED_PROFILING=1",
    ]
    for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        command += ["--env", key + "=" + str(max(1, int(config["cpu"])))]
    for key, value in config.get("environment", {}).items():
        command += ["--env", key + "=" + str(value)]
    if config.get("gpu_uuid"):
        if config.get("gpu_runtime"):
            command += [
                "--runtime",
                config["gpu_runtime"],
                "--env",
                "NVIDIA_VISIBLE_DEVICES=" + config["gpu_uuid"],
                "--env",
                "NVIDIA_DRIVER_CAPABILITIES=compute,utility",
            ]
        else:
            command += ["--gpus", "device=" + config["gpu_uuid"]]
    else:
        command += ["--env", "CUDA_VISIBLE_DEVICES=", "--env", "NVIDIA_VISIBLE_DEVICES=void"]
    for mount in config.get("mounts", []):
        command += ["--mount", f"type=bind,src={mount['source']},dst={mount['target']},readonly"]
    command += [config["image"], *config["command"]]
    subprocess.check_output(command, text=True)
    started = time.monotonic()
    peaks = {"ram": 0, "disk": 0, "pids": 0, "vram": 0, "cpu": config["cpu"]}
    group = None
    samples = []
    owned_pids = set()
    try:
        while True:
            check_availability(config)
            info = json.loads(subprocess.check_output(["docker", "inspect", ident], text=True))[0]
            if info["State"]["Pid"]:
                try:
                    group = cgroup(info["State"]["Pid"])
                except (OSError, StopIteration):
                    pass  # Process may exit between Docker inspect and /proc lookup.
            sample = {"seconds": time.monotonic() - started}
            if group:
                sample["ram"] = max(
                    read_int(group / "memory.current"), read_int(group / "memory.peak")
                )
                sample["pids"] = read_int(group / "pids.current")
                sample["swap"] = read_int(group / "memory.swap.current")
                for name in ["memory.events", "memory.pressure", "cpu.stat", "io.stat"]:
                    try:
                        sample[name] = (group / name).read_text()
                    except OSError:
                        pass
            if info["State"]["Pid"]:
                try:
                    process = psutil.Process(info["State"]["Pid"])
                    tree = [process, *process.children(recursive=True)]
                    owned_pids.update(p.pid for p in tree)
                    memory = [p.memory_full_info() for p in tree]
                    sample["process_tree"] = {
                        "rss": sum(m.rss for m in memory),
                        "pss": sum(getattr(m, "pss", 0) for m in memory),
                        "threads": sum(p.num_threads() for p in tree),
                        "processes": len(tree),
                        "pids": [p.pid for p in tree],
                    }
                except (psutil.Error, OSError):
                    pass
            # Scratch is isolated; no scan of a shared production cache.
            sample["disk"] = scratch_bytes(scratch)
            if config.get("gpu_uuid"):
                # Attribute the platform tree; retain external activity separately.
                sample["gpu_processes"] = subprocess.check_output(
                    [
                        "nvidia-smi",
                        "-i",
                        config["gpu_uuid"],
                        "--query-compute-apps=pid,used_memory",
                        "--format=csv,noheader,nounits",
                    ],
                    text=True,
                )
                sample["gpu_device"] = subprocess.check_output(
                    [
                        "nvidia-smi",
                        "-i",
                        config["gpu_uuid"],
                        "--query-gpu=memory.used,memory.free,utilization.gpu",
                        "--format=csv,noheader,nounits",
                    ],
                    text=True,
                )
                processes = []
                for line in sample["gpu_processes"].splitlines():
                    pid, memory = line.split(",")
                    processes.append((int(pid.strip()), int(memory.strip()) * 1024**2))
                assigned = sum(memory for pid, memory in processes if pid in owned_pids)
                sample["vram"] = assigned
                sample["external_vram"] = (
                    int(sample["gpu_device"].split(",")[0]) * 1024**2 - assigned
                )
            for key in peaks:
                peaks[key] = max(peaks[key], sample.get(key, 0))
            samples.append(sample)
            if not info["State"]["Running"]:
                break
            if time.monotonic() - started > config.get("timeout_seconds", 3600):
                subprocess.run(["docker", "stop", "-t", "5", ident], check=True)
                raise TimeoutError("Isolated profiling deadline exceeded")
            time.sleep(0.1)
        report = {
            "configuration": config,
            "elapsed_seconds": time.monotonic() - started,
            "peaks": peaks,
            "reservation": reservation(peaks),
            "exit_code": info["State"]["ExitCode"],
            "oom_killed": info["State"].get("OOMKilled", False),
            "samples": samples,
            "framework_metrics": [
                json.loads(p.read_text()) for p in (scratch / "framework-metrics").glob("*.json")
            ],
            "logs": subprocess.check_output(
                ["docker", "logs", ident], stderr=subprocess.STDOUT, text=True
            ),
            "validated": False,
        }
        atomic_json(output, report)
        return report
    except Exception as exc:
        atomic_json(
            output,
            {
                "configuration": config,
                "elapsed_seconds": time.monotonic() - started,
                "peaks": peaks,
                "samples": samples,
                "exit_code": -1,
                "oom_killed": False,
                "profiling_error": str(exc),
                "validated": False,
            },
        )
        raise
    finally:
        subprocess.run(["docker", "stop", "-t", "5", ident], capture_output=True)
        # Retain container/logs until evidence is reviewed; never auto-remove failed probes.


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("configuration", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    run_probe(json.loads(args.configuration.read_text()), args.output)


if __name__ == "__main__":
    main()
