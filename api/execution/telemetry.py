from __future__ import annotations

import json
import shutil
import subprocess
import time
from pathlib import Path

import psutil

_cpu_samples = {}
# A running container keeps its init PID and cgroup. Resolving them once avoids
# a docker CLI call per container on every supervision tick.
_cgroups = {}


def set_memory_high(container, limit):
    """Early pressure threshold for an already identified owned container."""
    info = json.loads(
        subprocess.check_output(["docker", "inspect", container], text=True, timeout=5)
    )[0]
    pid = info["State"]["Pid"]
    relative = next(
        line.split(":", 2)[2]
        for line in Path(f"/proc/{pid}/cgroup").read_text().splitlines()
        if line.startswith("0::")
    )
    (Path("/sys/fs/cgroup") / relative.lstrip("/") / "memory.high").write_text(
        str(int(limit * 0.85))
    )


def _pressure(kind):
    try:
        lines = Path(f"/proc/pressure/{kind}").read_text().splitlines()
        full = next(x for x in lines if x.startswith("full "))
        return float(dict(p.split("=") for p in full.split()[1:])["avg10"]) > 5
    except (OSError, StopIteration, KeyError, ValueError):
        return False


def snapshot(scratch: Path, gpu_uuid: str = "") -> dict:
    data = {
        "ram_available": psutil.virtual_memory().available,
        "ram_total": psutil.virtual_memory().total,
        "disk_free": shutil.disk_usage(scratch).free,
        "memory_pressure": _pressure("memory"),
        "io_pressure": _pressure("io"),
        "cpu_count": len(psutil.Process().cpu_affinity()),
    }
    data["cpu_used"] = psutil.cpu_percent() / 100 * data["cpu_count"]
    if gpu_uuid:
        try:
            result = subprocess.check_output(
                [
                    "nvidia-smi",
                    "-i",
                    gpu_uuid,
                    "--query-gpu=uuid,memory.total,memory.free,utilization.gpu",
                    "--format=csv,noheader,nounits",
                ],
                timeout=3,
                text=True,
            )
            ident, total, free, utilization = [x.strip() for x in result.strip().split(",")]
            data.update(
                gpu_uuid=ident,
                vram_total=int(total) * 1024**2,
                vram_free=int(free) * 1024**2,
                gpu_utilization=int(utilization),
            )
        except (OSError, ValueError, subprocess.SubprocessError):
            pass  # Missing telemetry forbids GPU admission.
    return data


def _container_cgroup(container):
    cached = _cgroups.get(container)
    if cached:
        return cached
    info = json.loads(
        subprocess.check_output(["docker", "inspect", container], text=True, timeout=5)
    )[0]
    pid = info["State"]["Pid"]
    if not pid:
        return None
    relative = next(
        line.split(":", 2)[2]
        for line in Path(f"/proc/{pid}/cgroup").read_text().splitlines()
        if line.startswith("0::")
    )
    _cgroups[container] = pid, Path("/sys/fs/cgroup") / relative.lstrip("/")
    return _cgroups[container]


def container_observations(containers: dict[str, str]) -> dict:
    result = {}
    owners = {}
    for container in set(_cgroups) - set(containers.values()):
        del _cgroups[container]
    for attempt in set(_cpu_samples) - set(containers):
        del _cpu_samples[attempt]
    for attempt, container in containers.items():
        try:
            resolved = _container_cgroup(container)
            if resolved is None:
                continue
            pid, group = resolved
            # Zero VRAM attribution is conservative: reservations remain fully unspent.
            result[attempt] = {"ram": int((group / "memory.current").read_text()), "vram": 0}
            usage = int(
                dict(line.split() for line in (group / "cpu.stat").read_text().splitlines())[
                    "usage_usec"
                ]
            )
            now = time.monotonic()
            previous = _cpu_samples.get(attempt)
            if previous and now > previous[0]:
                result[attempt]["cpu"] = max(0, (usage - previous[1]) / 1e6 / (now - previous[0]))
            _cpu_samples[attempt] = (now, usage)
            for process in [psutil.Process(pid), *psutil.Process(pid).children(recursive=True)]:
                owners[process.pid] = attempt
        except (
            OSError,
            ValueError,
            KeyError,
            StopIteration,
            psutil.Error,
            subprocess.SubprocessError,
        ):
            # An exited container's cgroup disappears; never reuse its old PID.
            _cgroups.pop(container, None)
            result[attempt] = {}
    try:
        output = subprocess.check_output(
            ["nvidia-smi", "--query-compute-apps=pid,used_memory", "--format=csv,noheader,nounits"],
            text=True,
            timeout=3,
        )
        for line in output.splitlines():
            pid, memory = line.split(",")
            owner = owners.get(int(pid.strip()))
            if owner:
                result[owner]["vram"] += int(memory.strip()) * 1024**2
    except (OSError, ValueError, subprocess.SubprocessError):
        pass
    return result
