"""Low-overhead host baseline sampler; never starts model work or changes services."""

import argparse
import json
import os
import subprocess
import time
import urllib.request
from pathlib import Path


def sample():
    memory = {}
    for line in Path("/proc/meminfo").read_text().splitlines():
        name, value = line.split(":", 1)
        memory[name] = int(value.split()[0]) * 1024
    result = {
        "time": time.time(),
        "memory": memory,
        "cpu": Path("/proc/stat").read_text().splitlines()[0],
        "load": Path("/proc/loadavg").read_text().split()[:3],
        "disk": {"free": os.statvfs("/").f_bavail * os.statvfs("/").f_frsize},
    }
    endpoint = os.environ.get("EXECUTION_BASELINE_HEALTH_URL")
    if endpoint:
        started = time.monotonic()
        try:
            with urllib.request.urlopen(endpoint, timeout=5) as response:
                result["web"] = {"status": response.status, "seconds": time.monotonic() - started}
        except (OSError, ValueError) as error:
            result["web"] = {"error": type(error).__name__, "seconds": time.monotonic() - started}
    for kind in ["cpu", "memory", "io"]:
        path = Path("/proc/pressure") / kind
        if path.exists():
            result[kind + "_pressure"] = path.read_text()
    try:
        result["gpu"] = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=uuid,memory.total,memory.free,utilization.gpu",
                "--format=csv,noheader,nounits",
            ],
            timeout=3,
            text=True,
        )
    except (OSError, subprocess.SubprocessError):
        pass
    try:
        result["containers"] = subprocess.check_output(
            ["docker", "stats", "--no-stream", "--format", "{{json .}}"], timeout=8, text=True
        )
    except (OSError, subprocess.SubprocessError):
        pass
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--duration", type=int, default=3600)
    parser.add_argument("--interval", type=float, default=10)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    with args.output.open("x") as handle:
        while time.monotonic() - started < args.duration:
            tick = time.monotonic()
            handle.write(json.dumps(sample()) + "\n")
            handle.flush()
            time.sleep(max(0, args.interval - (time.monotonic() - tick)))


if __name__ == "__main__":
    main()
