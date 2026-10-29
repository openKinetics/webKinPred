"""Host compatibility identity; image/model identities are pinned separately."""

import hashlib
import json
import os
import platform
import subprocess


def identity(gpu_uuid=""):
    data = {
        "kernel": platform.release(),
        "architecture": platform.machine(),
        "cpu_count": os.cpu_count(),
    }
    if gpu_uuid:
        data["gpu"] = subprocess.check_output(
            [
                "nvidia-smi",
                "-i",
                gpu_uuid,
                "--query-gpu=uuid,name,driver_version,memory.total",
                "--format=csv,noheader,nounits",
            ],
            text=True,
            timeout=3,
        ).strip()
    data["fingerprint"] = hashlib.sha256(json.dumps(data, sort_keys=True).encode()).hexdigest()
    return data
