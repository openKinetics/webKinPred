"""Optional framework high-water marks, importable in legacy ML interpreters."""

import atexit
import json
import os
import time
from pathlib import Path

_installed = False


def install(torch):
    global _installed
    if _installed or not os.environ.get("WEBKINPRED_PROFILING"):
        return
    _installed = True
    started = time.monotonic()
    versions = {"torch": torch.__version__, "cuda": torch.version.cuda}

    def report():
        data = {
            "pid": os.getpid(),
            "versions": versions,
            "elapsed_seconds": time.monotonic() - started,
        }
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            data.update(
                allocated_peak=torch.cuda.max_memory_allocated(),
                reserved_peak=torch.cuda.max_memory_reserved(),
            )
        directory = Path("/work/framework-metrics")
        directory.mkdir(exist_ok=True)
        (directory / (str(os.getpid()) + ".json")).write_text(json.dumps(data))

    atexit.register(report)
