from __future__ import annotations

import errno


class RecoverableExecutionError(RuntimeError):
    category = "resource"


class MissingEmbedding(RecoverableExecutionError):
    category = "missing_embedding"


def classify(error: BaseException | None = None, *, exit_code=None, oom_killed=False) -> str:
    if oom_killed or isinstance(error, MemoryError):
        return "oom"
    if isinstance(error, RecoverableExecutionError):
        return error.category
    message = str(error).lower()
    if "missing_embedding" in message:
        return "missing_embedding"
    if any(
        value in message
        for value in (
            "gpu_unavailable",
            "no cuda gpus are available",
            "cuda driver initialization failed",
            "cuda-capable device(s) is/are busy or unavailable",
            "cuda driver version is insufficient",
        )
    ):
        return "gpu_unavailable"
    if any(
        value in message
        for value in (
            "out of memory",
            "cuda error: memory",
            "can't allocate memory",
            "cannot allocate memory",
            "std::bad_alloc",
            "cublas_status_alloc_failed",
            "cudnn_status_alloc_failed",
        )
    ):
        return "oom"
    if any(value in message for value in ("no space left on device", "disk quota exceeded")):
        return "disk"
    if any(
        value in message
        for value in (
            "resource temporarily unavailable",
            "can't start new thread",
            "cannot start new thread",
            "pthread_create failed",
            "pthread_create() failed",
        )
    ):
        return "cpu"
    if isinstance(error, (TimeoutError, ConnectionError)):
        return "network"
    if error is not None and error.__cause__ is not None:
        cause_category = classify(error.__cause__)
        if cause_category != "unknown":
            return cause_category
    if error is not None and hasattr(error, "returncode"):
        import subprocess

        if isinstance(error, subprocess.CalledProcessError):
            evidence = str(error.output or "") + str(error.stderr or "")
            category = classify(RuntimeError(evidence), exit_code=error.returncode)
            if category != "unknown":
                return category
    if isinstance(error, OSError):
        return {
            errno.ENOMEM: "oom",
            errno.ENOSPC: "disk",
            errno.EDQUOT: "disk",
            errno.EAGAIN: "cpu",
            errno.ECONNRESET: "network",
            errno.ETIMEDOUT: "network",
            errno.ESTALE: "storage",
            errno.EIO: "storage",
            errno.EHOSTUNREACH: "network",
            errno.ENETUNREACH: "network",
            errno.EPIPE: "network",
        }.get(error.errno, "unknown")
    if exit_code in (-9, 137, -15, 143):
        return "interrupted"
    return "unknown"
