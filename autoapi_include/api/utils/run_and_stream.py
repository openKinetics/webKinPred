import logging
import os
import subprocess

from api.services.progress_service import (
    get_pid_key,
    is_cancelled,
    push_line,
    redis_conn,
)
from api.utils.log_sanitiser import sanitise_log_line
from api.utils.similarity_config import TARGET_DBS

TMP_DIR = os.environ.get("MMSEQS_TMP_DIR", "/tmp")
os.makedirs(TMP_DIR, exist_ok=True)
_log = logging.getLogger(__name__)


def run_and_stream(
    cmd, session_id: str, cwd: str | None = None, env: dict | None = None, fail_ok=False
):
    echoed = "$ " + " ".join(cmd)
    san_line = sanitise_log_line(echoed, TARGET_DBS)
    push_line(session_id, san_line)

    pid_key = get_pid_key(session_id)
    proc = None
    rc: int | None = None
    try:
        proc = subprocess.Popen(
            cmd,
            cwd=cwd,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            universal_newlines=True,
            preexec_fn=os.setsid,
        )
        # Store the PID in Redis with a 15-minute expiry as a safety net
        redis_conn.set(pid_key, proc.pid, ex=900)
        if proc.stdout is not None:
            for raw in proc.stdout:
                raw = raw.rstrip("\n")
                # The is_cancelled check is now a secondary guard
                if is_cancelled(session_id):
                    break
                safe = sanitise_log_line(raw, TARGET_DBS)
                push_line(session_id, safe)
        rc = proc.wait()
    finally:
        if proc:
            _log.debug(
                "Deleting progress session PID key",
                extra={
                    "event": "progress_stream.pid_key_deleted",
                    "session_id": session_id,
                    "pid": proc.pid,
                },
            )
            redis_conn.delete(pid_key)

    if is_cancelled(session_id):
        _log.info(
            "Progress stream command cancelled",
            extra={"event": "progress_stream.command_cancelled", "session_id": session_id},
        )
        return
    if rc is None:
        raise RuntimeError("Command did not produce an exit code.")

    if rc != 0 and not fail_ok:
        push_line(session_id, f"[ERROR] Command failed with exit code {rc}")
        raise subprocess.CalledProcessError(rc, cmd)
    elif rc != 0 and fail_ok:
        push_line(session_id, f"[WARN] Command returned non-zero exit code {rc} (continuing)")
    else:
        push_line(session_id, "[OK] Completed")
