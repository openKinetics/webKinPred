import subprocess
import os

try:
    from webKinPred.config_docker import (
        TARGET_DBS,
    )
except ImportError:
    from webKinPred.config_local import (
        TARGET_DBS,
    )
from api.services.progress_service import (
    push_line,
    is_cancelled,
    get_pid_key,
    redis_conn,
)
from api.utils.log_sanitiser import sanitise_log_line

TMP_DIR = os.environ.get("MMSEQS_TMP_DIR", "/tmp")
os.makedirs(TMP_DIR, exist_ok=True)

def run_and_stream(
    cmd, session_id: str, cwd: str | None = None, env: dict | None = None, fail_ok=False
):
    echoed = "$ " + " ".join(cmd)
    san_line = sanitise_log_line(echoed, TARGET_DBS)
    push_line(session_id, san_line)

    pid_key = get_pid_key(session_id)
    proc = None
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
            print(f"[cleanup] Deleting PID key for session {session_id}")
            redis_conn.delete(pid_key)

    if is_cancelled(session_id):
        print(f"[run_and_stream] Step for session {session_id} was cancelled.")
        return
    if rc != 0 and not fail_ok:
        push_line(session_id, f"[ERROR] Command failed with exit code {rc}")
        raise subprocess.CalledProcessError(rc, cmd)
    elif rc != 0 and fail_ok:
        push_line(
            session_id, f"[WARN] Command returned non-zero exit code {rc} (continuing)"
        )
    else:
        push_line(session_id, "[OK] Completed")