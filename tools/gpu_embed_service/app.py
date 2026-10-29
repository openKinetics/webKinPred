from __future__ import annotations

import json
import os
import queue
import shlex
import shutil
import subprocess
import sys
import tempfile
import threading
import time
import uuid
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from fastapi import FastAPI, Header, HTTPException, Query
from pydantic import BaseModel, Field

# Import builtin step runner (same directory as this file).
_HERE = Path(__file__).parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))
import run_step as _run_step_module


class EmbedJobRequest(BaseModel):
    request_id: str | None = None
    method_key: str
    target: str
    profile: str | None = None
    step_work: dict[str, list[str]] = Field(default_factory=dict)
    seq_id_to_seq: dict[str, str] = Field(default_factory=dict)


class EmbedJobStatus(BaseModel):
    job_id: str
    status: str
    method_key: str
    target: str
    profile: str | None = None
    step_work: dict[str, list[str]] = Field(default_factory=dict)
    started_at: float | None = None
    finished_at: float | None = None
    worker_log_path: str | None = None
    error: str | None = None


@dataclass
class _JobState:
    request: EmbedJobRequest
    status: str = "queued"
    started_at: float | None = None
    finished_at: float | None = None
    worker_log_path: str | None = None
    error: str | None = None


app = FastAPI(title="webKinPred GPU Embedding Service", version="0.1.0")

_jobs: dict[str, _JobState] = {}
_jobs_lock = threading.Lock()
_job_queue: "queue.Queue[str]" = queue.Queue()
_worker_started = False
_worker_lock = threading.Lock()


def _parse_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _worker_log_dir() -> Path:
    configured = str(os.environ.get("GPU_EMBED_JOB_LOG_DIR", "")).strip()
    candidates: list[Path] = []
    if configured:
        candidates.append(Path(configured))
    candidates.append(Path("/tmp/webkinpred-gpu-embed/jobs"))
    for candidate in candidates:
        try:
            candidate.mkdir(parents=True, exist_ok=True)
            return candidate.resolve()
        except OSError:
            continue
    raise RuntimeError("Unable to create worker log directory.")


def _job_log_path(job_id: str) -> Path:
    return (_worker_log_dir() / f"{job_id}.log").resolve()


def _append_job_log(log_path: Path, message: str) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    ts = datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%S.%fZ")
    with log_path.open("a", encoding="utf-8") as handle:
        handle.write(f"{ts} {message}\n")


def _read_log_tail(log_path: Path, max_lines: int = 120) -> str:
    if max_lines <= 0:
        return ""
    try:
        text = log_path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return ""
    lines = text.splitlines()
    if not lines:
        return ""
    return "\n".join(lines[-max_lines:])


def _gpu_health_snapshot() -> dict[str, Any]:
    # Prefer NVIDIA's own CLI because it's present whenever drivers are installed.
    nvidia_smi = shutil.which("nvidia-smi") or "/usr/bin/nvidia-smi"
    try:
        proc = subprocess.run(
            [
                nvidia_smi,
                "--query-gpu=name,memory.free,memory.total",
                "--format=csv,noheader,nounits",
            ],
            check=False,
            text=True,
            capture_output=True,
        )
        if proc.returncode == 0:
            lines = [line.strip() for line in proc.stdout.splitlines() if line.strip()]
            if lines:
                # Example line: "NVIDIA RTX A4500, 17519, 20470"
                parts = [p.strip() for p in lines[0].split(",")]
                if len(parts) >= 3:
                    gpu_name = parts[0]
                    free_mb = _parse_float(parts[1])
                    total_mb = _parse_float(parts[2])
                    free_gb = (free_mb / 1024.0) if free_mb is not None else None
                    total_gb = (total_mb / 1024.0) if total_mb is not None else None
                    return {
                        "online": True,
                        "gpu_name": gpu_name or "unknown",
                        "free_vram_gb": round(free_gb, 2) if free_gb is not None else None,
                        "total_vram_gb": round(total_gb, 2) if total_gb is not None else None,
                    }
    except Exception:
        pass

    # Fallback if nvidia-smi is unavailable in path.
    fallback_name = str(os.environ.get("GPU_NAME", "unknown")).strip() or "unknown"
    return {
        "online": True,
        "gpu_name": fallback_name,
        "free_vram_gb": None,
        "total_vram_gb": None,
    }


def _token_is_valid(auth_header: str | None) -> bool:
    token = str(os.environ.get("GPU_EMBED_SERVICE_TOKEN", "")).strip()
    # Token must always be configured — enforced at startup via _on_startup.
    # An empty token never grants access; the service refuses to start without one.
    if not token:
        return False
    if not auth_header:
        return False
    parts = auth_header.split(" ", 1)
    return len(parts) == 2 and parts[0].lower() == "bearer" and parts[1].strip() == token


def _require_auth(authorization: str | None) -> None:
    if not _token_is_valid(authorization):
        raise HTTPException(status_code=401, detail="unauthorized")


def _run_command(
    cmd: list[str],
    *,
    env: dict[str, str],
    log_path: Path,
) -> None:
    display_cmd = " ".join(shlex.quote(part) for part in cmd)
    _append_job_log(log_path, f"$ {display_cmd}")

    proc = subprocess.Popen(
        cmd,
        shell=False,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        bufsize=1,
    )
    assert proc.stdout is not None
    assert proc.stderr is not None

    lock = threading.Lock()

    def _reader(stream, label: str) -> None:
        for line in iter(stream.readline, ""):
            line = line.rstrip("\n")
            if not line:
                continue
            with lock:
                _append_job_log(log_path, f"[{label}] {line}")
        stream.close()

    t_out = threading.Thread(target=_reader, args=(proc.stdout, "stdout"), daemon=True)
    t_err = threading.Thread(target=_reader, args=(proc.stderr, "stderr"), daemon=True)
    t_out.start()
    t_err.start()
    rc = proc.wait()
    t_out.join()
    t_err.join()

    if rc != 0:
        max_tail_lines_raw = str(os.environ.get("GPU_EMBED_ERROR_LOG_TAIL_LINES", "120")).strip()
        try:
            max_tail_lines = max(20, min(2000, int(max_tail_lines_raw)))
        except ValueError:
            max_tail_lines = 120
        tail = _read_log_tail(log_path, max_lines=max_tail_lines)
        raise RuntimeError(
            f"step command failed (rc={rc}); worker log: {log_path}\n"
            f"--- worker log tail ---\n{tail}"
        )


def _resolve_step_cmd_parts(
    cmd_template: str,
    *,
    step_key: str,
    seq_ids_arg: str,
    seq_count: int,
    seq_id_to_seq_file: str,
    job_id: str,
) -> list[str]:
    """
    Parse a step command template (from an env-var override) into an argument
    list safe for subprocess with shell=False.

    The template is first tokenised with shlex.split() so that quoted paths
    containing spaces survive intact.  Placeholder tokens are then replaced by
    exact string equality inside each individual argument, so user-supplied
    values (seq_ids_arg, job_id) are *never* interpreted by a shell — even if
    they contain semicolons, backticks, or other metacharacters.

    Supported placeholders: {step_key}, {seq_ids}, {seq_count},
                            {seq_id_to_seq_file}, {job_id}
    """
    substitutions = {
        "{step_key}": step_key,
        "{seq_ids}": seq_ids_arg,
        "{seq_count}": str(seq_count),
        "{seq_id_to_seq_file}": seq_id_to_seq_file,
        "{job_id}": job_id,
    }
    parts = shlex.split(cmd_template)
    resolved: list[str] = []
    for part in parts:
        for placeholder, value in substitutions.items():
            part = part.replace(placeholder, value)
        resolved.append(part)
    return resolved


def _execute_step(
    step_key: str,
    seq_ids: list[str],
    seq_id_to_seq: dict[str, str],
    *,
    job_id: str | None = None,
    log_path: Path,
) -> None:
    # If an override command is configured via env var, use it.
    # Available placeholders: {step_key}, {seq_ids}, {seq_count},
    #                         {seq_id_to_seq_file}, {job_id}
    env_key = f"GPU_EMBED_STEP_CMD_{step_key.upper()}"
    cmd_template = str(os.environ.get(env_key, "")).strip()

    if cmd_template:
        seq_ids_arg = ",".join(seq_ids)
        seq_count = len(seq_ids)
        step_seq_map = {sid: seq_id_to_seq[sid] for sid in seq_ids if sid in seq_id_to_seq}
        tmp_file: str | None = None
        cmd_env = dict(os.environ)
        cmd_env.setdefault("PYTHONUNBUFFERED", "1")
        try:
            with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as fh:
                json.dump(step_seq_map, fh)
                tmp_file = fh.name
            cmd_parts = _resolve_step_cmd_parts(
                cmd_template,
                step_key=step_key,
                seq_ids_arg=seq_ids_arg,
                seq_count=seq_count,
                seq_id_to_seq_file=tmp_file,
                job_id=job_id or "",
            )
            _run_command(cmd_parts, env=cmd_env, log_path=log_path)
        finally:
            if tmp_file and os.path.exists(tmp_file):
                os.unlink(tmp_file)
        return

    # No override: execute builtin run_step as a subprocess so worker logs
    # stream into the dedicated job log file.
    repo_root = Path(
        os.environ.get("GPU_REPO_ROOT")
        or os.environ.get("GPU_EMBED_REPO_ROOT")
        or str(_run_step_module._default_repo_root())
    ).resolve()
    media_path = Path(
        os.environ.get("KINFORM_MEDIA_PATH", "/mnt/webkinpred/media")
    ).resolve()
    tools_path = Path(
        os.environ.get("KINFORM_TOOLS_PATH", str(repo_root / "tools"))
    ).resolve()
    step_seq_map = {sid: seq_id_to_seq[sid] for sid in seq_ids if sid in seq_id_to_seq}
    cmd_env = dict(os.environ)
    cmd_env.setdefault("PYTHONUNBUFFERED", "1")

    seq_id_to_seq_file = ""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as fh:
        json.dump(step_seq_map, fh)
        seq_id_to_seq_file = fh.name

    try:
        _run_command(
            [
                sys.executable,
                str((_HERE / "run_step.py").resolve()),
                "--step",
                step_key,
                "--repo-root",
                str(repo_root),
                "--media-path",
                str(media_path),
                "--tools-path",
                str(tools_path),
                "--seq-id-to-seq-file",
                seq_id_to_seq_file,
                "--job-id",
                str(job_id or ""),
            ],
            env=cmd_env,
            log_path=log_path,
        )
    finally:
        if seq_id_to_seq_file and os.path.exists(seq_id_to_seq_file):
            os.unlink(seq_id_to_seq_file)


def _run_job(job_id: str) -> None:
    with _jobs_lock:
        state = _jobs[job_id]
        state.status = "running"
        state.started_at = time.time()
        log_path_str = state.worker_log_path

    if not log_path_str:
        raise RuntimeError(f"Missing worker log path for job {job_id}.")
    log_path = Path(log_path_str)
    _append_job_log(log_path, f"JOB_START job_id={job_id}")

    req = state.request

    try:
        for step_key, seq_ids in req.step_work.items():
            if not seq_ids:
                continue
            _append_job_log(
                log_path,
                f"STEP_START job_id={job_id} step={step_key} seq_count={len(seq_ids)}",
            )
            _execute_step(
                step_key,
                seq_ids,
                req.seq_id_to_seq,
                job_id=job_id,
                log_path=log_path,
            )
            _append_job_log(log_path, f"STEP_DONE job_id={job_id} step={step_key}")

        with _jobs_lock:
            state.status = "done"
            state.finished_at = time.time()
        _append_job_log(log_path, f"JOB_DONE job_id={job_id}")
    except Exception as exc:
        with _jobs_lock:
            state.status = "failed"
            state.error = str(exc)
            state.finished_at = time.time()
        _append_job_log(log_path, f"JOB_FAILED job_id={job_id} error={exc}")


def _worker_loop() -> None:
    while True:
        job_id = _job_queue.get()
        try:
            _run_job(job_id)
        finally:
            _job_queue.task_done()


def _ensure_worker() -> None:
    import durable_proxy
    if durable_proxy.enabled():
        return
    global _worker_started
    with _worker_lock:
        if _worker_started:
            return
        thread = threading.Thread(target=_worker_loop, name="gpu-embed-worker", daemon=True)
        thread.start()
        _worker_started = True


def _status_payload(job_id: str, state: _JobState) -> EmbedJobStatus:
    return EmbedJobStatus(
        job_id=job_id,
        status=state.status,
        method_key=state.request.method_key,
        target=state.request.target,
        profile=state.request.profile,
        step_work=state.request.step_work,
        started_at=state.started_at,
        finished_at=state.finished_at,
        worker_log_path=state.worker_log_path,
        error=state.error,
    )


@app.on_event("startup")
def _on_startup() -> None:
    token = str(os.environ.get("GPU_EMBED_SERVICE_TOKEN", "")).strip()
    if not token:
        raise RuntimeError(
            "GPU_EMBED_SERVICE_TOKEN is not set. "
            "The service cannot start without a configured auth token. "
            "Set the variable in gpu_service.env (see gpu_service.env.example)."
        )
    _ensure_worker()


@app.get("/health")
def health(authorization: str | None = Header(default=None)) -> dict:
    _require_auth(authorization)
    gpu = _gpu_health_snapshot()
    import durable_proxy
    if durable_proxy.enabled():
        return {**gpu, **durable_proxy.call("embed-health")}
    return {
        "online": bool(gpu.get("online", True)),
        "gpu_name": gpu.get("gpu_name"),
        "free_vram_gb": gpu.get("free_vram_gb"),
        "total_vram_gb": gpu.get("total_vram_gb"),
        "active_jobs": sum(1 for s in _jobs.values() if s.status == "running"),
        "queued_jobs": _job_queue.qsize(),
    }


@app.post("/embed/jobs")
def submit_embed_job(payload: EmbedJobRequest, authorization: str | None = Header(default=None)) -> dict:
    _require_auth(authorization)
    import durable_proxy
    if durable_proxy.enabled():
        return durable_proxy.call("embed-submit", payload.model_dump())
    _ensure_worker()

    if not payload.step_work:
        raise HTTPException(status_code=400, detail="step_work is required")

    job_id = uuid.uuid4().hex
    worker_log_path = _job_log_path(job_id)
    _append_job_log(worker_log_path, f"JOB_QUEUED job_id={job_id}")
    state = _JobState(request=payload, worker_log_path=str(worker_log_path))

    with _jobs_lock:
        _jobs[job_id] = state
    _job_queue.put(job_id)

    return {"job_id": job_id, "status": "queued"}


@app.get("/embed/jobs/{job_id}")
def get_embed_job(job_id: str, authorization: str | None = Header(default=None)) -> dict:
    import durable_proxy
    if durable_proxy.enabled():
        _require_auth(authorization)
        return durable_proxy.call("embed-status", {"job_id": job_id})
    _require_auth(authorization)
    with _jobs_lock:
        state = _jobs.get(job_id)
        if state is None:
            raise HTTPException(status_code=404, detail="job not found")
        payload = _status_payload(job_id, state)
    return payload.model_dump()


@app.get("/embed/jobs/{job_id}/logs")
def get_embed_job_logs(
    job_id: str,
    tail: int = Query(default=200, ge=1, le=5000),
    authorization: str | None = Header(default=None),
) -> dict:
    _require_auth(authorization)
    import durable_proxy
    if durable_proxy.enabled():
        result = durable_proxy.call("embed-logs", {"job_id": job_id})
        return {"job_id": job_id, "tail_lines": tail, "log_tail": "\n".join(result["lines"][-int(tail):])}
    with _jobs_lock:
        state = _jobs.get(job_id)
        if state is None:
            raise HTTPException(status_code=404, detail="job not found")
        log_path_str = state.worker_log_path

    if not log_path_str:
        raise HTTPException(status_code=404, detail="worker log not found")

    log_path = Path(log_path_str)
    if not log_path.exists():
        raise HTTPException(status_code=404, detail="worker log not found")

    return {
        "job_id": job_id,
        "tail_lines": tail,
        "log_tail": _read_log_tail(log_path, max_lines=tail),
    }
