from __future__ import annotations

import json
import logging
import os
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass

from api.services.embedding_plan_service import build_embedding_plan, gpu_step_work
from api.services.embedding_progress_service import start_embedding_tracking
from api.services.gpu_precompute_status_service import record_gpu_precompute_result


_DEFAULT_HEALTH_TTL = 10
_DEFAULT_JOB_TIMEOUT = 21600
_DEFAULT_POLL_INTERVAL = 1.0
_DEFAULT_POLL_LOG_INTERVAL = 120
_DEFAULT_HTTP_TIMEOUT = 5.0

_status_cache_lock = threading.Lock()
_status_cache_payload: dict | None = None
_status_cache_ts = 0.0
_log = logging.getLogger(__name__)


@dataclass(frozen=True)
class GpuPrecomputeResult:
    attempted: bool
    used_gpu: bool
    completed: bool
    failed: bool
    reason: str | None = None


def _env_int(name: str, default: int) -> int:
    raw = str(os.environ.get(name, "")).strip()
    if not raw:
        return default
    try:
        value = int(raw)
    except ValueError:
        return default
    return value if value > 0 else default


def _env_bool(name: str, default: bool = False) -> bool:
    raw = str(os.environ.get(name, "")).strip().lower()
    if not raw:
        return default
    return raw in {"1", "true", "yes", "on"}


def _base_url() -> str:
    return str(os.environ.get("GPU_EMBED_SERVICE_URL", "")).strip().rstrip("/")


def _auth_header() -> dict[str, str]:
    token = str(os.environ.get("GPU_EMBED_SERVICE_TOKEN", "")).strip()
    if not token:
        return {}
    return {"Authorization": f"Bearer {token}"}


def _http_json(method: str, url: str, payload: dict | None = None, timeout: float = _DEFAULT_HTTP_TIMEOUT) -> dict:
    body: bytes | None = None
    headers = {"Accept": "application/json", **_auth_header()}
    if payload is not None:
        headers["Content-Type"] = "application/json"
        body = json.dumps(payload).encode("utf-8")

    req = urllib.request.Request(url=url, method=method.upper(), data=body, headers=headers)
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        raw = resp.read().decode("utf-8")

    data = json.loads(raw) if raw else {}
    if not isinstance(data, dict):
        raise RuntimeError(f"Expected JSON object from {url}, got {type(data).__name__}")
    return data


def get_gpu_status(*, force_refresh: bool = False) -> dict:
    global _status_cache_payload, _status_cache_ts

    base = _base_url()
    if not base:
        return {
            "configured": False,
            "online": False,
            "mode": "cpu",
            "reason": "not_configured",
        }

    ttl = _env_int("GPU_EMBED_HEALTH_TTL_SECONDS", _DEFAULT_HEALTH_TTL)
    now = time.time()

    if not force_refresh:
        with _status_cache_lock:
            if _status_cache_payload is not None and (now - _status_cache_ts) < ttl:
                return dict(_status_cache_payload)

    url = f"{base}/health"
    try:
        remote = _http_json("GET", url)
        is_online = bool(remote.get("online", True))
        payload = {
            "configured": True,
            "online": is_online,
            "mode": "gpu" if is_online else "cpu",
            "reason": None if is_online else "remote_offline",
            "gpu_name": remote.get("gpu_name"),
            "free_vram_gb": remote.get("free_vram_gb"),
            "total_vram_gb": remote.get("total_vram_gb"),
            "active_jobs": remote.get("active_jobs"),
            "queued_jobs": remote.get("queued_jobs"),
        }
    except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError, json.JSONDecodeError) as exc:
        payload = {
            "configured": True,
            "online": False,
            "mode": "cpu",
            "reason": f"unreachable: {exc}",
        }

    with _status_cache_lock:
        _status_cache_payload = dict(payload)
        _status_cache_ts = now

    return dict(payload)


def _poll_job(
    base: str,
    gpu_job_id: str,
    *,
    job_public_id: str,
    method_key: str,
    target: str,
) -> dict:
    timeout_secs = _env_int("GPU_EMBED_JOB_TIMEOUT_SECONDS", _DEFAULT_JOB_TIMEOUT)
    log_interval = _env_int("GPU_EMBED_POLL_LOG_INTERVAL_SECONDS", _DEFAULT_POLL_LOG_INTERVAL)
    started_at = time.monotonic()
    deadline = started_at + float(timeout_secs)
    next_log_at = started_at + float(log_interval)
    last_state: str | None = None
    poll_error_count = 0
    last_poll_error: str | None = None

    while True:
        try:
            status = _http_json("GET", f"{base}/embed/jobs/{urllib.parse.quote(gpu_job_id)}")
        except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError, json.JSONDecodeError) as exc:
            poll_error_count += 1
            last_poll_error = str(exc)
            now = time.monotonic()
            if now >= deadline:
                elapsed = int(now - started_at)
                _log.warning(
                    "GPU precompute polling timed out after repeated poll errors for job %s (%s/%s): gpu_job_id=%s elapsed=%ss timeout=%ss errors=%s last_error=%s",
                    job_public_id,
                    method_key,
                    target,
                    gpu_job_id,
                    elapsed,
                    timeout_secs,
                    poll_error_count,
                    last_poll_error,
                )
                return {"status": "timeout", "detail": {"last_error": last_poll_error}}

            if now >= next_log_at:
                elapsed = int(now - started_at)
                _log.warning(
                    "Transient GPU poll error for job %s (%s/%s): gpu_job_id=%s elapsed=%ss errors=%s last_error=%s",
                    job_public_id,
                    method_key,
                    target,
                    gpu_job_id,
                    elapsed,
                    poll_error_count,
                    last_poll_error,
                )
                next_log_at = now + float(log_interval)
            time.sleep(_DEFAULT_POLL_INTERVAL)
            continue

        state = str(status.get("status", "")).strip().lower()
        if state != last_state:
            last_state = state
            _log.info(
                "GPU precompute status update for job %s (%s/%s): gpu_job_id=%s state=%s",
                job_public_id,
                method_key,
                target,
                gpu_job_id,
                state or "unknown",
            )

        if state in {"done", "completed"}:
            return {"status": "done", "detail": status}
        if state in {"failed", "error"}:
            return {"status": "failed", "detail": status}

        now = time.monotonic()
        if now >= deadline:
            elapsed = int(now - started_at)
            _log.warning(
                "GPU precompute timed out for job %s (%s/%s): gpu_job_id=%s state=%s elapsed=%ss timeout=%ss errors=%s",
                job_public_id,
                method_key,
                target,
                gpu_job_id,
                state or "unknown",
                elapsed,
                timeout_secs,
                poll_error_count,
            )
            return {"status": "timeout", "detail": status}

        if now >= next_log_at:
            elapsed = int(now - started_at)
            _log.info(
                "Waiting for GPU precompute job %s (%s/%s): gpu_job_id=%s state=%s elapsed=%ss",
                job_public_id,
                method_key,
                target,
                gpu_job_id,
                state or "unknown",
                elapsed,
            )
            next_log_at = now + float(log_interval)

        time.sleep(_DEFAULT_POLL_INTERVAL)


def run_gpu_precompute_if_available(
    *,
    job_public_id: str,
    method_key: str,
    target: str,
    valid_sequences: list[str],
    env: dict | None,
    disabled: bool = False,
) -> GpuPrecomputeResult:
    fail_closed = _env_bool("GPU_EMBED_FAIL_CLOSED", default=False)

    def _record(result: GpuPrecomputeResult) -> None:
        try:
            record_gpu_precompute_result(
                job_public_id=job_public_id,
                method_key=method_key,
                target=target,
                attempted=result.attempted,
                used_gpu=result.used_gpu,
                completed=result.completed,
                failed=result.failed,
                reason=result.reason,
            )
        except Exception:
            # Telemetry is best-effort; never break prediction flow/tests.
            pass

    def _finish(result: GpuPrecomputeResult) -> GpuPrecomputeResult:
        _record(result)
        if fail_closed and not result.completed:
            raise RuntimeError(
                f"GPU precompute required but incomplete for {method_key}/{target}: "
                f"{result.reason or 'unknown_reason'}"
            )
        return result

    if disabled:
        _log.info(
            "Skipping GPU precompute for job %s (%s/%s): disabled by request",
            job_public_id, method_key, target,
        )
        disabled_result = GpuPrecomputeResult(
            False, False, False, False, "disabled_by_request"
        )
        _record(disabled_result)
        return disabled_result

    if not valid_sequences:
        _log.info(
            "Skipping GPU precompute for job %s (%s/%s): no valid sequences",
            job_public_id, method_key, target,
        )
        return _finish(GpuPrecomputeResult(False, False, False, False, "no_valid_sequences"))

    env_data = env or {}
    try:
        plan = build_embedding_plan(
            method_key=method_key,
            target=target,
            sequences=valid_sequences,
            env=env_data,
        )
    except Exception as exc:
        _log.exception(
            "Embedding plan failed for job %s (%s/%s): %s",
            job_public_id, method_key, target, exc,
        )
        return _finish(GpuPrecomputeResult(False, False, False, False, f"plan_failed: {exc}"))

    if plan.need_computation <= 0:
        _log.info(
            "Skipping GPU precompute for job %s (%s/%s): cache already complete",
            job_public_id, method_key, target,
        )
        return _finish(GpuPrecomputeResult(False, False, True, False, "cache_complete"))
    if not plan.gpu_supported:
        # Keep unsupported methods fail-open even in strict mode.
        _log.info(
            "Skipping GPU precompute for job %s (%s/%s): unsupported (%s)",
            job_public_id, method_key, target, plan.gpu_reason or "unsupported",
        )
        unsupported = GpuPrecomputeResult(
            False, False, False, False, plan.gpu_reason or "unsupported"
        )
        _record(unsupported)
        return unsupported

    base = _base_url()
    if not base:
        _log.info(
            "Skipping GPU precompute for job %s (%s/%s): service URL not configured",
            job_public_id, method_key, target,
        )
        return _finish(GpuPrecomputeResult(False, False, False, False, "not_configured"))

    status = get_gpu_status()
    if not status.get("online"):
        _log.warning(
            "GPU offline for job %s (%s/%s): %s. Falling back to local compute.",
            job_public_id, method_key, target, status.get("reason") or "offline",
        )
        return _finish(
            GpuPrecomputeResult(True, False, False, True, status.get("reason") or "offline")
        )

    step_work = gpu_step_work(plan)
    if not step_work:
        _log.info(
            "GPU precompute not needed for job %s (%s/%s): no missing GPU steps",
            job_public_id, method_key, target,
        )
        return _finish(GpuPrecomputeResult(True, False, True, False, "no_missing_gpu_steps"))

    missing_seq_ids = sorted({seq_id for seq_ids in step_work.values() for seq_id in seq_ids})
    seq_id_to_seq = {seq_id: plan.seq_id_to_seq[seq_id] for seq_id in missing_seq_ids if seq_id in plan.seq_id_to_seq}
    _log.info(
        "Submitting GPU precompute for job %s (%s/%s): %d missing files, %d step(s), %d sequence(s)",
        job_public_id,
        method_key,
        target,
        plan.need_computation,
        len(step_work),
        len(missing_seq_ids),
    )

    # Start tracker before GPU writes begin so inotify observes remote file creation.
    start_embedding_tracking(
        job_public_id=job_public_id,
        method_key=method_key,
        target=target,
        valid_sequences=valid_sequences,
        env=env_data,
    )

    payload = {
        "method_key": method_key,
        "target": target,
        "profile": plan.profile,
        "step_work": step_work,
        "seq_id_to_seq": seq_id_to_seq,
    }

    try:
        response = _http_json("POST", f"{base}/embed/jobs", payload=payload)
        gpu_job_id = str(response.get("job_id", "")).strip()
        if not gpu_job_id:
            _log.error(
                "GPU precompute submission failed for job %s (%s/%s): missing gpu job id in response",
                job_public_id, method_key, target,
            )
            return _finish(GpuPrecomputeResult(True, True, False, True, "missing_job_id"))

        _log.info(
            "Submitted job to GPU for job %s (%s/%s): gpu_job_id=%s",
            job_public_id, method_key, target, gpu_job_id,
        )
        _log.info(
            "Waiting for GPU job to complete for job %s (%s/%s): gpu_job_id=%s",
            job_public_id, method_key, target, gpu_job_id,
        )
        polled = _poll_job(
            base,
            gpu_job_id,
            job_public_id=job_public_id,
            method_key=method_key,
            target=target,
        )
        if polled["status"] in {"done", "timeout"}:
            # Remote service may report "done" while writing no cache files
            # (for example during smoke/no-op command wiring). Re-check local
            # cache state before treating GPU precompute as completed.
            # We also run this check on timeout to avoid false failures caused
            # by transient polling/network errors after remote completion.
            try:
                post_plan = build_embedding_plan(
                    method_key=method_key,
                    target=target,
                    sequences=valid_sequences,
                    env=env_data,
                )
            except Exception as exc:
                _log.exception(
                    "GPU post-check failed for job %s (%s/%s): %s",
                    job_public_id, method_key, target, exc,
                )
                return _finish(
                    GpuPrecomputeResult(True, True, False, True, f"postcheck_failed: {exc}")
                )

            if post_plan.need_computation > 0:
                timeout_suffix = (
                    " after poll timeout"
                    if polled["status"] == "timeout"
                    else ""
                )
                _log.warning(
                    "GPU job%s but outputs are incomplete for job %s (%s/%s): still missing %d file(s). Falling back to local compute.",
                    timeout_suffix,
                    job_public_id,
                    method_key,
                    target,
                    post_plan.need_computation,
                )
                return _finish(
                    GpuPrecomputeResult(
                        True,
                        True,
                        False,
                        True,
                        (
                            f"timeout_incomplete_remote_outputs:{post_plan.need_computation}"
                            if polled["status"] == "timeout"
                            else f"incomplete_remote_outputs:{post_plan.need_computation}"
                        ),
                    )
                )

            if polled["status"] == "timeout":
                _log.info(
                    "GPU outputs complete for job %s (%s/%s) despite poll timeout; proceeding with prediction.",
                    job_public_id,
                    method_key,
                    target,
                )
                return _finish(
                    GpuPrecomputeResult(True, True, True, False, "done_after_poll_timeout")
                )

            _log.info(
                "GPU finished for job %s (%s/%s), proceeding with prediction now.",
                job_public_id, method_key, target,
            )
            return _finish(GpuPrecomputeResult(True, True, True, False, "done"))
        _log.warning(
            "GPU precompute reported failure for job %s (%s/%s): %s. Falling back to local compute.",
            job_public_id, method_key, target, polled["status"],
        )
        return _finish(GpuPrecomputeResult(True, True, False, True, polled["status"]))
    except RuntimeError:
        # Preserve strict-mode failures as-is (do not wrap them again).
        raise
    except Exception as exc:
        _log.exception(
            "GPU precompute request failed for job %s (%s/%s): %s",
            job_public_id, method_key, target, exc,
        )
        return _finish(GpuPrecomputeResult(True, True, False, True, f"gpu_request_failed: {exc}"))
