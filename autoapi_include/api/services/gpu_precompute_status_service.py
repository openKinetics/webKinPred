"""
Persist per-job GPU precompute outcomes in Redis for API status reporting.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone

from api.services.progress_service import redis_conn

_KEY_PREFIX = "job_gpu_precompute:"
_TTL_SECONDS = 7200


def _redis_key(job_public_id: str) -> str:
    return f"{_KEY_PREFIX}{job_public_id}"


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def clear_gpu_precompute_status(job_public_id: str) -> None:
    redis_conn.delete(_redis_key(job_public_id))


def get_gpu_precompute_status(job_public_id: str) -> dict | None:
    raw = redis_conn.get(_redis_key(job_public_id))
    if not raw:
        return None
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        return None
    if not isinstance(data, dict):
        return None
    return data


def record_gpu_precompute_result(
    *,
    job_public_id: str,
    method_key: str,
    target: str,
    attempted: bool,
    used_gpu: bool,
    completed: bool,
    failed: bool,
    reason: str | None,
) -> None:
    existing = get_gpu_precompute_status(job_public_id) or {}
    events = existing.get("events")
    if not isinstance(events, list):
        events = []

    event = {
        "methodKey": method_key,
        "method_key": method_key,
        "target": target,
        "attempted": bool(attempted),
        "usedGpu": bool(used_gpu),
        "used_gpu": bool(used_gpu),
        "completed": bool(completed),
        "failed": bool(failed),
        "reason": reason,
        "updatedAt": _now_iso(),
    }
    events.append(event)
    # Keep history bounded.
    events = events[-20:]

    payload = {
        "methodKey": method_key,
        "method_key": method_key,
        "target": target,
        "attempted": bool(existing.get("attempted", False)) or bool(attempted),
        "usedGpu": bool(existing.get("usedGpu", False)) or bool(used_gpu),
        "used_gpu": bool(existing.get("used_gpu", False)) or bool(used_gpu),
        # Keep "completed" and "reason" as the latest result while preserving
        # aggregate attempted/used/failed signals across multiple method stages.
        "completed": bool(completed),
        "failed": bool(existing.get("failed", False)) or bool(failed),
        "reason": reason,
        "updatedAt": _now_iso(),
        "events": events,
    }

    redis_conn.set(_redis_key(job_public_id), json.dumps(payload), ex=_TTL_SECONDS)
