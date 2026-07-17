"""
Embedding progress tracking service.

Tracks per-job embedding cache progress in Redis without modifying the model
embedding scripts themselves. The tracker is:

- event-driven on Linux via inotify
- resilient via periodic reconciliation sweeps
- portable via polling fallback when inotify is unavailable
"""

from __future__ import annotations

import ctypes
import ctypes.util
import json
import logging
import os
import select
import struct
import sys
import threading
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

from api.services.embedding_plan_service import (
    expected_paths_by_seq as planner_expected_paths_by_seq,
    method_env_keys as planner_method_env_keys,
    missing_paths_by_seq_from_snapshot as planner_missing_paths_by_seq_from_snapshot,
    normalise_sequences_for_method as planner_normalise_sequences_for_method,
    resolve_media_and_tools as planner_resolve_media_and_tools,
    resolve_seq_ids_via_cli as planner_resolve_seq_ids_via_cli,
)
from tools.gpu_embed_service.cache_io import resolve_missing_ids
from api.services.progress_service import redis_conn

_KEY_PREFIX = "job_embedding_progress:"
_TTL_SECONDS = 7200
_RECONCILE_SECS = 10.0
_FALLBACK_POLL_SECS = 3.0
_REDIS_WRITE_MIN_INTERVAL_SECS = 0.25

_TRACKERS: dict[str, "_EmbeddingTracker"] = {}
_TRACKERS_LOCK = threading.Lock()
_log = logging.getLogger(__name__)


def _redis_key(job_public_id: str) -> str:
    return f"{_KEY_PREFIX}{job_public_id}"


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _sync_stage_embedding_progress(
    *,
    job_public_id: str,
    target: str,
    method_key: str,
    enabled: bool,
    state: str,
    total: int,
    cached_already: int,
    need_computation: int,
    computed: int,
    remaining: int,
) -> None:
    try:
        from api.services.job_progress_service import set_stage_embedding_progress

        set_stage_embedding_progress(
            job_public_id=job_public_id,
            target=target,
            method_key=method_key,
            enabled=enabled,
            state=state,
            total=total,
            cached_already=cached_already,
            need_computation=need_computation,
            computed=computed,
            remaining=remaining,
        )
    except Exception:
        # Tracking persistence must never block prediction execution.
        return


def _sync_stage_embedding_state(
    *,
    job_public_id: str,
    target: str,
    method_key: str,
    state: str,
) -> None:
    try:
        from api.services.job_progress_service import set_stage_embedding_state

        set_stage_embedding_state(
            job_public_id=job_public_id,
            target=target,
            method_key=method_key,
            state=state,
        )
    except Exception:
        # Tracking persistence must never block prediction execution.
        return


def clear_embedding_progress(job_public_id: str) -> None:
    redis_conn.delete(_redis_key(job_public_id))


def get_embedding_progress(job_public_id: str) -> dict | None:
    raw = redis_conn.get(_redis_key(job_public_id))
    if not raw:
        return None
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        return None

    for key in (
        "total",
        "cached_already",
        "need_computation",
        "computed",
        "remaining",
    ):
        if key in data:
            try:
                data[key] = int(data[key])
            except (TypeError, ValueError):
                data[key] = 0

    # convenience camelCase aliases for web UI compatibility
    if "method_key" in data and "methodKey" not in data:
        data["methodKey"] = data["method_key"]
    if "cached_already" in data and "cachedAlready" not in data:
        data["cachedAlready"] = data["cached_already"]
    if "need_computation" in data and "needComputation" not in data:
        data["needComputation"] = data["need_computation"]
    return data


@dataclass
class _PreparedPlan:
    method_key: str
    target: str
    total: int
    cached_already: int
    need_computation: int
    missing_paths_by_seq: dict[str, set[str]]
    path_to_seqs: dict[str, set[str]]
    watch_dirs: set[Path]


def start_embedding_tracking(
    *,
    job_public_id: str,
    method_key: str,
    target: str,
    valid_sequences: list[str] | None,
    env: dict | None = None,
) -> bool:
    """
    Start embedding progress tracking for a job/method run.

    Returns True when tracking was started or initialized (including immediate
    done state), False when tracking is not applicable or setup failed.
    """
    if not valid_sequences:
        clear_embedding_progress(job_public_id)
        _sync_stage_embedding_state(
            job_public_id=job_public_id,
            target=target,
            method_key=method_key,
            state="not_required",
        )
        return False

    if method_key == "DLKcat":
        clear_embedding_progress(job_public_id)
        _sync_stage_embedding_state(
            job_public_id=job_public_id,
            target=target,
            method_key=method_key,
            state="not_required",
        )
        return False

    try:
        plan = _prepare_plan(
            method_key=method_key,
            target=target,
            sequences=valid_sequences,
            env=env or {},
        )
    except Exception as exc:  # fail-open: prediction must continue
        _log.warning(
            "Embedding progress setup skipped",
            extra={
                "event": "embedding_progress.setup_skipped",
                "job_public_id": job_public_id,
                "method_key": method_key,
                "target": target,
                "exception_type": type(exc).__name__,
            },
            exc_info=True,
        )
        clear_embedding_progress(job_public_id)
        _sync_stage_embedding_state(
            job_public_id=job_public_id,
            target=target,
            method_key=method_key,
            state="error",
        )
        return False

    with _TRACKERS_LOCK:
        existing = _TRACKERS.get(job_public_id)

        # If the same stage is already being actively tracked (e.g. GPU precompute
        # started a tracker and the prediction subprocess calls us again for the
        # same method/target), reuse the running tracker rather than restarting.
        # Restarting would call _prepare_plan() on a filesystem that already has
        # some GPU-written files, inflating cached_already and resetting computed.
        if (
            existing is not None
            and existing.method_key == method_key
            and existing.target == target
            and not existing._stop_event.is_set()
        ):
            return True

        # Different stage starting, or previous tracker already stopped — replace it.
        if existing is not None:
            _TRACKERS.pop(job_public_id, None)
            existing.stop(final_state="done")

        tracker = _EmbeddingTracker(job_public_id=job_public_id, plan=plan)
        _TRACKERS[job_public_id] = tracker

    tracker.start()
    return True


def stop_embedding_tracking(job_public_id: str, final_state: str = "done") -> None:
    with _TRACKERS_LOCK:
        tracker = _TRACKERS.pop(job_public_id, None)
    if tracker is not None:
        tracker.stop(final_state=final_state)


def _method_env_keys(method_key: str) -> tuple[str | None, str | None]:
    return planner_method_env_keys(method_key)


def _resolve_media_and_tools(method_key: str, env: dict) -> tuple[Path, Path]:
    return planner_resolve_media_and_tools(method_key, env)


def _normalise_sequences_for_method(method_key: str, sequences: Iterable[str]) -> list[str]:
    return planner_normalise_sequences_for_method(method_key, sequences)


def _resolve_seq_ids_via_cli(sequences: list[str], tools_path: Path, media_path: Path) -> list[str]:
    return planner_resolve_seq_ids_via_cli(sequences, tools_path, media_path)


def _catpred_parameter(target: str) -> str | None:
    if target == "kcat":
        return "kcat"
    if target == "Km":
        return "km"
    if target.lower() == "km":
        return "km"
    return None


def _catpred_checkpoint_key(model_pt_path: Path) -> str:
    parent = model_pt_path.parent.name
    grandparent = model_pt_path.parent.parent.name
    return f"{grandparent}__{parent}" if grandparent else parent


def _discover_catpred_checkpoint_keys(checkpoint_root: Path, parameter: str) -> list[str]:
    parameter_root = checkpoint_root / parameter
    if not parameter_root.exists():
        return []

    keys: list[str] = []
    seen: set[str] = set()
    for model_file in sorted(parameter_root.rglob("model.pt")):
        key = _catpred_checkpoint_key(model_file)
        if key in seen:
            continue
        seen.add(key)
        keys.append(key)
    return keys


def _expected_paths_by_seq(
    *,
    method_key: str,
    target: str,
    seq_ids: list[str],
    media_path: Path,
    env: dict,
) -> dict[str, set[str]]:
    return planner_expected_paths_by_seq(
        method_key=method_key,
        target=target,
        seq_ids=seq_ids,
        media_path=media_path,
        env=env,
    )


def _prepare_plan(
    *,
    method_key: str,
    target: str,
    sequences: list[str],
    env: dict,
) -> _PreparedPlan:
    media_path, tools_path = _resolve_media_and_tools(method_key, env)
    unique_sequences = _normalise_sequences_for_method(method_key, sequences)
    if not unique_sequences:
        raise RuntimeError("No valid sequences for embedding progress tracking.")

    seq_ids = _resolve_seq_ids_via_cli(unique_sequences, tools_path, media_path)
    expected = _expected_paths_by_seq(
        method_key=method_key,
        target=target,
        seq_ids=seq_ids,
        media_path=media_path,
        env=env,
    )
    if not expected:
        raise RuntimeError("No embedding cache profile for this method/target.")

    missing_paths_by_seq = planner_missing_paths_by_seq_from_snapshot(expected)
    path_to_seqs: dict[str, set[str]] = {}
    watch_dirs: set[Path] = set()

    cached_already = 0
    need_computation = 0

    for seq_id, paths in expected.items():
        missing = missing_paths_by_seq.get(seq_id, set())
        cached_already += len(paths) - len(missing)
        need_computation += len(missing)

        if missing:
            for path_str in missing:
                path_to_seqs.setdefault(path_str, set()).add(seq_id)
                watch_dirs.add(Path(path_str).parent)

    total = sum(len(paths) for paths in expected.values())

    return _PreparedPlan(
        method_key=method_key,
        target=target,
        total=total,
        cached_already=cached_already,
        need_computation=need_computation,
        missing_paths_by_seq=missing_paths_by_seq,
        path_to_seqs=path_to_seqs,
        watch_dirs=watch_dirs,
    )


# inotify constants (Linux)
_IN_CLOSE_WRITE = 0x00000008
_IN_MOVED_TO = 0x00000080
_IN_CREATE = 0x00000100
_IN_DELETE_SELF = 0x00000400
_IN_MOVE_SELF = 0x00000800
_IN_IGNORED = 0x00008000
_IN_ISDIR = 0x40000000
_IN_NONBLOCK = 0x00000800
_IN_CLOEXEC = 0x00080000


class _EmbeddingTracker:
    def __init__(self, *, job_public_id: str, plan: _PreparedPlan):
        self.job_public_id = job_public_id
        self.method_key = plan.method_key
        self.target = plan.target
        self.total = plan.total
        self.cached_already = plan.cached_already
        self.need_computation = plan.need_computation
        self.computed = 0
        self.remaining = self.need_computation
        self.state = "running" if self.need_computation > 0 else "done"

        self._missing_paths_by_seq = plan.missing_paths_by_seq
        self._path_to_seqs = plan.path_to_seqs
        self._watch_dirs = plan.watch_dirs
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._lock = threading.Lock()

        self._last_payload_signature: tuple | None = None
        self._last_write_monotonic = 0.0
        self._dirty = False

    def start(self) -> None:
        self._write_snapshot(force=True)
        if self.need_computation == 0:
            return
        self._thread = threading.Thread(
            target=self._run,
            name=f"embedding-progress-{self.job_public_id}",
            daemon=True,
        )
        self._thread.start()

    def stop(self, final_state: str = "done") -> None:
        self._stop_event.set()
        if self._thread is not None and self._thread.is_alive():
            self._thread.join(timeout=3.0)
        with self._lock:
            if self.state != "done":
                self.state = final_state
            if self.remaining <= 0 and self.state != "error":
                self.state = "done"
        self._write_snapshot(force=True)

    def _payload(self) -> dict:
        return {
            "enabled": True,
            "state": self.state,
            "method_key": self.method_key,
            "methodKey": self.method_key,
            "target": self.target,
            "total": int(self.total),
            "cached_already": int(self.cached_already),
            "cachedAlready": int(self.cached_already),
            "need_computation": int(self.need_computation),
            "needComputation": int(self.need_computation),
            "computed": int(self.computed),
            "remaining": int(self.remaining),
            "updatedAt": _now_iso(),
        }

    def _signature(self) -> tuple:
        return (
            self.state,
            int(self.total),
            int(self.cached_already),
            int(self.need_computation),
            int(self.computed),
            int(self.remaining),
        )

    def _write_snapshot(self, force: bool = False) -> None:
        signature = self._signature()
        if not force and signature == self._last_payload_signature:
            return
        now = time.monotonic()
        if not force and (now - self._last_write_monotonic) < _REDIS_WRITE_MIN_INTERVAL_SECS:
            self._dirty = True
            return

        payload = self._payload()
        try:
            redis_conn.set(_redis_key(self.job_public_id), json.dumps(payload), ex=_TTL_SECONDS)
        except Exception as exc:
            # Progress is telemetry. Redis outages must not abort prediction.
            _log.warning(
                "Embedding progress Redis write skipped",
                extra={
                    "event": "embedding_progress.redis_write_skipped",
                    "job_public_id": self.job_public_id,
                    "method_key": self.method_key,
                    "target": self.target,
                    "exception_type": type(exc).__name__,
                },
                exc_info=True,
            )
        _sync_stage_embedding_progress(
            job_public_id=self.job_public_id,
            target=self.target,
            method_key=self.method_key,
            enabled=True,
            state=self.state,
            total=int(self.total),
            cached_already=int(self.cached_already),
            need_computation=int(self.need_computation),
            computed=int(self.computed),
            remaining=int(self.remaining),
        )
        self._last_payload_signature = signature
        self._last_write_monotonic = now
        self._dirty = False

    def _flush_if_due(self) -> None:
        if not self._dirty:
            return
        if (time.monotonic() - self._last_write_monotonic) < _REDIS_WRITE_MIN_INTERVAL_SECS:
            return
        self._write_snapshot(force=False)

    def _mark_path_present(self, path_str: str) -> bool:
        """
        Returns True if overall computed/remaining counters changed.
        """
        seqs = self._path_to_seqs.pop(path_str, None)
        if not seqs:
            return False

        progressed = False
        for seq_id in seqs:
            missing = self._missing_paths_by_seq.get(seq_id)
            if not missing or path_str not in missing:
                continue
            missing.remove(path_str)
            self.computed += 1
            self.remaining = max(0, self.remaining - 1)
            progressed = True
            if not missing:
                del self._missing_paths_by_seq[seq_id]

        if self.remaining == 0:
            self.state = "done"

        return progressed

    def _reconcile_pending(self) -> None:
        grouped: dict[tuple[Path, str], set[str]] = {}
        path_to_meta: dict[str, tuple[tuple[Path, str], str]] = {}

        for path_str in list(self._path_to_seqs.keys()):
            path = Path(path_str)
            group_key = (path.parent, path.suffix)
            seq_id = path.stem
            grouped.setdefault(group_key, set()).add(seq_id)
            path_to_meta[path_str] = (group_key, seq_id)

        ready_by_group: dict[tuple[Path, str], set[str]] = {}
        for group_key, seq_ids in grouped.items():
            cache_dir, suffix = group_key
            _missing, ready = resolve_missing_ids(
                seq_ids,
                cache_dir=cache_dir,
                suffix=suffix,
            )
            ready_by_group[group_key] = ready

        progressed = False
        for path_str, (group_key, seq_id) in path_to_meta.items():
            if seq_id in ready_by_group.get(group_key, set()):
                progressed = self._mark_path_present(path_str) or progressed
        if progressed:
            self._write_snapshot(force=False)

    def _run(self) -> None:
        use_inotify = False
        inotify_fd: int | None = None
        wd_to_dir: dict[int, Path] = {}

        try:
            inotify_fd = _inotify_init()
            if inotify_fd is not None:
                use_inotify = True
                for watch_dir in self._watch_dirs:
                    _add_watch_if_exists(inotify_fd, watch_dir, wd_to_dir)
        except Exception as exc:
            _log.info(
                "Embedding progress inotify disabled",
                extra={
                    "event": "embedding_progress.inotify_disabled",
                    "job_public_id": self.job_public_id,
                    "method_key": self.method_key,
                    "target": self.target,
                    "exception_type": type(exc).__name__,
                },
            )
            use_inotify = False
            inotify_fd = None

        next_reconcile = time.time() + _RECONCILE_SECS
        next_poll = time.time() + _FALLBACK_POLL_SECS

        try:
            while not self._stop_event.is_set() and self.remaining > 0:
                now = time.time()
                if use_inotify and inotify_fd is not None:
                    timeout = max(0.1, min(0.5, next_reconcile - now))
                    ready, _, _ = select.select([inotify_fd], [], [], timeout)
                    if ready:
                        progressed = _consume_inotify_events(
                            inotify_fd=inotify_fd,
                            wd_to_dir=wd_to_dir,
                            on_file=self._mark_path_present,
                        )
                        if progressed:
                            self._write_snapshot(force=False)
                else:
                    wait_for = max(0.1, min(0.5, next_poll - now))
                    self._stop_event.wait(wait_for)

                now = time.time()
                if now >= next_reconcile:
                    self._reconcile_pending()
                    next_reconcile = now + _RECONCILE_SECS

                if not use_inotify and now >= next_poll:
                    self._reconcile_pending()
                    next_poll = now + _FALLBACK_POLL_SECS
                self._flush_if_due()

            if self.remaining == 0:
                self.state = "done"
                self._write_snapshot(force=False)
        finally:
            if inotify_fd is not None:
                try:
                    os.close(inotify_fd)
                except OSError:
                    pass


def _inotify_init() -> int | None:
    if sys.platform != "linux":
        return None

    libc_name = ctypes.util.find_library("c")
    if not libc_name:
        return None
    libc = ctypes.CDLL(libc_name, use_errno=True)
    init_fn = getattr(libc, "inotify_init1", None)
    if init_fn is None:
        return None

    init_fn.argtypes = [ctypes.c_int]
    init_fn.restype = ctypes.c_int

    fd = init_fn(_IN_NONBLOCK | _IN_CLOEXEC)
    if fd < 0:
        err = ctypes.get_errno()
        raise OSError(err, os.strerror(err))
    return fd


def _inotify_add_watch(inotify_fd: int, path: Path) -> int | None:
    libc_name = ctypes.util.find_library("c")
    if not libc_name:
        return None
    libc = ctypes.CDLL(libc_name, use_errno=True)
    add_fn = getattr(libc, "inotify_add_watch", None)
    if add_fn is None:
        return None

    add_fn.argtypes = [ctypes.c_int, ctypes.c_char_p, ctypes.c_uint32]
    add_fn.restype = ctypes.c_int

    mask = _IN_CLOSE_WRITE | _IN_MOVED_TO | _IN_CREATE | _IN_DELETE_SELF | _IN_MOVE_SELF
    wd = add_fn(inotify_fd, str(path).encode("utf-8"), mask)
    if wd < 0:
        err = ctypes.get_errno()
        # Missing directories are common during warm-up; not fatal.
        if err in (2, 20):  # ENOENT / ENOTDIR
            return None
        raise OSError(err, os.strerror(err))
    return wd


def _add_watch_if_exists(inotify_fd: int, watch_dir: Path, wd_to_dir: dict[int, Path]) -> None:
    # Watch only concrete directories for this profile. Missing directories are
    # handled by reconciliation sweeps to avoid broad watches (e.g., '/').
    candidate = watch_dir
    if not candidate.exists() or not candidate.is_dir():
        return

    if candidate in wd_to_dir.values():
        return

    wd = _inotify_add_watch(inotify_fd, candidate)
    if wd is not None:
        wd_to_dir[wd] = candidate


def _consume_inotify_events(
    *,
    inotify_fd: int,
    wd_to_dir: dict[int, Path],
    on_file,
) -> bool:
    progressed = False

    while True:
        try:
            data = os.read(inotify_fd, 65536)
        except BlockingIOError:
            break
        except OSError:
            break

        if not data:
            break

        offset = 0
        while offset + 16 <= len(data):
            wd, mask, _cookie, name_len = struct.unpack_from("iIII", data, offset)
            offset += 16
            raw_name = data[offset : offset + name_len]
            offset += name_len

            watch_dir = wd_to_dir.get(wd)
            if watch_dir is None:
                continue

            if mask & _IN_IGNORED:
                wd_to_dir.pop(wd, None)
                continue

            name = raw_name.split(b"\x00", 1)[0].decode("utf-8", errors="ignore")
            if not name:
                continue

            event_path = (watch_dir / name).resolve()
            if mask & _IN_ISDIR:
                # Directory events are handled by periodic reconciliation.
                continue

            if mask & (_IN_CLOSE_WRITE | _IN_MOVED_TO):
                progressed = on_file(str(event_path)) or progressed

    return progressed
