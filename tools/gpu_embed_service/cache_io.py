from __future__ import annotations

import concurrent.futures
import json
import os
import shutil
import tempfile
import threading
import time
from collections import defaultdict
from pathlib import Path
from typing import Iterable

import numpy as np

try:
    import torch
except Exception:  # pragma: no cover - torch availability depends on runtime env
    torch = None  # type: ignore


_MANIFEST_NAME = "manifest.json"


def _unlink_if_exists(path: Path) -> None:
    try:
        path.unlink()
    except FileNotFoundError:
        return
    except OSError:
        return


def _atomic_write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(
        prefix=f".{path.stem}.",
        suffix=".json",
        dir=str(path.parent),
    )
    os.close(fd)
    tmp_path = Path(tmp_name)
    try:
        with tmp_path.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, separators=(",", ":"), sort_keys=True)
        os.replace(tmp_path, path)
    finally:
        if tmp_path.exists():
            _unlink_if_exists(tmp_path)


def _safe_stat_nbytes(path: Path) -> int:
    try:
        return int(path.stat().st_size)
    except OSError:
        return 0


def read_manifest_entries(cache_dir: Path, *, suffix: str | None = None) -> dict[str, dict]:
    """Read ready entries from <cache_dir>/manifest.json.

    Returns mapping:
      {seq_id: {"filename": "...", "bytes": int, "updated_at": float, "ready": True}}
    """
    manifest_path = (cache_dir / _MANIFEST_NAME).resolve()
    if not manifest_path.exists():
        return {}

    try:
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    if not isinstance(payload, dict):
        return {}

    raw_entries = payload.get("entries")
    if not isinstance(raw_entries, dict):
        return {}

    out: dict[str, dict] = {}
    for seq_id, item in raw_entries.items():
        if not isinstance(seq_id, str) or not seq_id:
            continue
        if not isinstance(item, dict):
            continue
        if not bool(item.get("ready", False)):
            continue

        filename = str(item.get("filename", "")).strip()
        if not filename:
            continue
        if suffix is not None and Path(filename).suffix != suffix:
            continue
        out[seq_id] = {
            "filename": filename,
            "bytes": int(item.get("bytes", 0) or 0),
            "updated_at": float(item.get("updated_at", 0.0) or 0.0),
            "ready": True,
        }
    return out


def snapshot_ready_ids(
    cache_dir: Path,
    *,
    suffix: str,
    only_ids: set[str] | None = None,
) -> set[str]:
    """Directory snapshot: one iterdir() pass for ready artifact stems."""
    if not cache_dir.is_dir():
        return set()

    out: set[str] = set()
    try:
        for entry in cache_dir.iterdir():
            if not entry.is_file():
                continue
            if entry.name == _MANIFEST_NAME:
                continue
            if entry.suffix != suffix:
                continue
            seq_id = entry.stem
            if not seq_id:
                continue
            if only_ids is not None and seq_id not in only_ids:
                continue
            out.add(seq_id)
    except OSError:
        return set()
    return out


def probe_ready_ids(
    cache_dir: Path,
    seq_ids: Iterable[str],
    *,
    suffix: str,
) -> set[str]:
    """Check exactly the requested IDs with one stat per candidate path.

    This is a targeted alternative to snapshot_ready_ids(): instead of scanning
    (and stat-ing) every entry in the cache directory, it probes only the paths
    we actually care about. On large, network-mounted cache directories (e.g.
    sshfs/NFS with hundreds of thousands of files) a full iterdir()+is_file()
    scan serialises into many thousands of round-trips and can take minutes to
    hours under load, whereas this probes O(len(seq_ids)) paths.
    """
    ready: set[str] = set()
    for seq_id in seq_ids:
        seq_id = str(seq_id).strip()
        if not seq_id:
            continue
        candidate = cache_dir / f"{seq_id}{suffix}"
        try:
            if candidate.is_file():
                ready.add(seq_id)
        except OSError:
            continue
    return ready


def resolve_missing_ids(
    seq_ids: Iterable[str],
    *,
    cache_dir: Path,
    suffix: str,
) -> tuple[list[str], set[str]]:
    """Resolve missing IDs via manifest first, then targeted per-ID probing.

    Returns:
      (missing_seq_ids_in_input_order, ready_seq_ids)
    """
    ordered: list[str] = []
    seen: set[str] = set()
    for seq_id in seq_ids:
        seq_id = str(seq_id).strip()
        if not seq_id or seq_id in seen:
            continue
        seen.add(seq_id)
        ordered.append(seq_id)
    wanted = set(ordered)

    manifest_entries = read_manifest_entries(cache_dir, suffix=suffix)
    ready = set(manifest_entries.keys()) & wanted

    # For IDs the manifest does not vouch for, probe only those exact paths
    # rather than scanning the whole (potentially enormous) cache directory.
    # The manifest is frequently incomplete relative to on-disk files (writes
    # from CPU fallbacks / legacy paths never update it), so this fallback
    # fires often — it must stay O(unresolved), not O(dir size).
    unresolved = wanted - ready
    if unresolved:
        ready |= probe_ready_ids(cache_dir, unresolved, suffix=suffix)

    missing = [seq_id for seq_id in ordered if seq_id not in ready]
    return missing, ready


def merge_manifest_entries(cache_dir: Path, updates: dict[str, dict]) -> None:
    """Merge successful writes into cache manifest, then atomically publish."""
    if not updates:
        return
    cache_dir = cache_dir.resolve()
    manifest_path = cache_dir / _MANIFEST_NAME

    payload: dict
    try:
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            payload = {}
    except Exception:
        payload = {}

    entries = payload.get("entries")
    if not isinstance(entries, dict):
        entries = {}

    for seq_id, entry in updates.items():
        if not isinstance(seq_id, str) or not seq_id:
            continue
        if not isinstance(entry, dict):
            continue
        filename = str(entry.get("filename", "")).strip()
        if not filename:
            continue
        entries[seq_id] = {
            "filename": filename,
            "bytes": int(entry.get("bytes", 0) or 0),
            "updated_at": float(entry.get("updated_at", time.time())),
            "ready": True,
        }

    payload["version"] = 1
    payload["updated_at"] = time.time()
    payload["entries"] = entries
    payload["count"] = len(entries)
    _atomic_write_json(manifest_path, payload)


def remove_manifest_entries(cache_dir: Path, seq_ids: Iterable[str]) -> None:
    """Remove entries from a cache manifest after ephemeral file cleanup.

    This keeps future planning fast: resolve_missing_ids can continue trusting
    the manifest without doing one mounted-filesystem stat per requested ID.
    """
    cache_dir = cache_dir.resolve()
    manifest_path = cache_dir / _MANIFEST_NAME
    try:
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            return
    except Exception:
        return

    entries = payload.get("entries")
    if not isinstance(entries, dict):
        return

    removed = False
    for seq_id in seq_ids:
        key = str(seq_id).strip()
        if key and key in entries:
            entries.pop(key, None)
            removed = True

    if not removed:
        return

    payload["version"] = 1
    payload["updated_at"] = time.time()
    payload["entries"] = entries
    payload["count"] = len(entries)
    _atomic_write_json(manifest_path, payload)


class SpoolAsyncCommitter:
    """Write artifacts to local spool, then async commit to shared cache.

    Manifests are published only after all queued commits succeed.
    """

    def __init__(
        self,
        *,
        max_workers: int = 8,
        spool_dir: Path | None = None,
        spool_fallback_dir: Path | None = None,
    ) -> None:
        self._max_workers = max(1, int(max_workers))
        self._pool = concurrent.futures.ThreadPoolExecutor(
            max_workers=self._max_workers,
            thread_name_prefix="gpu-cache-commit",
        )
        self._futures: list[concurrent.futures.Future] = []
        self._lock = threading.Lock()
        self._closed = False

        preferred = Path(spool_dir) if spool_dir is not None else Path("/dev/shm/webkinpred-gpu-cache")
        fallback = Path(spool_fallback_dir) if spool_fallback_dir is not None else Path("/tmp/webkinpred-gpu-cache")
        self._spool_root = self._ensure_spool_root(preferred, fallback)
        self._session_dir = self._spool_root / f"session_{os.getpid()}_{int(time.time() * 1000)}"
        self._session_dir.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def _ensure_spool_root(preferred: Path, fallback: Path) -> Path:
        for root in (preferred, fallback):
            try:
                root.mkdir(parents=True, exist_ok=True)
                probe = root / ".probe"
                probe.write_text("ok", encoding="utf-8")
                _unlink_if_exists(probe)
                return root.resolve()
            except OSError:
                continue
        raise RuntimeError(
            f"Could not create writable spool roots: preferred={preferred}, fallback={fallback}"
        )

    def _assert_open(self) -> None:
        if self._closed:
            raise RuntimeError("SpoolAsyncCommitter is closed.")

    def _spool_path(self, *, suffix: str) -> Path:
        fd, tmp_name = tempfile.mkstemp(
            prefix=".spool.",
            suffix=suffix,
            dir=str(self._session_dir),
        )
        os.close(fd)
        return Path(tmp_name)

    @staticmethod
    def _commit_spooled_file(spool_path: Path, dest_path: Path) -> tuple[Path, str, int]:
        dest_path.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp_name = tempfile.mkstemp(
            prefix=f".{dest_path.stem}.",
            suffix=dest_path.suffix,
            dir=str(dest_path.parent),
        )
        os.close(fd)
        tmp_path = Path(tmp_name)
        try:
            shutil.copyfile(spool_path, tmp_path)
            os.replace(tmp_path, dest_path)
            nbytes = _safe_stat_nbytes(dest_path)
            return dest_path.parent.resolve(), dest_path.name, nbytes
        finally:
            _unlink_if_exists(tmp_path)
            _unlink_if_exists(spool_path)

    def _submit_spooled(self, *, cache_dir: Path, seq_id: str, suffix: str, spool_path: Path) -> None:
        self._assert_open()
        dest_path = (cache_dir / f"{seq_id}{suffix}").resolve()
        future = self._pool.submit(self._commit_spooled_file, spool_path, dest_path)
        future._webkp_meta = {  # type: ignore[attr-defined]
            "cache_dir": cache_dir.resolve(),
            "seq_id": seq_id,
            "suffix": suffix,
        }
        with self._lock:
            self._futures.append(future)

    def submit_numpy(self, *, cache_dir: Path, seq_id: str, array: np.ndarray) -> None:
        spool_path = self._spool_path(suffix=".npy")
        np.save(spool_path, np.asarray(array))
        self._submit_spooled(cache_dir=cache_dir, seq_id=seq_id, suffix=".npy", spool_path=spool_path)

    def submit_torch_tensor(self, *, cache_dir: Path, seq_id: str, tensor) -> None:
        if torch is None:
            raise RuntimeError("torch is not available for submit_torch_tensor.")
        spool_path = self._spool_path(suffix=".pt")
        torch.save(tensor, spool_path)
        self._submit_spooled(cache_dir=cache_dir, seq_id=seq_id, suffix=".pt", spool_path=spool_path)

    def wait_for_completion(self) -> None:
        with self._lock:
            futures, self._futures = list(self._futures), []

        if not futures:
            return

        manifest_updates: dict[Path, dict[str, dict]] = defaultdict(dict)
        errors: list[str] = []

        for future in futures:
            meta = getattr(future, "_webkp_meta", {})  # type: ignore[attr-defined]
            cache_dir = Path(meta.get("cache_dir", ""))
            seq_id = str(meta.get("seq_id", ""))
            try:
                committed_cache_dir, filename, nbytes = future.result()
            except Exception as exc:
                errors.append(f"{seq_id}: {exc}")
                continue
            if committed_cache_dir != cache_dir or not seq_id:
                errors.append(f"{seq_id or 'unknown'}: commit metadata mismatch")
                continue
            manifest_updates[cache_dir][seq_id] = {
                "filename": filename,
                "bytes": int(nbytes),
                "updated_at": time.time(),
                "ready": True,
            }

        if errors:
            raise RuntimeError("Async cache commit failed: " + "; ".join(errors))

        for cache_dir, updates in manifest_updates.items():
            merge_manifest_entries(cache_dir, updates)

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        try:
            self._pool.shutdown(wait=True)
        finally:
            shutil.rmtree(self._session_dir, ignore_errors=True)

    def __enter__(self) -> "SpoolAsyncCommitter":
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        try:
            if exc_type is None:
                self.wait_for_completion()
        finally:
            self.close()
        return False
