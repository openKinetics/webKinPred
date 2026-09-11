#!/usr/bin/env python3
from __future__ import annotations

import concurrent.futures
import csv
import json
import os
import pickle
import queue
import shlex
import socket
import subprocess
import sys
import tempfile
import threading
import time
from collections import OrderedDict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

import numpy as np
try:
    import torch
except Exception:  # pragma: no cover - depends on runtime env
    torch = None  # type: ignore

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from tools.gpu_embed_service.kinform_stream_ipc import recv_frame, send_frame


_T5_FAMILY = "t5"
_ESM2_FAMILY = "esm2"
_ESMC_FAMILY = "esmc"
_PSEQ_WORKER = "pseq2sites"

_FAMILY_ROOTS: dict[str, tuple[str, ...]] = {
    _T5_FAMILY: ("prot_t5_layer_19", "prot_t5_last"),
    _ESM2_FAMILY: ("esm2_layer_26", "esm2_layer_29"),
    _ESMC_FAMILY: ("esmc_layer_24", "esmc_layer_32"),
}


def _timing_stage_for_worker(worker_name: str) -> str:
    if worker_name == _T5_FAMILY:
        return "prot_t5_residue"
    if worker_name == _ESM2_FAMILY:
        return "esm_residue"
    if worker_name == _ESMC_FAMILY:
        return "esmc_residue"
    if worker_name == _PSEQ_WORKER:
        return "pseq2sites_preds"
    return worker_name


def _env_bool(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    value = str(raw).strip().lower()
    if value in {"1", "true", "yes", "on"}:
        return True
    if value in {"0", "false", "no", "off"}:
        return False
    return default


def _env_float(name: str, default: float) -> float:
    raw = os.environ.get(name)
    if raw is None:
        return float(default)
    try:
        return float(raw)
    except Exception:
        return float(default)


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None:
        return int(default)
    try:
        return int(str(raw).strip())
    except Exception:
        return int(default)


def _pseq_retry_batch_plan(env: dict[str, str]) -> list[int]:
    """
    Build pseq retry batch sizes from the configured starting batch size.
    Example: start=8 -> [8, 4, 2, 1].
    """
    start = max(1, int(_env_float("KINFORM_PARALLEL_PSEQ_STREAM_BATCH_SIZE", 8.0)))
    plan = [start]
    for candidate in (4, 2, 1):
        if candidate < start and candidate not in plan:
            plan.append(candidate)
    return plan


def _log_level_value(name: str) -> int:
    mapping = {
        "debug": 10,
        "info": 20,
        "warn": 30,
        "warning": 30,
        "error": 40,
        "quiet": 100,
    }
    return mapping.get(name.strip().lower(), 20)


def _log(
    env: dict[str, str],
    level: str,
    message: str,
    *,
    job_id: str | None = None,
) -> None:
    configured = _log_level_value(str(env.get("KINFORM_PARALLEL_LOG_LEVEL", "info")))
    current = _log_level_value(level)
    if current < configured:
        return
    job = job_id or "unknown"
    print(f"KINFORM_PARALLEL_{level.upper()} job_id={job} {message}")


def _artifact_path(media_path: Path, root: str, kind: str, seq_id: str) -> Path:
    # Keep path arithmetic cheap in hot loops; avoid resolve() network metadata lookups.
    return media_path / "sequence_info" / root / f"{kind}_vecs" / f"{seq_id}.npy"


def _binding_score_path(media_path: Path, seq_id: str) -> Path:
    return media_path / "sequence_info" / "pseq2sites_scores" / f"{seq_id}.npy"


def _truncate_first_last_array(arr: np.ndarray, keep: int = 1024) -> np.ndarray:
    if arr.shape[0] <= keep:
        return arr
    half = keep // 2
    return np.concatenate([arr[:half], arr[-half:]], axis=0)


def _align_residue_to_binding_weights(
    residue_embedding: np.ndarray,
    weights: np.ndarray,
) -> np.ndarray:
    if residue_embedding.shape[0] == weights.shape[0]:
        return residue_embedding
    if weights.shape[0] == 1024 and residue_embedding.shape[0] > 1024:
        return _truncate_first_last_array(residue_embedding, keep=1024)
    return residue_embedding


def _read_binding_site_scores(
    binding_sites_path: Path,
    *,
    target_seq_ids: set[str] | None = None,
) -> dict[str, np.ndarray]:
    if not binding_sites_path.exists():
        return {}

    out: dict[str, np.ndarray] = {}
    wanted = set(target_seq_ids) if target_seq_ids else None
    try:
        with binding_sites_path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle, delimiter="\t")
            if not reader.fieldnames:
                return {}
            key_col = "PDB" if "PDB" in reader.fieldnames else reader.fieldnames[0]
            score_col = (
                "Pred_BS_Scores"
                if "Pred_BS_Scores" in reader.fieldnames
                else (reader.fieldnames[1] if len(reader.fieldnames) > 1 else "")
            )
            if not score_col:
                return {}
            for row in reader:
                seq_id = str(row.get(key_col, "")).strip()
                score_text = str(row.get(score_col, "")).strip()
                if not seq_id or not score_text:
                    continue
                if wanted is not None and seq_id not in wanted:
                    continue
                try:
                    weights = np.fromiter((float(x) for x in score_text.split(",")), dtype=np.float64)
                except ValueError:
                    continue
                if weights.size:
                    out[seq_id] = weights
                    if wanted is not None and len(out) >= len(wanted):
                        break
    except Exception:
        return {}
    return out


def _load_binding_score_cache(
    *,
    media_path: Path,
    seq_ids: Iterable[str],
    max_workers: int = 16,
) -> dict[str, np.ndarray]:
    seq_list = list(seq_ids)
    if not seq_list:
        return {}

    def _load_one(seq_id: str) -> tuple[str, np.ndarray | None]:
        path = _binding_score_path(media_path, seq_id)
        if not path.exists():
            return seq_id, None
        try:
            arr = np.load(path).astype(np.float32, copy=False).reshape(-1)
            if arr.size == 0:
                return seq_id, None
            return seq_id, arr
        except Exception:
            return seq_id, None

    out: dict[str, np.ndarray] = {}
    with concurrent.futures.ThreadPoolExecutor(
        max_workers=max(1, int(max_workers)),
        thread_name_prefix="kinform-bs-load",
    ) as pool:
        for seq_id, arr in pool.map(_load_one, seq_list):
            if arr is not None:
                out[seq_id] = arr
    return out


class BindingSiteScoreCache:
    def __init__(self, binding_sites_path: Path, *, target_seq_ids: set[str] | None = None) -> None:
        self.binding_sites_path = binding_sites_path
        self.target_seq_ids = set(target_seq_ids) if target_seq_ids else None
        self._mtime_ns: int | None = None
        self._scores: dict[str, np.ndarray] = {}

    def read(self) -> dict[str, np.ndarray]:
        if not self.binding_sites_path.exists():
            self._mtime_ns = None
            self._scores = {}
            return {}
        stat = self.binding_sites_path.stat()
        if self._mtime_ns != stat.st_mtime_ns:
            self._scores = _read_binding_site_scores(
                self.binding_sites_path,
                target_seq_ids=self.target_seq_ids,
            )
            self._mtime_ns = stat.st_mtime_ns
        return self._scores


def weighted_mean_from_residue(residue_embedding: np.ndarray, weights: np.ndarray) -> np.ndarray:
    if residue_embedding.ndim != 2:
        raise ValueError(f"Expected 2D residue embedding array, got shape {residue_embedding.shape}")
    if weights.ndim != 1:
        raise ValueError(f"Expected 1D weights array, got shape {weights.shape}")
    residue_embedding = _align_residue_to_binding_weights(residue_embedding, weights)
    if residue_embedding.shape[0] != weights.shape[0]:
        raise ValueError(
            f"Weight length ({weights.shape[0]}) != residue length ({residue_embedding.shape[0]})"
        )
    safe_weights = np.nan_to_num(
        weights.astype(np.float32, copy=False),
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )
    denom = float(np.sum(safe_weights))
    if not np.isfinite(denom) or denom <= 0.0:
        # Degenerate scores (all zeros/non-finite): keep pipeline moving by
        # falling back to global mean pooling.
        return residue_embedding.astype(np.float32).mean(axis=0).astype(np.float32, copy=False)
    normalized = safe_weights / denom
    vec = (residue_embedding.astype(np.float32) * normalized[:, None]).sum(axis=0)
    return vec.astype(np.float32, copy=False)


def weighted_mean_from_residue_gpu(
    residue_embedding: torch.Tensor,
    weights: np.ndarray,
    *,
    device: torch.device,
) -> np.ndarray:
    if torch is None:
        raise RuntimeError("torch is required for GPU weighted derivation.")
    if residue_embedding.ndim != 2:
        raise ValueError(f"Expected 2D residue embedding tensor, got shape {tuple(residue_embedding.shape)}")
    if residue_embedding.device != device:
        residue_embedding = residue_embedding.to(device)
    if residue_embedding.dtype != torch.float32:
        residue_embedding = residue_embedding.float()

    w = torch.as_tensor(weights, dtype=torch.float32, device=device)
    if w.ndim != 1:
        raise ValueError(f"Expected 1D weights tensor, got shape {tuple(w.shape)}")
    if residue_embedding.shape[0] != w.shape[0] and w.shape[0] == 1024 and residue_embedding.shape[0] > 1024:
        half = 1024 // 2
        residue_embedding = torch.cat((residue_embedding[:half], residue_embedding[-half:]), dim=0)
    if residue_embedding.shape[0] != w.shape[0]:
        raise ValueError(
            f"Weight length ({w.shape[0]}) != residue length ({residue_embedding.shape[0]})"
        )
    w = torch.nan_to_num(w, nan=0.0, posinf=0.0, neginf=0.0)
    denom = torch.sum(w)
    denom_value = float(denom.item())
    if not np.isfinite(denom_value) or denom_value <= 0.0:
        out = torch.mean(residue_embedding, dim=0)
        return out.detach().cpu().numpy().astype(np.float32, copy=False)
    normalized = w / denom
    out = torch.sum(residue_embedding * normalized.unsqueeze(1), dim=0)
    return out.detach().cpu().numpy().astype(np.float32, copy=False)


def _save_array_atomic(path: Path, arr: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=f".{path.stem}.", suffix=".npy", dir=str(path.parent))
    os.close(fd)
    tmp_path = Path(tmp_name)
    try:
        np.save(tmp_path, arr)
        os.replace(tmp_path, path)
    finally:
        if tmp_path.exists():
            tmp_path.unlink(missing_ok=True)


def _remove_path_if_exists(path: Path) -> None:
    try:
        if path.exists():
            path.unlink()
    except OSError:
        pass


class AsyncWriter:
    """Background thread pool for non-blocking SSHFS writes."""

    def __init__(self, max_workers: int = 4) -> None:
        self._pool = concurrent.futures.ThreadPoolExecutor(
            max_workers=max_workers,
            thread_name_prefix="kinform-writer",
        )
        self._futures: list[concurrent.futures.Future] = []
        self._lock = threading.Lock()

    def submit(self, path: Path, arr: np.ndarray) -> None:
        arr_copy = arr.copy()
        f = self._pool.submit(_save_array_atomic, path, arr_copy)
        with self._lock:
            self._futures.append(f)

    def join(self) -> None:
        with self._lock:
            futures, self._futures = list(self._futures), []
        for f in futures:
            f.result()

    def shutdown(self) -> None:
        self._pool.shutdown(wait=True)


@dataclass
class ArtifactTargets:
    seq_ids: list[str]
    media_path: Path
    binding_sites_path: Path
    binding_site_existing_ids: set[str] | None = None
    weighted_targets: dict[tuple[str, str], set[str]] = field(default_factory=dict)
    mean_targets: dict[tuple[str, str], set[str]] = field(default_factory=dict)
    binding_site_targets: set[str] = field(default_factory=set)

    def __post_init__(self) -> None:
        checks: list[tuple[tuple[str, str], str, str, Path]] = []
        for family, roots in _FAMILY_ROOTS.items():
            for root in roots:
                key = (family, root)
                self.weighted_targets[key] = set()
                self.mean_targets[key] = set()
                for seq_id in self.seq_ids:
                    checks.append((key, seq_id, "weighted", _artifact_path(self.media_path, root, "weighted", seq_id)))
                    checks.append((key, seq_id, "mean", _artifact_path(self.media_path, root, "mean", seq_id)))

        def _check(item: tuple) -> tuple:
            key, seq_id, kind, path = item
            return (key, seq_id, kind, path.exists())

        with concurrent.futures.ThreadPoolExecutor(max_workers=32, thread_name_prefix="kinform-stat") as pool:
            results = list(pool.map(_check, checks))

        for key, seq_id, kind, exists in results:
            if not exists:
                if kind == "weighted":
                    self.weighted_targets[key].add(seq_id)
                else:
                    self.mean_targets[key].add(seq_id)

        if self.binding_site_existing_ids is None:
            seen_ids = set(_read_binding_site_scores(self.binding_sites_path).keys())
        else:
            seen_ids = set(self.binding_site_existing_ids)
        self.binding_site_targets = {sid for sid in self.seq_ids if sid not in seen_ids}

    def all_done(self, bs_scores: dict[str, np.ndarray]) -> bool:
        for (family, root), targets in self.weighted_targets.items():
            for seq_id in targets:
                if not _artifact_path(self.media_path, root, "weighted", seq_id).exists():
                    return False
        for (family, root), targets in self.mean_targets.items():
            for seq_id in targets:
                if not _artifact_path(self.media_path, root, "mean", seq_id).exists():
                    return False
        for seq_id in self.binding_site_targets:
            if seq_id not in bs_scores:
                return False
        return True

    def missing_weighted_count(self, family: str) -> tuple[int, int]:
        missing = 0
        total = 0
        for (fam, root), targets in self.weighted_targets.items():
            if fam != family:
                continue
            total += len(targets)
            for seq_id in targets:
                if not _artifact_path(self.media_path, root, "weighted", seq_id).exists():
                    missing += 1
        return missing, total

    def missing_mean_count(self, family: str) -> tuple[int, int]:
        missing = 0
        total = 0
        for (fam, root), targets in self.mean_targets.items():
            if fam != family:
                continue
            total += len(targets)
            for seq_id in targets:
                if not _artifact_path(self.media_path, root, "mean", seq_id).exists():
                    missing += 1
        return missing, total


ResidueKey = tuple[str, str, str]


@dataclass
class ResidueEntry:
    nbytes: int
    array: np.ndarray | None = None
    spill_path: Path | None = None


class ResidueCache:
    def __init__(
        self,
        *,
        max_bytes: int,
        spill_dir: Path,
        spill_fallback_dir: Path,
        env: dict[str, str],
        job_id: str | None,
    ) -> None:
        self.max_bytes = max(1, int(max_bytes))
        self.spill_dir = spill_dir
        self.spill_fallback_dir = spill_fallback_dir
        self.env = env
        self.job_id = job_id
        self.entries: OrderedDict[ResidueKey, ResidueEntry] = OrderedDict()
        self._resident_bytes = 0

    @property
    def resident_bytes(self) -> int:
        return self._resident_bytes

    def has(self, key: ResidueKey) -> bool:
        return key in self.entries

    def keys(self) -> Iterable[ResidueKey]:
        return tuple(self.entries.keys())

    def put(self, key: ResidueKey, residue: np.ndarray) -> None:
        arr = np.ascontiguousarray(residue, dtype=np.float32)
        nbytes = int(arr.nbytes)

        if key in self.entries:
            self.remove(key)

        self._ensure_budget_for(nbytes)

        self.entries[key] = ResidueEntry(nbytes=nbytes, array=arr, spill_path=None)
        self.entries.move_to_end(key)
        self._resident_bytes += nbytes

    def _ensure_budget_for(self, incoming_nbytes: int) -> None:
        while self._resident_bytes + incoming_nbytes > self.max_bytes and self.entries:
            evict_key, evict_entry = next(iter(self.entries.items()))
            if evict_entry.array is None:
                # Already spilled; move on to next entry.
                self.entries.move_to_end(evict_key)
                continue
            spill_path = self._spill_key(evict_key, evict_entry)
            evict_entry.spill_path = spill_path
            evict_entry.array = None
            self._resident_bytes -= evict_entry.nbytes
            self.entries[evict_key] = evict_entry
            self.entries.move_to_end(evict_key)
            _log(
                self.env,
                "debug",
                f"spilled residue key={evict_key} path={spill_path}",
                job_id=self.job_id,
            )

    def _spill_key(self, key: ResidueKey, entry: ResidueEntry) -> Path:
        family, root, seq_id = key
        rel_name = f"{family}__{root}__{seq_id}.npy"
        for base in (self.spill_dir, self.spill_fallback_dir):
            try:
                base.mkdir(parents=True, exist_ok=True)
                path = base / rel_name
                assert entry.array is not None
                arr = entry.array.astype(np.float32, copy=False)
                _save_array_atomic(path, arr)
                return path
            except Exception:
                continue
        raise RuntimeError(f"Could not spill residue for key={key} to any spill directory.")

    def get_numpy(self, key: ResidueKey) -> np.ndarray:
        entry = self.entries.get(key)
        if entry is None:
            raise KeyError(key)
        self.entries.move_to_end(key)

        if entry.array is not None:
            return entry.array

        if entry.spill_path is None or not entry.spill_path.exists():
            raise RuntimeError(f"Spill file missing for residue key={key}")

        arr = np.load(entry.spill_path).astype(np.float32, copy=False)
        nbytes = int(arr.nbytes)
        self._ensure_budget_for(nbytes)
        entry.array = arr
        entry.nbytes = nbytes
        self._resident_bytes += nbytes
        self.entries[key] = entry
        self.entries.move_to_end(key)
        return arr

    def remove(self, key: ResidueKey) -> None:
        entry = self.entries.pop(key, None)
        if entry is None:
            return
        if entry.array is not None:
            self._resident_bytes -= entry.nbytes
            entry.array = None
        if entry.spill_path is not None:
            _remove_path_if_exists(entry.spill_path)
            entry.spill_path = None

    def clear(self) -> None:
        for key in list(self.entries.keys()):
            self.remove(key)


@dataclass
class WorkerState:
    name: str
    attempts: int = 0
    process: subprocess.Popen | None = None
    tmp_inputs_dir: Path | None = None
    active_seq_ids: set[str] = field(default_factory=set)
    started_at_monotonic: float | None = None
    active_seq_count: int = 0
    elapsed_seconds_total: float = 0.0
    stream_done_received: bool = False
    waiting_for_stream_done_since: float | None = None
    pseq_batch_size: int | None = None

    def running(self) -> bool:
        return self.process is not None and self.process.poll() is None


@dataclass
class _ServerClient:
    sock: socket.socket
    send_lock: threading.Lock = field(default_factory=threading.Lock)


class StreamEventServer:
    def __init__(self, socket_path: Path) -> None:
        self.socket_path = socket_path
        self.events: queue.Queue[tuple[str, int, dict | None, bytes | None]] = queue.Queue()
        self._sock: socket.socket | None = None
        self._stop = threading.Event()
        self._lock = threading.Lock()
        self._clients: dict[int, _ServerClient] = {}
        self._accept_thread: threading.Thread | None = None
        self._client_threads: dict[int, threading.Thread] = {}
        self._next_client_id = 1

    def start(self) -> None:
        self.socket_path.parent.mkdir(parents=True, exist_ok=True)
        if self.socket_path.exists():
            self.socket_path.unlink(missing_ok=True)

        sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        sock.bind(str(self.socket_path))
        sock.listen(16)
        sock.settimeout(0.2)
        self._sock = sock

        self._accept_thread = threading.Thread(target=self._accept_loop, name="kinform-stream-accept", daemon=True)
        self._accept_thread.start()

    def _accept_loop(self) -> None:
        assert self._sock is not None
        while not self._stop.is_set():
            try:
                conn, _ = self._sock.accept()
            except socket.timeout:
                continue
            except Exception:
                if self._stop.is_set():
                    return
                continue

            conn.settimeout(None)
            with self._lock:
                client_id = self._next_client_id
                self._next_client_id += 1
                self._clients[client_id] = _ServerClient(sock=conn)
            self.events.put(("connect", client_id, None, None))

            thread = threading.Thread(
                target=self._client_loop,
                args=(client_id, conn),
                name=f"kinform-stream-client-{client_id}",
                daemon=True,
            )
            with self._lock:
                self._client_threads[client_id] = thread
            thread.start()

    def _client_loop(self, client_id: int, conn: socket.socket) -> None:
        try:
            while not self._stop.is_set():
                header, payload = recv_frame(conn)
                self.events.put(("event", client_id, header, payload))
        except EOFError:
            pass
        except Exception as exc:
            self.events.put(("error", client_id, {"error": str(exc)}, None))
        finally:
            self.events.put(("disconnect", client_id, None, None))
            with self._lock:
                client = self._clients.pop(client_id, None)
                self._client_threads.pop(client_id, None)
            if client is not None:
                try:
                    client.sock.close()
                except Exception:
                    pass

    def send(self, client_id: int, header: dict, payload: bytes = b"") -> None:
        with self._lock:
            client = self._clients.get(client_id)
        if client is None:
            raise RuntimeError(f"Stream client {client_id} is not connected.")
        with client.send_lock:
            send_frame(client.sock, header, payload)

    def recv_event(self, timeout_seconds: float) -> tuple[str, int, dict | None, bytes | None] | None:
        try:
            return self.events.get(timeout=max(0.0, timeout_seconds))
        except queue.Empty:
            return None

    def drain_events(self, *, max_items: int | None = None) -> list[tuple[str, int, dict | None, bytes | None]]:
        out: list[tuple[str, int, dict | None, bytes | None]] = []
        while True:
            if max_items is not None and len(out) >= max_items:
                return out
            try:
                out.append(self.events.get_nowait())
            except queue.Empty:
                return out

    def close(self) -> None:
        self._stop.set()

        if self._sock is not None:
            try:
                self._sock.close()
            except Exception:
                pass
            self._sock = None

        with self._lock:
            clients = list(self._clients.values())
            self._clients.clear()
        for client in clients:
            try:
                client.sock.close()
            except Exception:
                pass

        if self._accept_thread is not None:
            self._accept_thread.join(timeout=1.0)
        for thread in list(self._client_threads.values()):
            thread.join(timeout=1.0)

        if self.socket_path.exists():
            self.socket_path.unlink(missing_ok=True)


def _to_seq_subset(seq_id_to_seq: dict[str, str], seq_ids: set[str]) -> dict[str, str]:
    return {sid: seq_id_to_seq[sid] for sid in seq_id_to_seq if sid in seq_ids}


def _write_worker_inputs(seq_id_to_seq: dict[str, str]) -> tuple[Path, Path, Path]:
    tmp_dir = Path(tempfile.mkdtemp(prefix="kinform_parallel_worker_"))
    seq_file = tmp_dir / "seq_ids.txt"
    id_to_seq_pkl = tmp_dir / "id_to_seq.pkl"
    seq_map_json = tmp_dir / "seq_id_to_seq.json"

    with seq_file.open("w", encoding="utf-8") as handle:
        for seq_id in seq_id_to_seq:
            handle.write(f"{seq_id}\n")

    with id_to_seq_pkl.open("wb") as handle:
        pickle.dump(seq_id_to_seq, handle, protocol=4)

    seq_map_json.write_text(json.dumps(seq_id_to_seq), encoding="utf-8")
    return seq_file, id_to_seq_pkl, seq_map_json


def _start_worker(cmd: list[str], env: dict[str, str]) -> subprocess.Popen:
    print("+", " ".join(shlex.quote(c) for c in cmd))
    return subprocess.Popen(cmd, env=env)


def _terminate_worker(state: WorkerState) -> None:
    if state.process is None:
        return
    if state.process.poll() is None:
        state.process.terminate()
        try:
            state.process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            state.process.kill()
            state.process.wait(timeout=10)
    state.process = None


def _cleanup_worker_inputs(state: WorkerState) -> None:
    if state.tmp_inputs_dir is None:
        return
    tmp_dir = state.tmp_inputs_dir
    state.tmp_inputs_dir = None
    try:
        for child in tmp_dir.glob("*"):
            child.unlink(missing_ok=True)
        tmp_dir.rmdir()
    except OSError:
        pass


def _derive_mean_if_ready(
    *,
    targets: ArtifactTargets,
    family: str,
    root: str,
    seq_id: str,
) -> bool:
    if seq_id not in targets.mean_targets.get((family, root), set()):
        return False
    mean_path = _artifact_path(targets.media_path, root, "mean", seq_id)
    if mean_path.exists():
        return False
    residue_path = _artifact_path(targets.media_path, root, "residue", seq_id)
    if not residue_path.exists():
        return False
    residue = np.load(residue_path)
    _save_array_atomic(mean_path, residue.mean(axis=0).astype(np.float32))
    return True


def _derive_weighted_if_ready(
    *,
    targets: ArtifactTargets,
    family: str,
    root: str,
    seq_id: str,
    bs_scores: dict[str, np.ndarray],
    weighted_retry_errors: dict[tuple[str, str, str], int],
) -> bool:
    if seq_id not in targets.weighted_targets.get((family, root), set()):
        return False
    weighted_path = _artifact_path(targets.media_path, root, "weighted", seq_id)
    if weighted_path.exists():
        return False
    residue_path = _artifact_path(targets.media_path, root, "residue", seq_id)
    if not residue_path.exists():
        return False
    weights = bs_scores.get(seq_id)
    if weights is None:
        return False
    key = (family, root, seq_id)
    try:
        residue = np.load(residue_path)
        weighted_vec = weighted_mean_from_residue(residue, weights)
        _save_array_atomic(weighted_path, weighted_vec)
        _remove_path_if_exists(residue_path)
        weighted_retry_errors.pop(key, None)
        return True
    except Exception:
        weighted_retry_errors[key] = weighted_retry_errors.get(key, 0) + 1
        if weighted_retry_errors[key] > 1:
            raise
        return False


def _residue_consumers_done(
    *,
    targets: ArtifactTargets,
    family: str,
    root: str,
    seq_id: str,
    bs_scores: dict[str, np.ndarray],
) -> bool:
    if seq_id in targets.mean_targets.get((family, root), set()):
        if not _artifact_path(targets.media_path, root, "mean", seq_id).exists():
            return False
    if seq_id in targets.weighted_targets.get((family, root), set()):
        if not _artifact_path(targets.media_path, root, "weighted", seq_id).exists():
            return False
    if family == _T5_FAMILY and root == "prot_t5_last":
        if seq_id in targets.binding_site_targets and seq_id not in bs_scores:
            return False
    return True


def _cleanup_satisfied_residue_files(
    *,
    targets: ArtifactTargets,
    bs_scores: dict[str, np.ndarray],
) -> int:
    removed = 0
    for family, roots in _FAMILY_ROOTS.items():
        for root in roots:
            for seq_id in targets.seq_ids:
                residue_path = _artifact_path(targets.media_path, root, "residue", seq_id)
                if not residue_path.exists():
                    continue
                if not _residue_consumers_done(
                    targets=targets,
                    family=family,
                    root=root,
                    seq_id=seq_id,
                    bs_scores=bs_scores,
                ):
                    continue
                _remove_path_if_exists(residue_path)
                if not residue_path.exists():
                    removed += 1
    return removed


def _needs_t5_worker_seq_ids_file(
    *,
    targets: ArtifactTargets,
    bs_scores: dict[str, np.ndarray],
) -> set[str]:
    out: set[str] = set()
    t5_last_root = "prot_t5_last"
    for seq_id in targets.seq_ids:
        needs = False
        for root in _FAMILY_ROOTS[_T5_FAMILY]:
            if seq_id in targets.mean_targets[(_T5_FAMILY, root)] and not _artifact_path(
                targets.media_path, root, "mean", seq_id
            ).exists():
                needs = True
                break
            if seq_id in targets.weighted_targets[(_T5_FAMILY, root)]:
                weighted_exists = _artifact_path(targets.media_path, root, "weighted", seq_id).exists()
                residue_exists = _artifact_path(targets.media_path, root, "residue", seq_id).exists()
                if not weighted_exists and not residue_exists:
                    needs = True
                    break

        if not needs and seq_id in targets.binding_site_targets and seq_id not in bs_scores:
            if not _artifact_path(targets.media_path, t5_last_root, "residue", seq_id).exists():
                needs = True
        if needs:
            out.add(seq_id)
    return out


def _needs_esm_worker_seq_ids_file(
    *,
    family: str,
    targets: ArtifactTargets,
) -> set[str]:
    out: set[str] = set()
    for seq_id in targets.seq_ids:
        needs = False
        for root in _FAMILY_ROOTS[family]:
            if seq_id in targets.mean_targets[(family, root)] and not _artifact_path(
                targets.media_path, root, "mean", seq_id
            ).exists():
                needs = True
                break
            if seq_id in targets.weighted_targets[(family, root)]:
                weighted_exists = _artifact_path(targets.media_path, root, "weighted", seq_id).exists()
                residue_exists = _artifact_path(targets.media_path, root, "residue", seq_id).exists()
                if not weighted_exists and not residue_exists:
                    needs = True
                    break
        if needs:
            out.add(seq_id)
    return out


def _needs_pseq_worker_seq_ids(
    *,
    targets: ArtifactTargets,
    bs_scores: dict[str, np.ndarray],
) -> set[str]:
    return {seq_id for seq_id in targets.binding_site_targets if seq_id not in bs_scores}


def _run_kinform_parallel_pipeline_file_polling(
    *,
    env: dict[str, str],
    repo_root: Path,
    media_path: Path,
    seq_id_to_seq: dict[str, str],
    job_id: str | None = None,
) -> None:
    seq_ids = list(seq_id_to_seq.keys())
    if not seq_ids:
        _log(env, "info", "No sequence IDs provided to KinForm parallel pipeline.", job_id=job_id)
        return

    binding_sites_path = (media_path / "pseq2sites" / "binding_sites_all.tsv").resolve()
    targets = ArtifactTargets(
        seq_ids=seq_ids,
        media_path=media_path,
        binding_sites_path=binding_sites_path,
    )
    score_cache = BindingSiteScoreCache(binding_sites_path)
    weighted_retry_errors: dict[tuple[str, str, str], int] = {}

    scores = score_cache.read()
    if targets.all_done(scores):
        _cleanup_satisfied_residue_files(targets=targets, bs_scores=scores)
        _log(env, "info", "All KinForm artifacts already exist; skipping parallel pipeline.", job_id=job_id)
        return

    t5_script = (repo_root / "models" / "KinForm" / "code" / "protein_embeddings" / "t5_embeddings.py").resolve()
    prot_script = (repo_root / "models" / "KinForm" / "code" / "protein_embeddings" / "prot_embeddings.py").resolve()
    pseq_stream_script = (
        repo_root
        / "models"
        / "KinForm"
        / "code"
        / "pseq2sites"
        / "pseq2sites_stream_worker.py"
    ).resolve()
    pseq_retry_plan = _pseq_retry_batch_plan(env)
    t5_batch_size = max(1, _env_int("KINFORM_PARALLEL_T5_BATCH_SIZE", 1))
    esm2_batch_size = max(1, _env_int("KINFORM_PARALLEL_ESM2_BATCH_SIZE", 1))
    esmc_batch_size = max(1, _env_int("KINFORM_PARALLEL_ESMC_BATCH_SIZE", 1))
    max_gpu_workers = max(1, _env_int("KINFORM_PARALLEL_MAX_GPU_WORKERS", 3))
    include_pseq_in_gpu_cap = _env_bool("KINFORM_PARALLEL_INCLUDE_PSEQ_IN_GPU_CAP", True)

    workers: dict[str, WorkerState] = {
        _T5_FAMILY: WorkerState(name=_T5_FAMILY),
        _ESM2_FAMILY: WorkerState(name=_ESM2_FAMILY),
        _ESMC_FAMILY: WorkerState(name=_ESMC_FAMILY),
        _PSEQ_WORKER: WorkerState(name=_PSEQ_WORKER),
    }
    _log(
        env,
        "info",
        (
            "launch config "
            f"max_gpu_workers={max_gpu_workers} "
            f"include_pseq_in_gpu_cap={include_pseq_in_gpu_cap} "
            f"batch_t5={t5_batch_size} "
            f"batch_esm2={esm2_batch_size} "
            f"batch_esmc={esmc_batch_size}"
        ),
        job_id=job_id,
    )

    def needed_ids(worker_name: str, bs_scores: dict[str, np.ndarray]) -> set[str]:
        if worker_name == _T5_FAMILY:
            return _needs_t5_worker_seq_ids_file(targets=targets, bs_scores=bs_scores)
        if worker_name == _ESM2_FAMILY:
            return _needs_esm_worker_seq_ids_file(family=_ESM2_FAMILY, targets=targets)
        if worker_name == _ESMC_FAMILY:
            return _needs_esm_worker_seq_ids_file(family=_ESMC_FAMILY, targets=targets)
        if worker_name == _PSEQ_WORKER:
            return _needs_pseq_worker_seq_ids(targets=targets, bs_scores=bs_scores)
        return set()

    def build_cmd(
        worker_name: str,
        seq_subset: dict[str, str],
        seq_file: Path,
        id_to_seq_pkl: Path,
        seq_map_json: Path,
        *,
        pseq_batch_size: int | None = None,
    ) -> list[str]:
        if worker_name == _T5_FAMILY:
            return [
                env["KINFORM_T5_PATH"],
                str(t5_script),
                "--seq_file",
                str(seq_file),
                "--id_to_seq_file",
                str(id_to_seq_pkl),
                "--batch_size",
                str(t5_batch_size),
                "--setting",
                "residue+mean",
                "--layers",
                "19",
                "None",
            ]
        if worker_name == _ESM2_FAMILY:
            return [
                env["KINFORM_ESM_PATH"],
                str(prot_script),
                "--seq_file",
                str(seq_file),
                "--models",
                "esm2",
                "--layers",
                "26",
                "29",
                "--setting",
                "residue+mean",
                "--id_to_seq_file",
                str(id_to_seq_pkl),
                "--batch_size",
                str(esm2_batch_size),
            ]
        if worker_name == _ESMC_FAMILY:
            return [
                env["KINFORM_ESMC_PATH"],
                str(prot_script),
                "--seq_file",
                str(seq_file),
                "--models",
                "esmc",
                "--layers",
                "24",
                "32",
                "--setting",
                "residue+mean",
                "--id_to_seq_file",
                str(id_to_seq_pkl),
                "--batch_size",
                str(esmc_batch_size),
            ]
        if worker_name == _PSEQ_WORKER:
            if pseq_batch_size is None:
                pseq_batch_size = pseq_retry_plan[0]
            return [
                env["KINFORM_PSEQ2SITES_PATH"],
                str(pseq_stream_script),
                "--seq-id-to-seq-file",
                str(seq_map_json),
                "--binding-sites-path",
                str(binding_sites_path),
                "--poll-interval-seconds",
                "0.5",
                "--batch-size",
                str(max(1, int(pseq_batch_size))),
            ]
        raise RuntimeError(f"Unknown KinForm worker '{worker_name}'.")

    gpu_workers = (
        (_T5_FAMILY, _ESM2_FAMILY, _ESMC_FAMILY, _PSEQ_WORKER)
        if include_pseq_in_gpu_cap
        else (_T5_FAMILY, _ESM2_FAMILY, _ESMC_FAMILY)
    )
    launch_order = (_T5_FAMILY, _PSEQ_WORKER, _ESM2_FAMILY, _ESMC_FAMILY)

    def active_gpu_workers() -> int:
        return sum(1 for name in gpu_workers if workers[name].process is not None)

    def can_launch_worker(worker_name: str) -> bool:
        if worker_name not in gpu_workers:
            return True
        return active_gpu_workers() < max_gpu_workers

    def launch_worker(worker_name: str, bs_scores: dict[str, np.ndarray]) -> bool:
        state = workers[worker_name]
        if not can_launch_worker(worker_name):
            return False
        seq_ids_to_run = needed_ids(worker_name, bs_scores)
        if not seq_ids_to_run:
            return False
        max_attempts = len(pseq_retry_plan) if worker_name == _PSEQ_WORKER else 2
        if state.attempts >= max_attempts:
            raise RuntimeError(
                f"{worker_name} exhausted retries with remaining seq_ids={sorted(seq_ids_to_run)}"
            )
        seq_subset = _to_seq_subset(seq_id_to_seq, seq_ids_to_run)
        seq_file, id_to_seq_pkl, seq_map_json = _write_worker_inputs(seq_subset)
        state.tmp_inputs_dir = seq_file.parent
        pseq_batch_size = (
            pseq_retry_plan[state.attempts] if worker_name == _PSEQ_WORKER else None
        )
        cmd = build_cmd(
            worker_name,
            seq_subset,
            seq_file,
            id_to_seq_pkl,
            seq_map_json,
            pseq_batch_size=pseq_batch_size,
        )
        state.process = _start_worker(cmd, env)
        state.active_seq_ids = set(seq_ids_to_run)
        state.pseq_batch_size = pseq_batch_size
        state.attempts += 1
        pseq_batch_msg = (
            f" pseq_batch_size={pseq_batch_size}" if worker_name == _PSEQ_WORKER else ""
        )
        _log(
            env,
            "info",
            (
                f"launched worker={worker_name} attempt={state.attempts} "
                f"seq_count={len(seq_ids_to_run)}{pseq_batch_msg}"
            ),
            job_id=job_id,
        )
        return True

    def poll_worker(worker_name: str, bs_scores: dict[str, np.ndarray]) -> bool:
        state = workers[worker_name]
        if state.process is None:
            return False
        rc = state.process.poll()
        if rc is None:
            return False
        _cleanup_worker_inputs(state)
        state.process = None
        remaining = needed_ids(worker_name, bs_scores)
        if rc == 0 and not remaining:
            _log(env, "info", f"worker={worker_name} completed.", job_id=job_id)
            return True
        if rc != 0:
            _log(
                env,
                "warn",
                f"worker={worker_name} failed rc={rc}; remaining_seq_count={len(remaining)}",
                job_id=job_id,
            )
        else:
            _log(
                env,
                "warn",
                f"worker={worker_name} exited but artifacts still missing; remaining_seq_count={len(remaining)}",
                job_id=job_id,
            )
        max_attempts = len(pseq_retry_plan) if worker_name == _PSEQ_WORKER else 2
        if remaining and state.attempts < max_attempts:
            return launch_worker(worker_name, bs_scores)
        if remaining:
            raise RuntimeError(
                f"worker={worker_name} failed after retry; remaining seq_ids={sorted(remaining)}"
            )
        return True

    for name in launch_order:
        launch_worker(name, scores)

    poll_interval_seconds = 0.5
    progress_interval_seconds = 10.0
    last_progress_ts = 0.0

    try:
        while True:
            scores = score_cache.read()

            derived_weighted = 0
            derived_mean = 0
            for family, roots in _FAMILY_ROOTS.items():
                for root in roots:
                    for seq_id in seq_ids:
                        if _derive_mean_if_ready(
                            targets=targets,
                            family=family,
                            root=root,
                            seq_id=seq_id,
                        ):
                            derived_mean += 1
                        if _derive_weighted_if_ready(
                            targets=targets,
                            family=family,
                            root=root,
                            seq_id=seq_id,
                            bs_scores=scores,
                            weighted_retry_errors=weighted_retry_errors,
                        ):
                            derived_weighted += 1

            if derived_weighted or derived_mean:
                _log(
                    env,
                    "debug",
                    f"derived mean={derived_mean} weighted={derived_weighted} this iteration",
                    job_id=job_id,
                )

            scores = score_cache.read()
            removed_residue = _cleanup_satisfied_residue_files(
                targets=targets,
                bs_scores=scores,
            )
            if removed_residue:
                _log(
                    env,
                    "debug",
                    f"cleaned residue_files={removed_residue} after derived outputs completed",
                    job_id=job_id,
                )
            if targets.all_done(scores):
                _log(env, "info", "all target artifacts are complete.", job_id=job_id)
                break

            had_activity = bool(derived_weighted or derived_mean)
            for name in workers:
                if poll_worker(name, scores):
                    had_activity = True

            for name in launch_order:
                state = workers[name]
                if state.process is None:
                    if launch_worker(name, scores):
                        had_activity = True

            if all(state.process is None for state in workers.values()) and not had_activity:
                missing_fragments = []
                for family in (_T5_FAMILY, _ESM2_FAMILY, _ESMC_FAMILY):
                    mw, tw = targets.missing_weighted_count(family)
                    mm, tm = targets.missing_mean_count(family)
                    missing_fragments.append(f"{family}:weighted={mw}/{tw},mean={mm}/{tm}")
                bs_missing = len(_needs_pseq_worker_seq_ids(targets=targets, bs_scores=scores))
                missing_fragments.append(f"binding_sites_missing={bs_missing}/{len(targets.binding_site_targets)}")
                raise RuntimeError("KinForm parallel pipeline stalled: " + " ".join(missing_fragments))

            now = time.monotonic()
            if now - last_progress_ts >= progress_interval_seconds:
                last_progress_ts = now
                bs_ready = len(targets.binding_site_targets) - len(
                    _needs_pseq_worker_seq_ids(targets=targets, bs_scores=scores)
                )
                bs_total = len(targets.binding_site_targets)
                t5_w_missing, t5_w_total = targets.missing_weighted_count(_T5_FAMILY)
                esm2_w_missing, esm2_w_total = targets.missing_weighted_count(_ESM2_FAMILY)
                esmc_w_missing, esmc_w_total = targets.missing_weighted_count(_ESMC_FAMILY)
                _log(
                    env,
                    "info",
                    (
                        f"progress bs={bs_ready}/{bs_total} "
                        f"weighted_t5={t5_w_total - t5_w_missing}/{t5_w_total} "
                        f"weighted_esm2={esm2_w_total - esm2_w_missing}/{esm2_w_total} "
                        f"weighted_esmc={esmc_w_total - esmc_w_missing}/{esmc_w_total}"
                    ),
                    job_id=job_id,
                )

            time.sleep(poll_interval_seconds)
    finally:
        for state in workers.values():
            _terminate_worker(state)
            _cleanup_worker_inputs(state)


def _safe_job_slug(job_id: str | None) -> str:
    if not job_id:
        return "job"
    return "".join(ch if ch.isalnum() else "_" for ch in job_id)[:32] or "job"


def _decode_array(header: dict, payload: bytes) -> np.ndarray:
    dtype = np.dtype(str(header.get("dtype", "float32")))
    shape_raw = header.get("shape")
    if not isinstance(shape_raw, list) or not shape_raw:
        raise ValueError("Invalid shape in stream header.")
    shape = tuple(int(x) for x in shape_raw)
    arr = np.frombuffer(payload, dtype=dtype)
    expected = int(np.prod(shape))
    if arr.size != expected:
        raise ValueError(f"Payload element count mismatch: got {arr.size}, expected {expected}")
    return arr.reshape(shape).astype(np.float32, copy=False)


def _stream_all_done(
    targets: ArtifactTargets,
    bs_scores: dict[str, np.ndarray],
    submitted_paths: set[Path],
) -> bool:
    for (family, root), missing_ids in targets.weighted_targets.items():
        for seq_id in missing_ids:
            if _artifact_path(targets.media_path, root, "weighted", seq_id) not in submitted_paths:
                return False
    for (family, root), missing_ids in targets.mean_targets.items():
        for seq_id in missing_ids:
            if _artifact_path(targets.media_path, root, "mean", seq_id) not in submitted_paths:
                return False
    for seq_id in targets.binding_site_targets:
        if seq_id not in bs_scores:
            return False
    return True


def _stream_missing_weighted_count(
    targets: ArtifactTargets, family: str, submitted_paths: set[Path]
) -> tuple[int, int]:
    missing, total = 0, 0
    for (fam, root), target_ids in targets.weighted_targets.items():
        if fam != family:
            continue
        total += len(target_ids)
        for seq_id in target_ids:
            if _artifact_path(targets.media_path, root, "weighted", seq_id) not in submitted_paths:
                missing += 1
    return missing, total


def _stream_missing_mean_count(
    targets: ArtifactTargets, family: str, submitted_paths: set[Path]
) -> tuple[int, int]:
    missing, total = 0, 0
    for (fam, root), target_ids in targets.mean_targets.items():
        if fam != family:
            continue
        total += len(target_ids)
        for seq_id in target_ids:
            if _artifact_path(targets.media_path, root, "mean", seq_id) not in submitted_paths:
                missing += 1
    return missing, total


def _run_kinform_parallel_pipeline_stream(
    *,
    env: dict[str, str],
    repo_root: Path,
    media_path: Path,
    seq_id_to_seq: dict[str, str],
    job_id: str | None,
) -> None:
    pipeline_started_at = time.monotonic()
    if torch is None:
        raise RuntimeError("torch is required for KinForm stream mode.")
    seq_ids = list(seq_id_to_seq.keys())
    if not seq_ids:
        _log(env, "info", "No sequence IDs provided to KinForm stream pipeline.", job_id=job_id)
        return

    require_cuda = _env_bool("KINFORM_REQUIRE_CUDA", True)
    if require_cuda and not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for KinForm stream mode (KINFORM_REQUIRE_CUDA=1).")

    binding_sites_path = (media_path / "pseq2sites" / "binding_sites_all.tsv").resolve()
    binding_score_load_workers = max(1, int(_env_float("KINFORM_PARALLEL_BS_CACHE_LOAD_WORKERS", 16.0)))
    bs_scores: dict[str, np.ndarray] = _load_binding_score_cache(
        media_path=media_path,
        seq_ids=seq_ids,
        max_workers=binding_score_load_workers,
    )
    targets = ArtifactTargets(
        seq_ids=seq_ids,
        media_path=media_path,
        binding_sites_path=binding_sites_path,
        binding_site_existing_ids=set(bs_scores.keys()),
    )
    tsv_refresh_enabled = _env_bool("KINFORM_PARALLEL_STREAM_TSV_REFRESH_ENABLE", False)
    score_cache = (
        BindingSiteScoreCache(
            binding_sites_path,
            target_seq_ids=targets.binding_site_targets,
        )
        if tsv_refresh_enabled
        else None
    )
    if score_cache is not None and targets.binding_site_targets:
        for sid, scores in score_cache.read().items():
            if sid not in bs_scores:
                bs_scores[sid] = scores.astype(np.float32, copy=False)

    if _stream_all_done(targets, bs_scores, set()):
        _log(env, "info", "All KinForm artifacts already exist; skipping stream pipeline.", job_id=job_id)
        return

    async_write_workers = max(1, int(_env_float("KINFORM_PARALLEL_ASYNC_WRITE_WORKERS", 12.0)))
    async_writer = AsyncWriter(max_workers=async_write_workers)
    submitted_paths: set[Path] = set()
    submitted_binding_score_paths: set[Path] = {
        _binding_score_path(media_path, sid) for sid in bs_scores
    }

    max_gb = _env_float("KINFORM_PARALLEL_RESIDUE_CACHE_GB", 4.0)
    max_bytes = int(max_gb * 1024 * 1024 * 1024)
    spill_base = Path(env.get("KINFORM_PARALLEL_SPILL_DIR", "/dev/shm/webkinpred-kinform")).resolve()
    spill_fallback_base = Path(
        env.get("KINFORM_PARALLEL_SPILL_FALLBACK_DIR", "/tmp/webkinpred-kinform")
    ).resolve()
    job_slug = _safe_job_slug(job_id)
    residue_cache = ResidueCache(
        max_bytes=max_bytes,
        spill_dir=spill_base / job_slug,
        spill_fallback_dir=spill_fallback_base / job_slug,
        env=env,
        job_id=job_id,
    )

    socket_dir = Path(env.get("KINFORM_PARALLEL_STREAM_SOCKET_DIR", "/tmp/webkinpred-gpu-embed/kinform")).resolve()
    socket_name = f"{job_slug}_{os.getpid()}_{int(time.time() * 1000) % 1000000}.sock"
    socket_path = socket_dir / socket_name
    server = StreamEventServer(socket_path)
    server.start()

    t5_script = (repo_root / "models" / "KinForm" / "code" / "protein_embeddings" / "t5_embeddings.py").resolve()
    prot_script = (repo_root / "models" / "KinForm" / "code" / "protein_embeddings" / "prot_embeddings.py").resolve()
    pseq_stream_script = (
        repo_root
        / "models"
        / "KinForm"
        / "code"
        / "pseq2sites"
        / "pseq2sites_stream_worker.py"
    ).resolve()
    pseq_retry_plan = _pseq_retry_batch_plan(env)
    t5_batch_size = max(
        1,
        _env_int(
            "KINFORM_PARALLEL_STREAM_T5_BATCH_SIZE",
            _env_int("KINFORM_PARALLEL_T5_BATCH_SIZE", 4),
        ),
    )
    esm2_batch_size = max(
        1,
        _env_int(
            "KINFORM_PARALLEL_STREAM_ESM2_BATCH_SIZE",
            _env_int("KINFORM_PARALLEL_ESM2_BATCH_SIZE", 4),
        ),
    )
    esmc_batch_size = max(
        1,
        _env_int(
            "KINFORM_PARALLEL_STREAM_ESMC_BATCH_SIZE",
            _env_int("KINFORM_PARALLEL_ESMC_BATCH_SIZE", 4),
        ),
    )
    max_gpu_workers = max(1, _env_int("KINFORM_PARALLEL_MAX_GPU_WORKERS", 3))
    include_pseq_in_gpu_cap = _env_bool("KINFORM_PARALLEL_INCLUDE_PSEQ_IN_GPU_CAP", True)

    workers: dict[str, WorkerState] = {
        _T5_FAMILY: WorkerState(name=_T5_FAMILY),
        _ESM2_FAMILY: WorkerState(name=_ESM2_FAMILY),
        _ESMC_FAMILY: WorkerState(name=_ESMC_FAMILY),
        _PSEQ_WORKER: WorkerState(name=_PSEQ_WORKER),
    }
    _log(
        env,
        "info",
        (
            "stream launch config "
            f"max_gpu_workers={max_gpu_workers} "
            f"include_pseq_in_gpu_cap={include_pseq_in_gpu_cap} "
            f"batch_t5={t5_batch_size} "
            f"batch_esm2={esm2_batch_size} "
            f"batch_esmc={esmc_batch_size}"
        ),
        job_id=job_id,
    )
    weighted_retry_errors: dict[tuple[str, str, str], int] = {}
    weighted_compute_count_total = 0
    weighted_compute_seconds_total = 0.0
    weighted_compute_count_by_family: dict[str, int] = {
        _T5_FAMILY: 0,
        _ESM2_FAMILY: 0,
        _ESMC_FAMILY: 0,
    }
    weighted_compute_seconds_by_family: dict[str, float] = {
        _T5_FAMILY: 0.0,
        _ESM2_FAMILY: 0.0,
        _ESMC_FAMILY: 0.0,
    }
    first_bs_ready_at: float | None = None
    first_weighted_ready_at: float | None = None

    pseq_client_id: int | None = None
    pseq_client_lock = threading.Lock()
    sent_to_pseq: set[str] = set()
    queued_to_pseq: set[str] = set()
    pseq_sends_per_tick = max(1, int(_env_float("KINFORM_PARALLEL_PSEQ_SENDS_PER_TICK", 10000.0)))
    pseq_send_queue_size = max(4, int(_env_float("KINFORM_PARALLEL_PSEQ_SEND_QUEUE_SIZE", 256.0)))
    pseq_send_queue: queue.Queue[tuple[str, dict, bytes] | None] = queue.Queue(maxsize=pseq_send_queue_size)
    pseq_send_results: queue.Queue[tuple[str, str, str]] = queue.Queue()
    pseq_sender_stop = threading.Event()
    score_refresh_seconds = (
        max(1.0, _env_float("KINFORM_PARALLEL_TSV_REFRESH_SECONDS", 30.0))
        if tsv_refresh_enabled
        else 0.0
    )
    last_score_refresh_at = 0.0
    stream_recv_timeout_seconds = max(
        0.01, _env_float("KINFORM_PARALLEL_STREAM_RECV_TIMEOUT_SECONDS", 0.05)
    )
    max_events_per_tick = max(
        16, int(_env_float("KINFORM_PARALLEL_STREAM_MAX_EVENTS_PER_TICK", 512.0))
    )
    worker_done_wait_seconds = max(1.0, _env_float("KINFORM_PARALLEL_WORKER_DONE_WAIT_SECONDS", 30.0))

    def get_pseq_client_id() -> int | None:
        with pseq_client_lock:
            return pseq_client_id

    def set_pseq_client_id(value: int | None) -> None:
        nonlocal pseq_client_id
        with pseq_client_lock:
            pseq_client_id = value

    def reset_pseq_send_state() -> None:
        sent_to_pseq.clear()
        queued_to_pseq.clear()
        while True:
            try:
                pseq_send_queue.get_nowait()
            except queue.Empty:
                break
        while True:
            try:
                pseq_send_results.get_nowait()
            except queue.Empty:
                break

    def drain_pseq_send_results() -> bool:
        had_update = False
        while True:
            try:
                status, seq_id, detail = pseq_send_results.get_nowait()
            except queue.Empty:
                break
            had_update = True
            queued_to_pseq.discard(seq_id)
            if status == "sent":
                sent_to_pseq.add(seq_id)
                continue
            sent_to_pseq.discard(seq_id)
            if detail:
                _log(
                    env,
                    "debug",
                    f"pseq send retry seq_id={seq_id} reason={detail}",
                    job_id=job_id,
                )
        return had_update

    def _pseq_sender_loop() -> None:
        while not pseq_sender_stop.is_set():
            try:
                item = pseq_send_queue.get(timeout=0.2)
            except queue.Empty:
                continue
            if item is None:
                return
            seq_id, header, payload = item
            if pseq_sender_stop.is_set():
                return
            client_id = get_pseq_client_id()
            if client_id is None:
                pseq_send_results.put(("retry", seq_id, "pseq client not connected"))
                continue
            try:
                server.send(client_id, header, payload)
                pseq_send_results.put(("sent", seq_id, ""))
            except Exception as exc:
                pseq_send_results.put(("retry", seq_id, str(exc)))

    pseq_sender_thread = threading.Thread(
        target=_pseq_sender_loop,
        name=f"kinform-pseq-sender-{job_id or 'job'}",
        daemon=True,
    )
    pseq_sender_thread.start()

    def has_residue(family: str, root: str, seq_id: str) -> bool:
        return residue_cache.has((family, root, seq_id))

    def needed_ids(worker_name: str) -> set[str]:
        if worker_name == _PSEQ_WORKER:
            return {sid for sid in targets.binding_site_targets if sid not in bs_scores}

        if worker_name == _T5_FAMILY:
            out: set[str] = set()
            for sid in seq_ids:
                needs = False
                for root in _FAMILY_ROOTS[_T5_FAMILY]:
                    mean_path = _artifact_path(media_path, root, "mean", sid)
                    weighted_path = _artifact_path(media_path, root, "weighted", sid)
                    mean_missing = (
                        sid in targets.mean_targets[(_T5_FAMILY, root)]
                        and mean_path not in submitted_paths
                    )
                    weighted_missing = (
                        sid in targets.weighted_targets[(_T5_FAMILY, root)]
                        and weighted_path not in submitted_paths
                    )
                    if mean_missing or (weighted_missing and not has_residue(_T5_FAMILY, root, sid)):
                        needs = True
                        break
                if not needs and sid in targets.binding_site_targets and sid not in bs_scores:
                    if not has_residue(_T5_FAMILY, "prot_t5_last", sid):
                        needs = True
                if needs:
                    out.add(sid)
            return out

        family = _ESM2_FAMILY if worker_name == _ESM2_FAMILY else _ESMC_FAMILY
        out: set[str] = set()
        for sid in seq_ids:
            needs = False
            for root in _FAMILY_ROOTS[family]:
                mean_path = _artifact_path(media_path, root, "mean", sid)
                weighted_path = _artifact_path(media_path, root, "weighted", sid)
                mean_missing = (
                    sid in targets.mean_targets[(family, root)]
                    and mean_path not in submitted_paths
                )
                weighted_missing = (
                    sid in targets.weighted_targets[(family, root)]
                    and weighted_path not in submitted_paths
                )
                if mean_missing or (weighted_missing and not has_residue(family, root, sid)):
                    needs = True
                    break
            if needs:
                out.add(sid)
        return out

    def build_cmd(
        worker_name: str,
        seq_subset: dict[str, str],
        seq_file: Path,
        id_to_seq_pkl: Path,
        seq_map_json: Path,
        *,
        pseq_batch_size: int | None = None,
    ) -> list[str]:
        common_stream = [
            "--stream-mode",
            "--stream-socket",
            str(socket_path),
            "--stream-job-id",
            str(job_id or ""),
            "--worker-name",
            worker_name,
        ]
        if _env_bool("KINFORM_PARALLEL_STREAM_LEGACY_RESIDUE_WRITE", False):
            common_stream.append("--legacy-residue-write")

        if worker_name == _T5_FAMILY:
            return [
                env["KINFORM_T5_PATH"],
                str(t5_script),
                "--seq_file",
                str(seq_file),
                "--id_to_seq_file",
                str(id_to_seq_pkl),
                "--batch_size",
                str(t5_batch_size),
                "--setting",
                "residue+mean",
                "--layers",
                "19",
                "None",
                *common_stream,
            ]
        if worker_name == _ESM2_FAMILY:
            return [
                env["KINFORM_ESM_PATH"],
                str(prot_script),
                "--seq_file",
                str(seq_file),
                "--models",
                "esm2",
                "--layers",
                "26",
                "29",
                "--setting",
                "residue+mean",
                "--id_to_seq_file",
                str(id_to_seq_pkl),
                "--batch_size",
                str(esm2_batch_size),
                *common_stream,
            ]
        if worker_name == _ESMC_FAMILY:
            return [
                env["KINFORM_ESMC_PATH"],
                str(prot_script),
                "--seq_file",
                str(seq_file),
                "--models",
                "esmc",
                "--layers",
                "24",
                "32",
                "--setting",
                "residue+mean",
                "--id_to_seq_file",
                str(id_to_seq_pkl),
                "--batch_size",
                str(esmc_batch_size),
                *common_stream,
            ]
        if worker_name == _PSEQ_WORKER:
            if pseq_batch_size is None:
                pseq_batch_size = pseq_retry_plan[0]
            return [
                env["KINFORM_PSEQ2SITES_PATH"],
                str(pseq_stream_script),
                "--seq-id-to-seq-file",
                str(seq_map_json),
                "--binding-sites-path",
                str(binding_sites_path),
                "--batch-size",
                str(max(1, int(pseq_batch_size))),
                "--stream-mode",
                "--stream-socket",
                str(socket_path),
                "--stream-job-id",
                str(job_id or ""),
                "--worker-name",
                worker_name,
            ]
        raise RuntimeError(f"Unknown KinForm worker '{worker_name}'.")

    gpu_workers = (
        (_T5_FAMILY, _ESM2_FAMILY, _ESMC_FAMILY, _PSEQ_WORKER)
        if include_pseq_in_gpu_cap
        else (_T5_FAMILY, _ESM2_FAMILY, _ESMC_FAMILY)
    )
    launch_order = (_T5_FAMILY, _PSEQ_WORKER, _ESM2_FAMILY, _ESMC_FAMILY)

    def active_gpu_workers() -> int:
        return sum(1 for name in gpu_workers if workers[name].process is not None)

    def can_launch_worker(worker_name: str) -> bool:
        if worker_name not in gpu_workers:
            return True
        return active_gpu_workers() < max_gpu_workers

    def launch_worker(worker_name: str) -> bool:
        state = workers[worker_name]
        if not can_launch_worker(worker_name):
            return False
        seq_ids_to_run = needed_ids(worker_name)
        if not seq_ids_to_run:
            return False
        max_attempts = len(pseq_retry_plan) if worker_name == _PSEQ_WORKER else 2
        if state.attempts >= max_attempts:
            raise RuntimeError(
                f"{worker_name} exhausted retries with remaining seq_ids={sorted(seq_ids_to_run)}"
            )
        seq_subset = _to_seq_subset(seq_id_to_seq, seq_ids_to_run)
        seq_file, id_to_seq_pkl, seq_map_json = _write_worker_inputs(seq_subset)
        state.tmp_inputs_dir = seq_file.parent
        pseq_batch_size = (
            pseq_retry_plan[state.attempts] if worker_name == _PSEQ_WORKER else None
        )
        cmd = build_cmd(
            worker_name,
            seq_subset,
            seq_file,
            id_to_seq_pkl,
            seq_map_json,
            pseq_batch_size=pseq_batch_size,
        )
        state.process = _start_worker(cmd, env)
        state.active_seq_ids = set(seq_ids_to_run)
        state.active_seq_count = len(seq_ids_to_run)
        state.started_at_monotonic = time.monotonic()
        state.stream_done_received = False
        state.waiting_for_stream_done_since = None
        state.pseq_batch_size = pseq_batch_size
        if worker_name == _PSEQ_WORKER:
            # New Pseq2Sites process needs fresh residue dispatch for any unresolved IDs.
            set_pseq_client_id(None)
            reset_pseq_send_state()
        state.attempts += 1
        pseq_batch_msg = (
            f" pseq_batch_size={pseq_batch_size}" if worker_name == _PSEQ_WORKER else ""
        )
        _log(
            env,
            "info",
            (
                f"launched worker={worker_name} attempt={state.attempts} "
                f"seq_count={len(seq_ids_to_run)}{pseq_batch_msg}"
            ),
            job_id=job_id,
        )
        return True

    def poll_worker(worker_name: str) -> bool:
        state = workers[worker_name]
        if state.process is None:
            return False
        rc = state.process.poll()
        if rc is None:
            return False

        if rc == 0 and not state.stream_done_received:
            now = time.monotonic()
            if state.waiting_for_stream_done_since is None:
                state.waiting_for_stream_done_since = now
                _log(
                    env,
                    "debug",
                    f"worker={worker_name} exited rc=0; waiting for WORKER_DONE event before retry checks.",
                    job_id=job_id,
                )
                return False
            waited_s = max(0.0, now - state.waiting_for_stream_done_since)
            if waited_s < worker_done_wait_seconds:
                return False
            _log(
                env,
                "warn",
                (
                    f"worker={worker_name} rc=0 but WORKER_DONE not seen after {waited_s:.3f}s; "
                    "continuing completion checks."
                ),
                job_id=job_id,
            )

        elapsed_s = 0.0
        if state.started_at_monotonic is not None:
            elapsed_s = max(0.0, time.monotonic() - state.started_at_monotonic)
            state.elapsed_seconds_total += elapsed_s
        stage_name = _timing_stage_for_worker(worker_name)
        _log(
            env,
            "info",
            (
                f"KINFORM_TIMING stage={stage_name} "
                f"attempt={state.attempts} seq_count={state.active_seq_count} "
                f"attempt_elapsed_s={elapsed_s:.3f} rc={rc}"
            ),
            job_id=job_id,
        )
        state.started_at_monotonic = None
        state.active_seq_count = 0
        state.waiting_for_stream_done_since = None

        _cleanup_worker_inputs(state)
        state.process = None

        remaining = needed_ids(worker_name)
        if rc == 0 and not remaining:
            _log(env, "info", f"worker={worker_name} completed.", job_id=job_id)
            return True

        if rc != 0:
            _log(
                env,
                "warn",
                f"worker={worker_name} failed rc={rc}; remaining_seq_count={len(remaining)}",
                job_id=job_id,
            )
        else:
            _log(
                env,
                "warn",
                f"worker={worker_name} exited but artifacts still missing; remaining_seq_count={len(remaining)}",
                job_id=job_id,
            )

        max_attempts = len(pseq_retry_plan) if worker_name == _PSEQ_WORKER else 2
        if remaining and state.attempts < max_attempts:
            return launch_worker(worker_name)

        if remaining:
            raise RuntimeError(
                f"worker={worker_name} failed after retry; remaining seq_ids={sorted(remaining)}"
            )
        return True

    def try_queue_t5_to_pseq(seq_id: str) -> bool:
        if seq_id in sent_to_pseq or seq_id in queued_to_pseq:
            return False
        if seq_id not in targets.binding_site_targets or seq_id in bs_scores:
            return False
        key: ResidueKey = (_T5_FAMILY, "prot_t5_last", seq_id)
        if not residue_cache.has(key):
            return False

        arr = residue_cache.get_numpy(key)
        payload = arr.astype(np.float32, copy=False).tobytes(order="C")
        header = {
            "type": "PSEQ_RESIDUE",
            "job_id": job_id or "",
            "seq_id": seq_id,
            "sequence": seq_id_to_seq.get(seq_id, ""),
            "dtype": "float32",
            "shape": [int(x) for x in arr.shape],
        }
        try:
            pseq_send_queue.put_nowait((seq_id, header, payload))
        except queue.Full:
            return False
        queued_to_pseq.add(seq_id)
        return True

    def attempt_weighted_for_seq(seq_id: str) -> int:
        nonlocal weighted_compute_count_total
        nonlocal weighted_compute_seconds_total
        nonlocal first_weighted_ready_at
        derived = 0
        for family, roots in _FAMILY_ROOTS.items():
            for root in roots:
                if seq_id not in targets.weighted_targets.get((family, root), set()):
                    continue
                weighted_path = _artifact_path(media_path, root, "weighted", seq_id)
                if weighted_path in submitted_paths:
                    continue
                key: ResidueKey = (family, root, seq_id)
                if not residue_cache.has(key):
                    continue
                weights = bs_scores.get(seq_id)
                if weights is None:
                    continue
                started = time.monotonic()
                try:
                    residue_arr = residue_cache.get_numpy(key)
                    weighted_vec = weighted_mean_from_residue(residue_arr, weights)
                    residue_cache.remove(key)
                    submitted_paths.add(weighted_path)
                    async_writer.submit(weighted_path, weighted_vec)
                    weighted_retry_errors.pop((family, root, seq_id), None)
                    elapsed = max(0.0, time.monotonic() - started)
                    derived += 1
                    weighted_compute_count_total += 1
                    weighted_compute_seconds_total += elapsed
                    weighted_compute_count_by_family[family] = weighted_compute_count_by_family.get(family, 0) + 1
                    weighted_compute_seconds_by_family[family] = (
                        weighted_compute_seconds_by_family.get(family, 0.0) + elapsed
                    )
                    if first_weighted_ready_at is None:
                        first_weighted_ready_at = time.monotonic()
                except Exception:
                    weighted_retry_errors[key] = weighted_retry_errors.get(key, 0) + 1
                    if weighted_retry_errors.get(key, 0) > 1:
                        raise
        return derived

    # Initial launch (subject to max_gpu_workers cap).
    for name in launch_order:
        launch_worker(name)

    progress_interval_seconds = 10.0
    last_progress_ts = 0.0

    def _event_priority(item: tuple[str, int, dict | None, bytes | None]) -> tuple[int, int]:
        kind, _client_id, header, _payload = item
        if kind != "event" or header is None:
            return (0, 0)
        evt_type = str(header.get("type", "")).strip().upper()
        # Keep RESIDUE_READY and WORKER_DONE at the same priority so per-worker
        # ordering from the socket queue is preserved (WORKER_DONE is emitted last).
        # This avoids premature completion checks that can trigger false retries.
        if evt_type in {"RESIDUE_READY", "WORKER_DONE"}:
            return (1, 0)
        return (0, 0)

    try:
        while True:
            had_activity = False
            derived_weighted = 0
            if drain_pseq_send_results():
                had_activity = True

            event = server.recv_event(timeout_seconds=stream_recv_timeout_seconds)
            pending_events: list[tuple[str, int, dict | None, bytes | None]] = []
            if event is not None:
                pending_events.append(event)
            if len(pending_events) < max_events_per_tick:
                pending_events.extend(
                    server.drain_events(max_items=max_events_per_tick - len(pending_events))
                )
            pending_events.sort(key=_event_priority)

            for kind, client_id, header, payload in pending_events:
                had_activity = True
                if kind == "disconnect":
                    if get_pseq_client_id() == client_id:
                        set_pseq_client_id(None)
                        _log(env, "warn", f"pseq stream client disconnected id={client_id}", job_id=job_id)
                    continue

                if kind == "error":
                    _log(env, "warn", f"stream client={client_id} error={header}", job_id=job_id)
                    continue

                if kind != "event" or header is None:
                    continue

                evt_type = str(header.get("type", "")).strip().upper()
                if evt_type == "PSEQ_REGISTER":
                    set_pseq_client_id(client_id)
                    _log(env, "info", f"pseq stream client registered id={client_id}", job_id=job_id)
                    continue

                if evt_type == "WORKER_ERROR":
                    worker_name = str(header.get("worker", "unknown"))
                    msg = str(header.get("message", "worker reported error"))
                    _log(env, "warn", f"worker={worker_name} stream error={msg}", job_id=job_id)
                    continue

                if evt_type == "WORKER_DONE":
                    worker_name = str(header.get("worker", "unknown"))
                    state = workers.get(worker_name)
                    if state is not None:
                        state.stream_done_received = True
                    _log(env, "debug", f"worker={worker_name} emitted WORKER_DONE", job_id=job_id)
                    continue

                if evt_type == "RESIDUE_READY":
                    family = str(header.get("family", "")).strip().lower()
                    root = str(header.get("root", "")).strip()
                    seq_id = str(header.get("seq_id", "")).strip()
                    if not family or not root or not seq_id:
                        raise RuntimeError(f"Invalid RESIDUE_READY header: {header}")
                    if payload is None:
                        raise RuntimeError("RESIDUE_READY missing payload")

                    arr = _decode_array(header, payload)
                    key: ResidueKey = (family, root, seq_id)
                    residue_cache.put(key, arr)

                    mean_path = _artifact_path(media_path, root, "mean", seq_id)
                    if (mean_path not in submitted_paths
                            and seq_id in targets.mean_targets.get((family, root), set())):
                        mean_vec = arr.mean(axis=0).astype(np.float32)
                        submitted_paths.add(mean_path)
                        async_writer.submit(mean_path, mean_vec)

                    if family == _T5_FAMILY and root == "prot_t5_last":
                        try_queue_t5_to_pseq(seq_id)

                    if seq_id in bs_scores:
                        derived_weighted += attempt_weighted_for_seq(seq_id)
                    continue

                if evt_type == "BS_READY":
                    seq_id = str(header.get("seq_id", "")).strip()
                    if not seq_id:
                        raise RuntimeError(f"Invalid BS_READY header: {header}")
                    if payload is None:
                        raise RuntimeError("BS_READY missing payload")
                    if first_bs_ready_at is None:
                        first_bs_ready_at = time.monotonic()
                    weights_arr = _decode_array(header, payload).reshape(-1).astype(np.float32, copy=False)
                    bs_scores[seq_id] = weights_arr
                    bs_score_path = _binding_score_path(media_path, seq_id)
                    if bs_score_path not in submitted_binding_score_paths:
                        submitted_binding_score_paths.add(bs_score_path)
                        async_writer.submit(bs_score_path, weights_arr)
                    queued_to_pseq.discard(seq_id)
                    sent_to_pseq.add(seq_id)
                    derived_weighted += attempt_weighted_for_seq(seq_id)
                    continue

            # Keep scores refreshed from shared TSV for resilience/resume, but avoid
            # re-reading it every loop (it can be very large).
            now_refresh = time.monotonic()
            if (
                score_cache is not None
                and score_refresh_seconds > 0.0
                and now_refresh - last_score_refresh_at >= score_refresh_seconds
            ):
                last_score_refresh_at = now_refresh
                tsv_scores = score_cache.read()
                for sid, w in tsv_scores.items():
                    if sid not in bs_scores:
                        bs_scores[sid] = w.astype(np.float32, copy=False)

            # Send any queued T5-last residues once pseq client is connected.
            if get_pseq_client_id() is not None:
                sent_this_tick = 0
                for sid in list(needed_ids(_PSEQ_WORKER)):
                    if try_queue_t5_to_pseq(sid):
                        had_activity = True
                        sent_this_tick += 1
                        if sent_this_tick >= pseq_sends_per_tick:
                            break

            # Opportunistically derive weighted vectors for all ids that already have BS.
            for sid in seq_ids:
                if sid in bs_scores:
                    derived_weighted += attempt_weighted_for_seq(sid)

            if derived_weighted:
                had_activity = True
                _log(env, "debug", f"derived weighted={derived_weighted} this iteration", job_id=job_id)

            if _stream_all_done(targets, bs_scores, submitted_paths):
                current_pseq_client = get_pseq_client_id()
                if current_pseq_client is not None:
                    try:
                        server.send(
                            current_pseq_client,
                            {
                                "type": "PSEQ_FINISH",
                                "job_id": job_id or "",
                            },
                            b"",
                        )
                    except Exception:
                        pass
                _log(env, "info", "all target artifacts are complete.", job_id=job_id)
                break

            for name in workers:
                if poll_worker(name):
                    had_activity = True

            for name in launch_order:
                state = workers[name]
                if state.process is None:
                    if launch_worker(name):
                        had_activity = True

            if all(state.process is None for state in workers.values()) and not had_activity:
                missing_fragments = []
                for family in (_T5_FAMILY, _ESM2_FAMILY, _ESMC_FAMILY):
                    mw, tw = _stream_missing_weighted_count(targets, family, submitted_paths)
                    mm, tm = _stream_missing_mean_count(targets, family, submitted_paths)
                    missing_fragments.append(f"{family}:weighted={mw}/{tw},mean={mm}/{tm}")
                bs_missing = len({sid for sid in targets.binding_site_targets if sid not in bs_scores})
                missing_fragments.append(f"binding_sites_missing={bs_missing}/{len(targets.binding_site_targets)}")
                missing_fragments.append(f"residue_cache_keys={len(tuple(residue_cache.keys()))}")
                raise RuntimeError("KinForm stream pipeline stalled: " + " ".join(missing_fragments))

            now = time.monotonic()
            if now - last_progress_ts >= progress_interval_seconds:
                last_progress_ts = now
                bs_ready = len(targets.binding_site_targets) - len(
                    {sid for sid in targets.binding_site_targets if sid not in bs_scores}
                )
                bs_total = len(targets.binding_site_targets)
                t5_w_missing, t5_w_total = _stream_missing_weighted_count(targets, _T5_FAMILY, submitted_paths)
                esm2_w_missing, esm2_w_total = _stream_missing_weighted_count(targets, _ESM2_FAMILY, submitted_paths)
                esmc_w_missing, esmc_w_total = _stream_missing_weighted_count(targets, _ESMC_FAMILY, submitted_paths)
                _log(
                    env,
                    "info",
                    (
                        f"progress bs={bs_ready}/{bs_total} "
                        f"weighted_t5={t5_w_total - t5_w_missing}/{t5_w_total} "
                        f"weighted_esm2={esm2_w_total - esm2_w_missing}/{esm2_w_total} "
                        f"weighted_esmc={esmc_w_total - esmc_w_missing}/{esmc_w_total} "
                        f"weighted_calc_count={weighted_compute_count_total} "
                        f"weighted_calc_s={weighted_compute_seconds_total:.3f} "
                        f"weighted_calc_avg_ms={(weighted_compute_seconds_total * 1000.0 / weighted_compute_count_total) if weighted_compute_count_total else 0.0:.3f} "
                        f"pseq_sent={len(sent_to_pseq)} "
                        f"pseq_queued={len(queued_to_pseq)} "
                        f"pseq_queue_depth={pseq_send_queue.qsize()} "
                        f"residue_cache_gb={residue_cache.resident_bytes / (1024 ** 3):.3f}"
                    ),
                    job_id=job_id,
                )
    finally:
        total_elapsed_s = max(0.0, time.monotonic() - pipeline_started_at)
        _log(
            env,
            "info",
            f"KINFORM_TIMING stage=pipeline_total elapsed_s={total_elapsed_s:.3f}",
            job_id=job_id,
        )
        if first_bs_ready_at is not None:
            _log(
                env,
                "info",
                (
                    "KINFORM_TIMING stage=pseq2sites_preds "
                    f"first_bs_ready_latency_s={max(0.0, first_bs_ready_at - pipeline_started_at):.3f}"
                ),
                job_id=job_id,
            )
        if first_weighted_ready_at is not None:
            _log(
                env,
                "info",
                (
                    "KINFORM_TIMING stage=weighted_average "
                    f"first_weighted_latency_s={max(0.0, first_weighted_ready_at - pipeline_started_at):.3f}"
                ),
                job_id=job_id,
            )
        _log(
            env,
            "info",
            (
                "KINFORM_TIMING stage=weighted_average "
                f"count_total={weighted_compute_count_total} "
                f"elapsed_s={weighted_compute_seconds_total:.3f} "
                f"avg_ms={(weighted_compute_seconds_total * 1000.0 / weighted_compute_count_total) if weighted_compute_count_total else 0.0:.3f} "
                f"count_t5={weighted_compute_count_by_family.get(_T5_FAMILY, 0)} "
                f"count_esm2={weighted_compute_count_by_family.get(_ESM2_FAMILY, 0)} "
                f"count_esmc={weighted_compute_count_by_family.get(_ESMC_FAMILY, 0)}"
            ),
            job_id=job_id,
        )
        for worker_name, state in workers.items():
            in_flight_s = 0.0
            if state.started_at_monotonic is not None:
                in_flight_s = max(0.0, time.monotonic() - state.started_at_monotonic)
            elapsed_total_s = state.elapsed_seconds_total + in_flight_s
            _log(
                env,
                "info",
                (
                    f"KINFORM_TIMING stage={_timing_stage_for_worker(worker_name)} "
                    f"attempts={state.attempts} elapsed_total_s={elapsed_total_s:.3f}"
                ),
                job_id=job_id,
            )
        for state in workers.values():
            _terminate_worker(state)
            _cleanup_worker_inputs(state)
        pseq_sender_stop.set()
        try:
            pseq_send_queue.put_nowait(None)
        except Exception:
            pass
        if pseq_sender_thread.is_alive():
            pseq_sender_thread.join(timeout=2.0)
        residue_cache.clear()
        server.close()
        async_writer.join()
        async_writer.shutdown()


def run_kinform_parallel_pipeline(
    *,
    env: dict[str, str],
    repo_root: Path,
    media_path: Path,
    seq_id_to_seq: dict[str, str],
    job_id: str | None = None,
) -> None:
    stream_enabled = _env_bool("KINFORM_PARALLEL_STREAM_ENABLE", False)
    stream_allow_fallback = _env_bool("KINFORM_PARALLEL_STREAM_ALLOW_LEGACY_FALLBACK", True)

    if not stream_enabled:
        _run_kinform_parallel_pipeline_file_polling(
            env=env,
            repo_root=repo_root,
            media_path=media_path,
            seq_id_to_seq=seq_id_to_seq,
            job_id=job_id,
        )
        return

    try:
        _run_kinform_parallel_pipeline_stream(
            env=env,
            repo_root=repo_root,
            media_path=media_path,
            seq_id_to_seq=seq_id_to_seq,
            job_id=job_id,
        )
    except Exception as exc:
        if not stream_allow_fallback:
            raise
        _log(
            env,
            "warn",
            (
                "KINFORM_PARALLEL_FALLBACK "
                f"stream_reason={exc.__class__.__name__}:{exc}"
            ),
            job_id=job_id,
        )
        _run_kinform_parallel_pipeline_file_polling(
            env=env,
            repo_root=repo_root,
            media_path=media_path,
            seq_id_to_seq=seq_id_to_seq,
            job_id=job_id,
        )


def main() -> int:
    import argparse

    parser = argparse.ArgumentParser(description="Run KinForm parallel GPU embedding orchestrator.")
    parser.add_argument(
        "--seq-id-to-seq-file",
        required=True,
        help="Path to JSON mapping seq_id->sequence.",
    )
    parser.add_argument(
        "--repo-root",
        default=os.environ.get("GPU_REPO_ROOT", ""),
        help="Repository root path (defaults to GPU_REPO_ROOT).",
    )
    parser.add_argument(
        "--media-path",
        default=os.environ.get("KINFORM_MEDIA_PATH", ""),
        help="Media/cache root path (defaults to KINFORM_MEDIA_PATH).",
    )
    parser.add_argument(
        "--job-id",
        default="",
        help="Optional job ID used for structured logs.",
    )
    args = parser.parse_args()

    if not args.repo_root:
        raise RuntimeError("Missing --repo-root and GPU_REPO_ROOT is not set.")
    if not args.media_path:
        raise RuntimeError("Missing --media-path and KINFORM_MEDIA_PATH is not set.")

    seq_id_to_seq = json.loads(Path(args.seq_id_to_seq_file).read_text(encoding="utf-8"))
    run_kinform_parallel_pipeline(
        env=dict(os.environ),
        repo_root=Path(args.repo_root).resolve(),
        media_path=Path(args.media_path).resolve(),
        seq_id_to_seq=seq_id_to_seq,
        job_id=args.job_id or None,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
