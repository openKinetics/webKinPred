#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Extract ProtT5‐XL UniRef50 embeddings.

Four operating modes:
    1. --setting mean
       → compute & save the per-sequence **mean vector of the last layer**.
    2. --setting residue [--layer N]
       → compute & save the **per-residue matrix** from either the last
         layer (default) or the user–specified layer *N* (0-based, 0-23).
    3. --setting weighted [--layer N] --weights_file WEIGHTS.tsv
       → compute weighted mean embeddings using per-residue weights.
    4. --all_layers
       → compute & save the **mean vector of every encoder layer**;
         result shape: [24, hidden_size].

Embeddings are written under:
    results/protein_embeddings/prot_t5_last/mean_vecs/     (mean vectors, last layer)
    results/protein_embeddings/prot_t5_layer_{n}/mean_vecs/ (mean vectors, layer n)
    results/protein_embeddings/prot_t5_last/weighted_vecs/     (weighted vectors, last layer)
    results/protein_embeddings/prot_t5_layer_{n}/weighted_vecs/ (weighted vectors, layer n)

If all required files already exist the model is *not* loaded.
"""

import gc
import json
import os
import pickle
import re
import sys
import tempfile
import time
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from transformers import T5EncoderModel, T5Tokenizer

_REPO_ROOT = Path(__file__).resolve().parents[4]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from tools.gpu_embed_service.kinform_stream_ipc import StreamClient

PROTT5XL_MODEL_PATH = "Rostlab/prot_t5_xl_uniref50"
if os.environ.get("KINFORM_T5_MODEL_PATH"):
    PROTT5XL_MODEL_PATH = os.environ.get("KINFORM_T5_MODEL_PATH")
elif os.environ.get("KINFORM_MEDIA_PATH"):
    docker_model = Path("/app/models/UniKP-main/models/protT5_xl/prot_t5_xl_uniref50")
    if docker_model.exists():
        PROTT5XL_MODEL_PATH = str(docker_model)

# --------------------------------------------------------------------------- #
#                              HELPER FUNCTIONS                               #
# --------------------------------------------------------------------------- #
def _fetch_weights(seq_id: str, df: pd.DataFrame, key_col: str, weights_col: str) -> np.ndarray:
    """
    Return a 1-D float64 array of per-residue weights for `seq_id`.
    Raises if the sequence is missing.
    """
    row = df.loc[df[key_col] == seq_id, weights_col]
    if row.empty:
        raise ValueError(f"No weights found in {weights_col} for sequence {seq_id}")
    return np.fromiter((float(x) for x in row.iloc[0].split(",")), dtype=float)


def _truncate_first_last_array(arr: np.ndarray, keep: int = 1024) -> np.ndarray:
    if arr.shape[0] <= keep:
        return arr
    half = keep // 2
    return np.concatenate([arr[:half], arr[-half:]], axis=0)


def _align_residue_to_binding_weights(
    seq_id: str,
    residue_embedding: np.ndarray,
    weights: np.ndarray,
) -> np.ndarray:
    if residue_embedding.shape[0] == weights.shape[0]:
        return residue_embedding
    if weights.shape[0] == 1024 and residue_embedding.shape[0] > 1024:
        return _truncate_first_last_array(residue_embedding, keep=1024)
    raise ValueError(
        f"Weight length ({weights.shape[0]}) does not match embedding length "
        f"({residue_embedding.shape[0]}) for {seq_id}"
    )


def _weighted_mean(arr: np.ndarray, w: np.ndarray, normalize: bool = True) -> np.ndarray:
    """Length‑L weights → weighted mean over axis‑0."""
    w = np.asarray(w, dtype=float)
    if normalize:
        w = w / w.sum()
    return (arr * w[:, None]).sum(axis=0)


def _cleanup_residue_files(residue_dir: Path, keys: List[str]) -> None:
    """Best-effort removal of residue embedding files for the given keys."""
    removed = 0
    for key in keys:
        fp = residue_dir / f"{key}.npy"
        try:
            if fp.exists():
                fp.unlink()
                removed += 1
        except OSError:
            pass
    if removed:
        print(f"Removed {removed} residue embedding file(s) from {residue_dir}")


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


def _require_cuda_if_requested(context: str) -> None:
    raw = str(os.environ.get("KINFORM_REQUIRE_CUDA", "")).strip().lower()
    require_cuda = raw in {"1", "true", "yes", "on"}
    if require_cuda and not torch.cuda.is_available():
        raise RuntimeError(
            f"CUDA is required for {context} (KINFORM_REQUIRE_CUDA=1), "
            "but no CUDA device is available."
        )


# --------------------------------------------------------------------------- #
#                              EMBEDDING BACK-END                             #
# --------------------------------------------------------------------------- #
def get_prot_t5_embeddings(
    seq_dict: Dict[str, str],
    *,
    batch_size = 2,
    setting  = "mean",              # 'mean' | 'residue' | 'weighted' | 'mean+weighted'
    all_layers = False,
    layer = None,           # 0-23 for residue setting
    only_save = False,
    id_to_seq = None,
    weights_df: Optional[pd.DataFrame] = None,
    weights_key_col: str = "PDB",
    weights_col: str = "Pred_BS_Scores"):

    _require_cuda_if_requested("ProtT5 embedding")

    # ----------------------- sanity checks -------------------------------- #
    # Parse setting - can be combination like "mean+weighted"
    settings = set(s.strip() for s in setting.split("+"))
    valid_settings = {"mean", "residue", "weighted"}
    assert settings.issubset(valid_settings), f"Invalid setting: {setting}. Valid: {valid_settings}"
    
    if all_layers:
        assert layer is None, "--layer is invalid when --all_layers is set"
    if "weighted" in settings:
        assert weights_df is not None, "--weights_file is required when setting includes 'weighted'"
        assert not all_layers, "weighted setting is incompatible with --all_layers"

    # ------------------------- path handling ------------------------------ #
    # Get repository root relative to this file
    precomputed_root = Path(os.environ.get("KINFORM_MEDIA_PATH")) / "sequence_info" 
    assert all(k in id_to_seq and id_to_seq[k] == v for k, v in seq_dict.items()), (
        "Sequence mismatch between provided seq_dict and id_to_seq"
    )

    # Set up paths based on setting and layer
    paths = {}

    if all_layers:
        paths["all_layers"] = precomputed_root / "prot_t5_all_layers"
    else:
        if "residue" in settings:
            if layer is None:
                paths["residue"] = precomputed_root / "prot_t5_last/residue_vecs"
            else:
                paths["residue"] = precomputed_root / f"prot_t5_layer_{layer}/residue_vecs"

        if "mean" in settings:
            if layer is None:
                paths["mean"] = precomputed_root / "prot_t5_last/mean_vecs"
            else:
                paths["mean"] = precomputed_root / f"prot_t5_layer_{layer}/mean_vecs"

        if "weighted" in settings:
            if layer is None:
                paths["weighted"] = precomputed_root / "prot_t5_last/weighted_vecs"
            else:
                paths["weighted"] = precomputed_root / f"prot_t5_layer_{layer}/weighted_vecs"

    for p in paths.values():
        p.mkdir(parents=True, exist_ok=True)

    # ------------------------ skip existing files ------------------------- #
    if all_layers:
        key_to_exist = {
            k: (paths["all_layers"] / f"{k}.npy").exists()
            for k in seq_dict
        }
    else:
        # Check if all requested settings already exist for all sequences
        key_to_exist = {
            k: all((paths[s] / f"{k}.npy").exists() for s in settings)
            for k in seq_dict
        }

    if all(key_to_exist.values()):
        print("All required ProtT5 embeddings already on disk — skipping model load.")
        if not all_layers and "residue" not in settings:
            if layer is None:
                _residue_dir = precomputed_root / "prot_t5_last" / "residue_vecs"
            else:
                _residue_dir = precomputed_root / f"prot_t5_layer_{layer}" / "residue_vecs"
            _cleanup_residue_files(_residue_dir, list(seq_dict.keys()))
        if only_save:
            return None
        return _load_existing_embeddings(seq_dict, paths, settings, all_layers, layer)

    # ── derive mean/weighted from pre-computed residue files (no T5 load) ── #
    # If gen_features.py already ran T5 for binding-site prediction it will have
    # left per-residue .npy files for this layer on disk.  We can compute mean
    # and weighted embeddings from them with pure numpy, then delete the residue
    # files, avoiding a second full T5 forward pass.
    missing_keys = [k for k, ok in key_to_exist.items() if not ok]

    if (not all_layers
            and "residue" not in settings
            and settings.issubset({"mean", "weighted"})
            and missing_keys):
        if layer is None:
            _residue_dir = precomputed_root / "prot_t5_last" / "residue_vecs"
        else:
            _residue_dir = precomputed_root / f"prot_t5_layer_{layer}" / "residue_vecs"

        if all((_residue_dir / f"{k}.npy").exists() for k in missing_keys):
            print(f"Deriving embeddings from {len(missing_keys)} pre-computed residue file(s) "
                  f"(layer {'last' if layer is None else layer}) — T5 not loaded.")
            for key in missing_keys:
                residue_path = _residue_dir / f"{key}.npy"
                residue_emb = np.load(residue_path)  # [L, 1024] float32

                if "mean" in settings:
                    np.save(paths["mean"] / f"{key}.npy", residue_emb.mean(axis=0))

                if "weighted" in settings:
                    weights = _fetch_weights(key, weights_df, weights_key_col, weights_col)
                    weighted_residue_emb = _align_residue_to_binding_weights(key, residue_emb, weights)
                    np.save(paths["weighted"] / f"{key}.npy",
                            _weighted_mean(weighted_residue_emb, weights, normalize=True))

                residue_path.unlink()

            if only_save:
                return None
            return _load_existing_embeddings(seq_dict, paths, settings, all_layers, layer)
    # ─────────────────────────────────────────────────────────────────────── #

    # --------------------------- model load ------------------------------- #
    print(f"Loading ProtT5-XL UniRef50 from {PROTT5XL_MODEL_PATH} ...")
    local_only = bool(os.environ.get("KINFORM_MEDIA_PATH"))
    tokenizer = T5Tokenizer.from_pretrained(
        PROTT5XL_MODEL_PATH,
        do_lower_case=False,
        local_files_only=local_only,
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float16 if device.type == "cuda" else torch.float32
    model = T5EncoderModel.from_pretrained(
        PROTT5XL_MODEL_PATH,
        dtype=dtype,
        low_cpu_mem_usage=True,
        output_hidden_states=True,
        local_files_only=local_only,
    )
    model.eval()
    model = model.to(device)
    print(f"Using device: {device}")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()) // 1e6:,} M")

    # ----------------------- batching & encoding -------------------------- #
    missing_keys = [k for k, ok in key_to_exist.items() if not ok]
    print(f"Generating embeddings for {len(missing_keys)} new sequences")
    batches = [missing_keys[i:i + batch_size] for i in range(0, len(missing_keys), batch_size)]

    for batch_keys in tqdm(batches, desc="ProtT5 batches"):
        batch_seqs = [seq_dict[k] for k in batch_keys]
        # ProtT5 expects amino acids separated by space & ambiguous tokens as 'X'
        batch_strs = [
            " ".join(list(re.sub(r"[UZOB]", "X", s))) for s in batch_seqs
        ]
        token_data = tokenizer(
            batch_strs,
            return_tensors="pt",
            padding=True,
            add_special_tokens=True,
        ).to(device)

        with torch.inference_mode():
            outputs = model(**token_data)
        hidden_states: Tuple[torch.Tensor, ...] = outputs.hidden_states  # len=25, 0=embeddings

        # lengths incl. <eos>; exclude from downstream
        seq_lens = (token_data["attention_mask"] == 1).sum(dim=1) - 1  # [batch]

        for idx, key in enumerate(batch_keys):
            L = seq_lens[idx].item()
            if all_layers:
                layer_means: List[np.ndarray] = []
                for hs in hidden_states[1:]:  # skip embedding layer
                    vec = hs[idx, :L].mean(dim=0).float().cpu().numpy()
                    layer_means.append(vec)
                stack = np.stack(layer_means)             # [24, hidden]
                np.save(paths["all_layers"] / f"{key}.npy", stack)

            else:
                # Get the appropriate layer
                if layer is None:
                    layer_tensor = hidden_states[-1]           # last layer
                else:
                    layer_tensor = hidden_states[layer + 1]    # skip embed layer
                
                residue_emb = layer_tensor[idx, :L].float().cpu().numpy()  # [L, H]
                
                # Save residue embeddings if requested
                if "residue" in settings:
                    np.save(paths["residue"] / f"{key}.npy", residue_emb)
                
                # Compute and save mean if requested
                if "mean" in settings:
                    mean_vec = residue_emb.mean(axis=0)
                    np.save(paths["mean"] / f"{key}.npy", mean_vec)
                
                # Compute and save weighted if requested
                if "weighted" in settings:
                    # Fetch weights for this sequence
                    weights = _fetch_weights(key, weights_df, weights_key_col, weights_col)
                    weighted_residue_emb = _align_residue_to_binding_weights(key, residue_emb, weights)
                    
                    # Compute weighted mean
                    weighted_vec = _weighted_mean(weighted_residue_emb, weights, normalize=True)
                    np.save(paths["weighted"] / f"{key}.npy", weighted_vec)

        # ------------------- memory hygiene per batch -------------------- #
        del token_data, outputs, hidden_states
        torch.cuda.empty_cache()
        gc.collect()

    # ---------------------- return (optional) ----------------------------- #
    if only_save:
        if not all_layers and "residue" not in settings:
            if layer is None:
                _residue_dir = precomputed_root / "prot_t5_last" / "residue_vecs"
            else:
                _residue_dir = precomputed_root / f"prot_t5_layer_{layer}" / "residue_vecs"
            _cleanup_residue_files(_residue_dir, list(seq_dict.keys()))
        return None
    out = _load_existing_embeddings(seq_dict, paths, settings, all_layers, layer)
    if not all_layers and "residue" not in settings:
        if layer is None:
            _residue_dir = precomputed_root / "prot_t5_last" / "residue_vecs"
        else:
            _residue_dir = precomputed_root / f"prot_t5_layer_{layer}" / "residue_vecs"
        _cleanup_residue_files(_residue_dir, list(seq_dict.keys()))
    return out


# --------------------------------------------------------------------------- #
#                       MULTI-LAYER RESIDUE EXTRACTION                        #
# --------------------------------------------------------------------------- #
def _get_prot_t5_residue_multi_layer(
    seq_dict: Dict[str, str],
    layers: List,          # e.g. [19, None]  (None = last encoder layer)
    *,
    batch_size: int = 2,
    id_to_seq=None,
) -> None:
    """Load T5 once and save per-residue embeddings for every layer in `layers`.

    Produces exactly the same .npy files as calling
      get_prot_t5_embeddings(setting='residue', layer=L)
    for each L, but uses a single model load and a single forward pass per
    batch instead of one model load per layer.
    """
    _require_cuda_if_requested("ProtT5 multi-layer residue embedding")
    precomputed_root = Path(os.environ.get("KINFORM_MEDIA_PATH")) / "sequence_info"
    assert all(k in id_to_seq and id_to_seq[k] == v for k, v in seq_dict.items()), (
        "Sequence mismatch between provided seq_dict and id_to_seq"
    )

    # Build (and create) output directories for each requested layer — mirrors
    # the path logic in get_prot_t5_embeddings with setting='residue'.
    layer_dirs: Dict = {}
    for layer in layers:
        if layer is None:
            d = precomputed_root / "prot_t5_last" / "residue_vecs"
        else:
            d = precomputed_root / f"prot_t5_layer_{layer}" / "residue_vecs"
        d.mkdir(parents=True, exist_ok=True)
        layer_dirs[layer] = d

    # Only process sequences that are missing for at least one layer.
    missing_keys = [
        k for k in seq_dict
        if any(not (layer_dirs[l] / f"{k}.npy").exists() for l in layers)
    ]

    if not missing_keys:
        print("All residue embeddings already on disk — skipping model load.")
        return

    # ── single model load ──────────────────────────────────────────────────
    print(f"Loading ProtT5-XL UniRef50 from {PROTT5XL_MODEL_PATH}...")
    local_only = bool(os.environ.get("KINFORM_MEDIA_PATH"))
    tokenizer = T5Tokenizer.from_pretrained(
        PROTT5XL_MODEL_PATH,
        do_lower_case=False,
        local_files_only=local_only,
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float16 if device.type == "cuda" else torch.float32
    model = T5EncoderModel.from_pretrained(
        PROTT5XL_MODEL_PATH,
        dtype=dtype,
        low_cpu_mem_usage=True,
        output_hidden_states=True,
        local_files_only=local_only,
    )
    model.eval()
    model = model.to(device)
    print(f"Using device: {device}")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()) // 1e6:,} M")

    batches = [missing_keys[i:i + batch_size] for i in range(0, len(missing_keys), batch_size)]
    for batch_keys in tqdm(batches, desc="ProtT5 residue (multi-layer)"):
        batch_seqs = [seq_dict[k] for k in batch_keys]
        batch_strs = [" ".join(list(re.sub(r"[UZOB]", "X", s))) for s in batch_seqs]
        token_data = tokenizer(
            batch_strs, return_tensors="pt", padding=True, add_special_tokens=True,
        ).to(device)

        with torch.inference_mode():
            outputs = model(**token_data)
        hidden_states: Tuple[torch.Tensor, ...] = outputs.hidden_states  # len=25

        seq_lens = (token_data["attention_mask"] == 1).sum(dim=1) - 1  # excl. <eos>
        for idx, key in enumerate(batch_keys):
            L = seq_lens[idx].item()
            for layer in layers:
                out_path = layer_dirs[layer] / f"{key}.npy"
                if out_path.exists():
                    continue  # already written (e.g. partial previous run)
                layer_tensor = hidden_states[-1] if layer is None else hidden_states[layer + 1]
                np.save(out_path, layer_tensor[idx, :L].float().cpu().numpy())

        del token_data, outputs, hidden_states
        torch.cuda.empty_cache()
        gc.collect()

    del model, tokenizer
    gc.collect()


class _BackgroundWriter:
    """Drains (path, arr) write jobs on a background thread so inference never blocks on SSHFS."""

    def __init__(self) -> None:
        import queue as _q
        import threading as _t
        self._q: _q.Queue = _q.Queue()
        self._exc: Optional[BaseException] = None
        self._thread = _t.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def _loop(self) -> None:
        while True:
            item = self._q.get()
            if item is None:
                return
            path, arr = item
            try:
                _save_array_atomic(path, arr)
            except Exception as exc:
                self._exc = exc

    def submit(self, path: Path, arr: np.ndarray) -> None:
        self._q.put((path, arr.copy()))

    def join(self) -> None:
        self._q.put(None)
        self._thread.join()
        if self._exc is not None:
            raise self._exc


def _stream_prot_t5_residue_mean_multi_layer(
    seq_dict: Dict[str, str],
    *,
    layers: List,
    batch_size: int,
    id_to_seq: Dict[str, str],
    stream_socket: str,
    stream_job_id: str,
    worker_name: str,
    legacy_residue_write: bool = False,
) -> None:
    _require_cuda_if_requested("ProtT5 stream embedding")
    stage_name = "prot_t5_residue"
    stage_started_at = time.monotonic()
    precomputed_root = Path(os.environ.get("KINFORM_MEDIA_PATH")) / "sequence_info"
    assert all(k in id_to_seq and id_to_seq[k] == v for k, v in seq_dict.items()), (
        "Sequence mismatch between provided seq_dict and id_to_seq"
    )
    write_mean_files = str(os.environ.get("KINFORM_STREAM_WRITE_MEAN_FILES", "1")).strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }

    layer_mean_dirs: Dict = {}
    layer_residue_dirs: Dict = {}
    layer_root_names: Dict = {}
    for layer in layers:
        if layer is None:
            root_name = "prot_t5_last"
        else:
            root_name = f"prot_t5_layer_{layer}"
        layer_root_names[layer] = root_name
        if write_mean_files:
            mean_dir = precomputed_root / root_name / "mean_vecs"
            mean_dir.mkdir(parents=True, exist_ok=True)
            layer_mean_dirs[layer] = mean_dir
        if legacy_residue_write:
            residue_dir = precomputed_root / root_name / "residue_vecs"
            residue_dir.mkdir(parents=True, exist_ok=True)
            layer_residue_dirs[layer] = residue_dir

    keys = list(seq_dict.keys())
    if not keys:
        return
    layers_per_sequence = len(layers)
    total_items = len(keys) * layers_per_sequence
    show_progress = str(os.environ.get("KINFORM_STREAM_TQDM", "")).strip().lower() in {"1", "true", "yes", "on"}
    emitted_items = 0
    last_progress_at = stage_started_at
    progress_interval_seconds = 10.0

    stream = StreamClient(stream_socket)
    bg_writer = _BackgroundWriter() if (write_mean_files or legacy_residue_write) else None
    try:
        model_load_started_at = time.monotonic()
        print(f"Loading ProtT5-XL UniRef50 from {PROTT5XL_MODEL_PATH}...")
        local_only = bool(os.environ.get("KINFORM_MEDIA_PATH"))
        tokenizer = T5Tokenizer.from_pretrained(
            PROTT5XL_MODEL_PATH,
            do_lower_case=False,
            local_files_only=local_only,
        )
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        dtype = torch.float16 if device.type == "cuda" else torch.float32
        model = T5EncoderModel.from_pretrained(
            PROTT5XL_MODEL_PATH,
            dtype=dtype,
            low_cpu_mem_usage=True,
            output_hidden_states=True,
            local_files_only=local_only,
        )
        model.eval()
        model = model.to(device)
        print(f"Using device: {device}")
        print(f"Model parameters: {sum(p.numel() for p in model.parameters()) // 1e6:,} M")
        model_load_elapsed_s = max(0.0, time.monotonic() - model_load_started_at)
        print(
            f"KINFORM_TIMING stage={stage_name} model_load_s={model_load_elapsed_s:.3f} "
            f"seq_count={len(keys)} layer_count={layers_per_sequence}"
        )

        batches = [keys[i:i + batch_size] for i in range(0, len(keys), batch_size)]
        for batch_keys in tqdm(
            batches,
            desc="ProtT5 stream batches",
            disable=not show_progress,
        ):
            batch_seqs = [seq_dict[k] for k in batch_keys]
            batch_strs = [" ".join(list(re.sub(r"[UZOB]", "X", s))) for s in batch_seqs]
            token_data = tokenizer(
                batch_strs, return_tensors="pt", padding=True, add_special_tokens=True
            ).to(device)
            with torch.inference_mode():
                outputs = model(**token_data)
            hidden_states: Tuple[torch.Tensor, ...] = outputs.hidden_states
            seq_lens = (token_data["attention_mask"] == 1).sum(dim=1) - 1

            for idx, key in enumerate(batch_keys):
                L = seq_lens[idx].item()
                sequence = seq_dict[key]
                for layer in layers:
                    layer_tensor = hidden_states[-1] if layer is None else hidden_states[layer + 1]
                    residue_emb = layer_tensor[idx, :L].float().cpu().numpy().astype(np.float32, copy=False)
                    if write_mean_files:
                        mean_vec = residue_emb.mean(axis=0).astype(np.float32, copy=False)
                        assert bg_writer is not None
                        bg_writer.submit(layer_mean_dirs[layer] / f"{key}.npy", mean_vec)

                    if legacy_residue_write:
                        assert bg_writer is not None
                        bg_writer.submit(layer_residue_dirs[layer] / f"{key}.npy", residue_emb)

                    header = {
                        "type": "RESIDUE_READY",
                        "worker": worker_name,
                        "job_id": stream_job_id,
                        "family": "t5",
                        "root": layer_root_names[layer],
                        "seq_id": key,
                        "sequence": sequence,
                        "dtype": "float32",
                        "shape": [int(x) for x in residue_emb.shape],
                    }
                    stream.send(header, residue_emb.tobytes(order="C"))
                    emitted_items += 1

            del token_data, outputs, hidden_states

            now = time.monotonic()
            if now - last_progress_at >= progress_interval_seconds:
                elapsed_s = max(0.0, now - stage_started_at)
                rate = (emitted_items / elapsed_s) if elapsed_s > 0 else 0.0
                print(
                    f"KINFORM_TIMING stage={stage_name} progress items={emitted_items}/{total_items} "
                    f"elapsed_s={elapsed_s:.3f} rate_items_per_s={rate:.3f}"
                )
                last_progress_at = now

        if bg_writer is not None:
            bg_writer.join()
        if legacy_residue_write:
            for residue_dir in layer_residue_dirs.values():
                _cleanup_residue_files(residue_dir, keys)
        stream.send(
            {
                "type": "WORKER_DONE",
                "worker": worker_name,
                "job_id": stream_job_id,
            },
            b"",
        )
        total_elapsed_s = max(0.0, time.monotonic() - stage_started_at)
        rate = (emitted_items / total_elapsed_s) if total_elapsed_s > 0 else 0.0
        print(
            f"KINFORM_TIMING stage={stage_name} summary items={emitted_items}/{total_items} "
            f"elapsed_s={total_elapsed_s:.3f} rate_items_per_s={rate:.3f}"
        )
    except Exception as exc:
        try:
            stream.send(
                {
                    "type": "WORKER_ERROR",
                    "worker": worker_name,
                    "job_id": stream_job_id,
                    "message": str(exc),
                },
                b"",
            )
        except Exception:
            pass
        raise
    finally:
        stream.close()


# --------------------------------------------------------------------------- #
#                               HELPER ROUTINE                                #
# --------------------------------------------------------------------------- #
def _load_existing_embeddings(
    seq_dict: Dict[str, str],
    paths: Dict[str, Path],
    settings: set,
    all_layers: bool,
    layer):
    """
    Load embeddings that are now guaranteed to exist.
    """
    if all_layers:
        return {
            k: np.load(paths["all_layers"] / f"{k}.npy")
            for k in seq_dict
        }
    elif len(settings) == 1:
        # Single setting - return flat dict
        setting = next(iter(settings))
        return {
            k: np.load(paths[setting] / f"{k}.npy")
            for k in seq_dict
        }
    else:
        # Multiple settings - return nested dict
        return {
            k: {
                s: np.load(paths[s] / f"{k}.npy")
                for s in settings
            }
            for k in seq_dict
        }


# --------------------------------------------------------------------------- #
#                             SCRIPT ENTRY POINT                              #
# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    """
    Extract ProtT5 embeddings for unique sequences.
    Default: Runs for layer 19 and last layer (residue setting).
    
    Usage:
        # Default behavior (residue embeddings)
        python t5_embeddings.py
        
        # Mean embeddings
        python t5_embeddings.py --setting mean
        
        # Weighted embeddings using binding site scores
        python t5_embeddings.py --setting weighted --weights_file path/to/binding_sites.tsv
        
        # Custom sequence file and layers
        python t5_embeddings.py --seq_file path/to/sequences.txt --layers 19 None
    """
    import argparse
    from pathlib import Path
    
    parser = argparse.ArgumentParser(
        description='Extract ProtT5 protein embeddings',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Default: residue embeddings for layer 19 and last layer
  python t5_embeddings.py
  
  # Mean embeddings
  python t5_embeddings.py --setting mean --layers 19 None
  
  # Weighted embeddings using binding site predictions
  python t5_embeddings.py --setting weighted --weights_file results/binding_sites/binding_sites_all.tsv --layers 19 None
  
  # Both mean and weighted (computes residue once, derives both)
  python t5_embeddings.py --setting mean+weighted --weights_file results/binding_sites/binding_sites_all.tsv --layers 19 None
  
  # All three: residue, mean, and weighted
  python t5_embeddings.py --setting residue+mean+weighted --weights_file results/binding_sites/binding_sites_all.tsv
  
  # Custom weights column names
  python t5_embeddings.py --setting weighted --weights_file my_weights.tsv --weights_key_col sequence_id --weights_col my_weights
  
  # Single layer only
  python t5_embeddings.py --layers 19
        """
    )
    
    parser.add_argument(
        '--seq_file',
        type=str,
        default=None,
        help='Path to text file containing sequence IDs (one per line). Default: data/unique_seq_ids.txt'
    )
    
    parser.add_argument(
        '--layers',
        type=str,
        nargs='+',
        default=None,
        help='Layer number(s) to extract (0-23) or "None" for last layer. Default: [19, None]'
    )
    
    parser.add_argument(
        '--batch_size',
        type=int,
        default=1,
        help='Batch size for processing. Default: 1'
    )
    
    parser.add_argument(
        '--setting',
        type=str,
        default='residue',
        help='Embedding type to extract. Can combine with "+", e.g., "mean+weighted". Options: mean, residue, weighted. Default: residue'
    )
    
    parser.add_argument(
        '--all_layers',
        action='store_true',
        help='Extract embeddings from all 24 layers (overrides --layers)'
    )
    
    parser.add_argument(
        '--weights_file',
        type=str,
        default=None,
        help='Path to TSV file with per-residue weights (required for --setting weighted)'
    )
    
    parser.add_argument(
        '--weights_key_col',
        type=str,
        default='PDB',
        help='Column name in weights file for sequence IDs. Default: PDB'
    )
    
    parser.add_argument(
        '--weights_col',
        type=str,
        default='Pred_BS_Scores',
        help='Column name in weights file for weight values. Default: Pred_BS_Scores'
    )
    parser.add_argument(
        '--id_to_seq_file',
        type=str,
        default=None,
        help='Path to pickle file with sequence_id to sequence mapping. If not provided, uses default path.'
    )
    parser.add_argument(
        '--stream-mode',
        action='store_true',
        help='Emit residue embeddings to orchestrator stream instead of relying on residue files.'
    )
    parser.add_argument(
        '--stream-socket',
        type=str,
        default='',
        help='Unix socket path used for stream mode IPC.'
    )
    parser.add_argument(
        '--stream-job-id',
        type=str,
        default='',
        help='Optional job ID attached to stream events.'
    )
    parser.add_argument(
        '--worker-name',
        type=str,
        default='t5',
        help='Worker name attached to stream events.'
    )
    parser.add_argument(
        '--legacy-residue-write',
        action='store_true',
        help='In stream mode, also write residue files to disk for compatibility/fallback.'
    )
    
    args = parser.parse_args()
    
    # Validate arguments
    settings_requested = set(s.strip() for s in args.setting.split("+"))
    valid_settings = {"mean", "residue", "weighted"}
    if not settings_requested.issubset(valid_settings):
        parser.error(f"Invalid setting: {args.setting}. Valid options: mean, residue, weighted (can combine with +)")
    
    if "weighted" in settings_requested and args.weights_file is None:
        parser.error("--weights_file is required when --setting includes 'weighted'")
    
    # Determine sequence file
    if args.seq_file:
        seq_file = Path(args.seq_file)
    else:
        raise ValueError("Please provide --seq_file with sequence IDs.")
    
    # Load sequence ID to sequence mapping
    if args.id_to_seq_file:
        id_to_seq: Dict[str, str] = pickle.load(open(args.id_to_seq_file, "rb"))
    else:
        raise ValueError("Please provide --id_to_seq_file with the ID-to-sequence mapping pickle file.")
    
    # Load unique sequence IDs from file
    with open(seq_file, "r") as f:
        seq_ids = [line.strip() for line in f if line.strip()]
    
    assert all(sid in id_to_seq for sid in seq_ids), (
        "Some sequence IDs in the sequence file are missing from the ID-to-sequence mapping."
    )

    # Build sequence dictionary
    seq_dict = {sid: id_to_seq[sid] for sid in seq_ids}
    print(f"Loaded {len(seq_dict)} unique sequences from {seq_file}")
    
    # Load weights file if needed
    weights_df = None
    if "weighted" in settings_requested:
        print(f"Loading weights from {args.weights_file}")
        weights_df = pd.read_csv(args.weights_file, sep='\t')
        print(f"Loaded weights for {len(weights_df)} sequences")
    
    # Parse layers (used by both regular and stream mode)
    if args.layers:
        layers = []
        for l in args.layers:
            if l.lower() == 'none':
                layers.append(None)
            else:
                layers.append(int(l))
    else:
        layers = [19, None]

    if args.stream_mode:
        if args.all_layers:
            parser.error("--all_layers is not supported in --stream-mode")
        if settings_requested != {"residue", "mean"}:
            parser.error("--stream-mode currently supports only --setting residue+mean")
        if not args.stream_socket:
            parser.error("--stream-socket is required in --stream-mode")
        print(f"\n{'='*70}")
        print(f"Streaming ProtT5 residue+mean embeddings for layers: {layers}")
        print(f"{'='*70}")
        _stream_prot_t5_residue_mean_multi_layer(
            seq_dict,
            layers=layers,
            batch_size=args.batch_size,
            id_to_seq=id_to_seq,
            stream_socket=args.stream_socket,
            stream_job_id=args.stream_job_id,
            worker_name=args.worker_name,
            legacy_residue_write=args.legacy_residue_write,
        )
        print(f"✓ Completed ProtT5 stream extraction for layers: {layers}")
    elif args.all_layers:
        # Extract all layers mode
        print(f"\n{'='*70}")
        print(f"Extracting ProtT5 embeddings for all 24 layers")
        print(f"{'='*70}")
        
        embeddings = get_prot_t5_embeddings(
            seq_dict,
            batch_size=args.batch_size,
            setting=args.setting,
            all_layers=True,
            layer=None,
            only_save=True,
            id_to_seq=id_to_seq,
            weights_df=weights_df,
            weights_key_col=args.weights_key_col,
            weights_col=args.weights_col,
        )
        
        print(f"✓ Completed ProtT5 all-layers embedding extraction")
    else:
        # When extracting residue embeddings for multiple layers, use a single
        # T5 forward pass for all layers instead of one load per layer.
        if settings_requested == {"residue"} and len(layers) > 1:
            print(f"\n{'='*70}")
            print(f"Extracting ProtT5 residue embeddings for layers: {layers} (single model load)")
            print(f"{'='*70}")
            _get_prot_t5_residue_multi_layer(
                seq_dict,
                layers=layers,
                batch_size=args.batch_size,
                id_to_seq=id_to_seq,
            )
            print(f"✓ Completed ProtT5 residue extraction for layers: {layers}")
        else:
            # All other cases: per-layer loop (single layer, mean, weighted, etc.)
            for layer in layers:
                layer_name = "last" if layer is None else str(layer)
                print(f"\n{'='*70}")
                print(f"Extracting ProtT5 {args.setting} embeddings for layer {layer_name}")
                print(f"{'='*70}")

                embeddings = get_prot_t5_embeddings(
                    seq_dict,
                    batch_size=args.batch_size,
                    setting=args.setting,
                    all_layers=False,
                    layer=layer,
                    only_save=True,
                    id_to_seq=id_to_seq,
                    weights_df=weights_df,
                    weights_key_col=args.weights_key_col,
                    weights_col=args.weights_col,
                )

                print(f"✓ Completed ProtT5 layer {layer_name} {args.setting} embedding extraction")
    
    print(f"\n{'='*70}")
    print(f"✓ All ProtT5 embeddings complete for {len(seq_dict)} sequences")
    print(f"{'='*70}")
