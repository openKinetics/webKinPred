from __future__ import annotations

import pickle
import inspect
from argparse import Namespace

import numpy as np
import pandas as pd
import torch
import esm
import os
import sys
from pathlib import Path
from os.path import join
import subprocess

# Use environment variables to determine paths
_HERE = Path(__file__).resolve()
_REPO_ROOT = _HERE.parents[3]
_DEFAULT_MEDIA = _REPO_ROOT / "media"
_DEFAULT_TOOLS = _REPO_ROOT / "tools"

_REPO_ROOT_STR = str(_REPO_ROOT)
if _REPO_ROOT_STR in sys.path:
    sys.path.remove(_REPO_ROOT_STR)
sys.path.insert(0, _REPO_ROOT_STR)

from tools.gpu_embed_service.cache_io import SpoolAsyncCommitter, resolve_missing_ids

_media_path = Path(os.environ.get("TURNUP_MEDIA_PATH", str(_DEFAULT_MEDIA)))
_tools_path = Path(os.environ.get("TURNUP_TOOLS_PATH", str(_DEFAULT_TOOLS)))

if os.environ.get("TURNUP_DATA_PATH"):
    data_dir = os.environ.get("TURNUP_DATA_PATH")
elif Path("/app/models/TurNup/data").exists():
    data_dir = "/app/models/TurNup/data"
else:
    data_dir = str((_HERE.parents[1] / "data").resolve())

SEQ_VEC_DIR = str((_media_path / "sequence_info" / "esm1b_turnup").resolve())
SEQMAP_PY = sys.executable
SEQMAP_CLI = str((_tools_path / "seqmap" / "main.py").resolve())
SEQMAP_DB = str((_media_path / "sequence_info" / "seqmap.sqlite3").resolve())

aa = set("abcdefghiklmnpqrstxvwyzv".upper())

os.makedirs(SEQ_VEC_DIR, exist_ok=True)


def _torch_load_compat(path, map_location=None):
    """Prefer weights-only loading when supported; keep legacy fallback."""
    try:
        return torch.load(path, map_location=map_location, weights_only=True)
    except TypeError:
        # torch<2.0 does not support weights_only.
        return torch.load(path, map_location=map_location)
    except (RuntimeError, pickle.UnpicklingError):
        # Some trusted upstream ESM checkpoints include argparse.Namespace
        # metadata. Keep weights-only mode where PyTorch supports allowlisting.
        add_safe_globals = getattr(torch.serialization, "add_safe_globals", None)
        if add_safe_globals is not None:
            try:
                add_safe_globals([Namespace])
                return torch.load(path, map_location=map_location, weights_only=True)
            except (RuntimeError, pickle.UnpicklingError):
                pass

        # Final fallback for bundled/trusted TurNup model artifacts.
        try:
            return torch.load(path, map_location=map_location, weights_only=False)
        except TypeError:
            return torch.load(path, map_location=map_location)


def _load_model_and_alphabet_core_compat(model_name, model_data, regression_data):
    core_loader = esm.pretrained.load_model_and_alphabet_core
    try:
        first_param = next(iter(inspect.signature(core_loader).parameters), None)
    except (TypeError, ValueError):
        first_param = None

    if first_param == "model_name":
        return core_loader(model_name, model_data, regression_data)
    return core_loader(model_data, regression_data)


def resolve_seq_ids_via_cli(sequences):
    """Resolve IDs for all sequences in order (increments uses_count per occurrence)."""
    payload = "\n".join(sequences) + "\n"
    cmd = [SEQMAP_PY, SEQMAP_CLI, "--db", SEQMAP_DB, "batch-get-or-create", "--stdin"]
    proc = subprocess.run(
        cmd, input=payload, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE
    )
    if proc.returncode != 0:
        raise RuntimeError(
            f"seqmap CLI failed (rc={proc.returncode})\nSTDOUT:\n{proc.stdout}\nSTDERR:\n{proc.stderr}"
        )
    ids = proc.stdout.strip().splitlines()
    if len(ids) != len(sequences):
        raise RuntimeError(
            f"seqmap returned {len(ids)} ids for {len(sequences)} sequences"
        )
    return ids


def validate_enzyme(seq, alphabet=aa):
    "Checks that a sequence only contains values from an alphabet"
    leftover = set(seq.upper()) - alphabet
    return not leftover


def load_esm1b_model(device: torch.device | None = None):
    """Load ESM1b model once and return model and batch_converter."""
    from tools.executor.policy import require_cached
    require_cached(False, "protein embeddings must be scheduled")
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Loading ESM1b model on {device}...")
    model_location = join(data_dir, "saved_models", "ESM1b", "esm1b_t33_650M_UR50S.pt")
    model_data = _torch_load_compat(model_location, map_location="cpu")
    regression_location = model_location[:-3] + "-contact-regression.pt"
    regression_data = _torch_load_compat(regression_location, map_location="cpu")
    model, alphabet = _load_model_and_alphabet_core_compat(
        "esm1b_t33_650M_UR50S", model_data, regression_data
    )
    model.eval()

    batch_converter = alphabet.get_batch_converter()
    PATH = join(
        data_dir, "saved_models", "ESM1b", "model_ESM_binary_A100_epoch_1_new_split.pkl"
    )
    model_dict = _torch_load_compat(PATH, map_location="cpu")
    model_dict_V2 = {k.split("model.")[-1]: v for k, v in model_dict.items()}
    for key in [
        "module.fc1.weight",
        "module.fc1.bias",
        "module.fc2.weight",
        "module.fc2.bias",
        "module.fc3.weight",
        "module.fc3.bias",
    ]:
        if key in model_dict_V2:
            del model_dict_V2[key]
    model.load_state_dict(model_dict_V2)
    model = model.to(device)
    print("ESM1b model loaded successfully!")

    return model, batch_converter


def calcualte_esm1b_ts_vectors(enzyme_list, esm_model=None, batch_converter=None):
    """
    Calculate ESM1b vectors for enzyme list.
    If esm_model and batch_converter are provided, use them (for efficiency).
    Otherwise, load the model (for backward compatibility).
    """
    df_enzyme = preprocess_enzymes(enzyme_list)

    df_enzyme["enzyme rep"] = pd.Series([None] * len(df_enzyme), dtype=object)

    # Resolve IDs in a single call (updates uses_count & last_seen_at)
    seqs = df_enzyme["model_input"].tolist()
    ids = resolve_seq_ids_via_cli(seqs)
    df_enzyme["ID"] = ids

    cache_dir = Path(SEQ_VEC_DIR).resolve()
    missing_ids, _ready_ids = resolve_missing_ids(
        ids,
        cache_dir=cache_dir,
        suffix=".npy",
    )
    missing_set = set(missing_ids)

    seq_id_to_row: dict[str, int] = {}
    seq_id_to_seq: dict[str, str] = {}
    for ind in df_enzyme.index:
        seq_id = str(df_enzyme.at[ind, "ID"]).strip()
        seq = str(df_enzyme.at[ind, "model_input"]).strip()
        if seq_id not in missing_set:
            df_enzyme.at[ind, "enzyme rep"] = np.load(cache_dir / f"{seq_id}.npy")
        else:
            seq_id_to_row[seq_id] = int(ind)
            seq_id_to_seq[seq_id] = seq

    if seq_id_to_seq:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # Load model if not provided (backward compatibility)
        if esm_model is None or batch_converter is None:
            print(f"Embedding {len(seq_id_to_seq)} new sequences...")
            esm_model, batch_converter = load_esm1b_model(device=device)
        else:
            esm_model = esm_model.to(device)
            esm_model.eval()
            print(
                f"Embedding {len(seq_id_to_seq)} new sequences using pre-loaded model on {device}..."
            )
        batch_size_raw = str(os.environ.get("TURNUP_EMBED_BATCH_SIZE", "8")).strip()
        try:
            batch_size = max(1, int(batch_size_raw))
        except ValueError:
            batch_size = 8
        async_workers_raw = str(
            os.environ.get("TURNUP_CACHE_ASYNC_WORKERS", os.environ.get("GPU_EMBED_CACHE_ASYNC_WORKERS", "4"))
        ).strip()
        try:
            async_workers = max(1, int(async_workers_raw))
        except ValueError:
            async_workers = 4
        spool_dir = Path(os.environ.get("GPU_EMBED_CACHE_SPOOL_DIR", "/dev/shm/webkinpred-gpu-cache"))
        spool_fallback = Path(os.environ.get("GPU_EMBED_CACHE_SPOOL_FALLBACK_DIR", "/tmp/webkinpred-gpu-cache"))

        valid_ids = [sid for sid in seq_id_to_seq if validate_enzyme(seq_id_to_seq[sid])]
        with SpoolAsyncCommitter(
            max_workers=async_workers,
            spool_dir=spool_dir,
            spool_fallback_dir=spool_fallback,
        ) as committer:
            for start in range(0, len(valid_ids), batch_size):
                batch_ids = valid_ids[start : start + batch_size]
                data = [(sid, seq_id_to_seq[sid]) for sid in batch_ids]
                _, _, tokens = batch_converter(data)
                tokens = tokens.to(device)
                with torch.no_grad():
                    results = esm_model(tokens, repr_layers=[33], return_contacts=False)
                batch_reps = (
                    results["representations"][33][:, 0, :]
                    .detach()
                    .cpu()
                    .numpy()
                    .astype(np.float32, copy=False)
                )
                for row_idx, seq_id in enumerate(batch_ids):
                    rep = batch_reps[row_idx]
                    committer.submit_numpy(cache_dir=cache_dir, seq_id=seq_id, array=rep)
                    df_enzyme.at[seq_id_to_row[seq_id], "enzyme rep"] = rep

    return df_enzyme


def preprocess_enzymes(enzyme_list):
    # If you want per-occurrence counting in uses_count, remove the set():
    # df_enzyme = pd.DataFrame(data={"amino acid sequence": list(enzyme_list)})
    df_enzyme = pd.DataFrame(data={"amino acid sequence": list(set(enzyme_list))})
    df_enzyme["ID"] = ["protein_" + str(ind) for ind in df_enzyme.index]
    # if length of sequence is longer than 1020 amino acids, we crop it:
    df_enzyme["model_input"] = [seq[:1022] for seq in df_enzyme["amino acid sequence"]]
    return df_enzyme
