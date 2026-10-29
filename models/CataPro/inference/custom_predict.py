#!/usr/bin/env python3
"""
CataPro prediction wrapper for webKinPred generic subprocess engine.

Contract:
    python custom_predict.py --input <input.json> --output <output.json>
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
from pathlib import Path
from typing import Callable

import numpy as np
import torch
from rdkit import Chem, RDLogger
from rdkit.Chem import MACCSkeys
from torch.utils.data import DataLoader, Dataset
from transformers import T5EncoderModel, T5Tokenizer

RDLogger.DisableLog("rdApp.warning")

_REPO_ROOT = Path(__file__).resolve().parents[3]
_REPO_ROOT_STR = str(_REPO_ROOT)
if _REPO_ROOT_STR in sys.path:
    sys.path.remove(_REPO_ROOT_STR)
sys.path.insert(0, _REPO_ROOT_STR)

from tools.gpu_embed_service.cache_io import SpoolAsyncCommitter, resolve_missing_ids

from act_model import ActivityModel
from model import KcatModel, KmModel


class FeatureDataset(Dataset):
    def __init__(self, features: np.ndarray):
        self.features = torch.from_numpy(features).to(torch.float32)

    def __getitem__(self, idx: int):
        return self.features[idx]

    def __len__(self) -> int:
        return int(self.features.shape[0])


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _default_media_path() -> Path:
    return _repo_root() / "media"


def _default_tools_path() -> Path:
    return _repo_root() / "tools"


def _default_model_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _env_path(key: str, default: Path) -> Path:
    value = os.environ.get(key)
    return Path(value).resolve() if value else default.resolve()


def _resolve_model_path(
    env_key: str,
    candidates: list[Path],
    label: str,
) -> Path:
    explicit = os.environ.get(env_key)
    if explicit:
        p = Path(explicit).resolve()
        if p.exists():
            return p
        raise RuntimeError(f"{label} path set in {env_key} does not exist: {p}")

    for candidate in candidates:
        if candidate.exists():
            return candidate

    joined = "\n".join(f"  - {c}" for c in candidates)
    raise RuntimeError(
        f"Could not locate {label} model directory.\n"
        f"Checked:\n{joined}\n"
        f"Set {env_key} to the correct path."
    )


def resolve_seq_ids_via_cli(sequences: list[str], seqmap_cli: Path, seqmap_db: Path) -> list[str]:
    payload = "\n".join(sequences) + "\n"
    cmd = [
        sys.executable,
        str(seqmap_cli),
        "--db",
        str(seqmap_db),
        "batch-get-or-create",
        "--stdin",
    ]
    proc = subprocess.run(
        cmd,
        input=payload,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    if proc.returncode != 0:
        raise RuntimeError(
            f"seqmap CLI failed (rc={proc.returncode})\n"
            f"STDOUT:\n{proc.stdout}\nSTDERR:\n{proc.stderr}"
        )
    ids = proc.stdout.strip().splitlines()
    if len(ids) != len(sequences):
        raise RuntimeError(
            f"seqmap returned {len(ids)} ids for {len(sequences)} sequences"
        )
    return ids


def _normalise_sequence(seq: str) -> str:
    seq = seq.strip()
    if len(seq) > 1000:
        seq = seq[:500] + seq[-500:]
    return seq


def _compute_prott5_mean_embedding(
    sequence: str,
    tokenizer: T5Tokenizer,
    model: T5EncoderModel,
    device: torch.device,
) -> np.ndarray:
    spaced = " ".join(_normalise_sequence(sequence))
    spaced = re.sub(r"[UZOB]", "X", spaced)
    encoded = tokenizer.batch_encode_plus([spaced], add_special_tokens=True, padding=True)
    input_ids = torch.tensor(encoded["input_ids"]).to(device)
    attention_mask = torch.tensor(encoded["attention_mask"]).to(device)

    with torch.inference_mode():
        hidden = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
        ).last_hidden_state

    hidden = hidden.float().cpu().numpy()[0]
    seq_len = int((attention_mask[0] == 1).sum().item())
    pooled = hidden[: seq_len - 1].mean(axis=0)
    return pooled.astype(np.float32)


def get_prott5_embeddings(
    sequences: list[str],
    prott5_model_path: Path,
    cache_dir: Path,
    seqmap_cli: Path,
    seqmap_db: Path,
) -> np.ndarray:
    cache_dir.mkdir(parents=True, exist_ok=True)
    seq_ids = resolve_seq_ids_via_cli(sequences, seqmap_cli, seqmap_db)
    missing_ids, _ready_ids = resolve_missing_ids(seq_ids, cache_dir=cache_dir, suffix=".npy")
    missing_set = set(missing_ids)
    missing_by_seq_id: dict[str, str] = {}
    for seq, seq_id in zip(sequences, seq_ids):
        if seq_id not in missing_set or seq_id in missing_by_seq_id:
            continue
        # Shared ProtT5 cache is keyed by seq_id; repeated sequences embed once.
        missing_by_seq_id[seq_id] = seq

    tokenizer = None
    model = None
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    if missing_by_seq_id:
        from tools.executor.policy import require_cached
        require_cached(False, "ProtT5 means")
        tokenizer = T5Tokenizer.from_pretrained(
            str(prott5_model_path),
            do_lower_case=False,
            local_files_only=True,
        )
        model = T5EncoderModel.from_pretrained(
            str(prott5_model_path),
            local_files_only=True,
            torch_dtype=torch.float16 if device.type == "cuda" else torch.float32,
        ).to(device)
        model.eval()
        async_workers = max(
            1,
            int(
                os.environ.get(
                    "CATAPRO_CACHE_ASYNC_WORKERS",
                    os.environ.get("GPU_EMBED_CACHE_ASYNC_WORKERS", "4"),
                )
            ),
        )
        spool_dir = Path(os.environ.get("GPU_EMBED_CACHE_SPOOL_DIR", "/dev/shm/webkinpred-gpu-cache"))
        spool_fallback = Path(os.environ.get("GPU_EMBED_CACHE_SPOOL_FALLBACK_DIR", "/tmp/webkinpred-gpu-cache"))
        with SpoolAsyncCommitter(
            max_workers=async_workers,
            spool_dir=spool_dir,
            spool_fallback_dir=spool_fallback,
        ) as committer:
            for seq_id, seq in missing_by_seq_id.items():
                emb = _compute_prott5_mean_embedding(seq, tokenizer, model, device)
                committer.submit_numpy(cache_dir=cache_dir, seq_id=seq_id, array=emb)

    out = []
    for seq_id in seq_ids:
        fp = cache_dir / f"{seq_id}.npy"
        out.append(np.load(fp))
    return np.stack(out).astype(np.float32)


def get_molt5_embeddings(
    smiles_list: list[str],
    model_path: Path,
    device: torch.device,
    batch_size: int = 32,
) -> np.ndarray:
    from tools.executor.policy import cache_only, molecular_vectors
    if cache_only():
        return molecular_vectors("CataPro", smiles_list)
    if not smiles_list:
        return np.zeros((0, 768), dtype=np.float32)

    # Compute MolT5 embeddings once per unique substrate, then map back to
    # the original row order.
    unique_smiles: list[str] = []
    unique_index_by_smiles: dict[str, int] = {}
    row_to_unique_idx: list[int] = []
    for smiles in smiles_list:
        idx = unique_index_by_smiles.get(smiles)
        if idx is None:
            idx = len(unique_smiles)
            unique_index_by_smiles[smiles] = idx
            unique_smiles.append(smiles)
        row_to_unique_idx.append(idx)

    tokenizer = T5Tokenizer.from_pretrained(str(model_path), local_files_only=True)
    model = T5EncoderModel.from_pretrained(
        str(model_path),
        local_files_only=True,
        torch_dtype=torch.float16 if device.type == "cuda" and os.environ.get("WEBKINPRED_MOLECULAR_FP32") != "1" else torch.float32,
    ).to(device)
    model.eval()

    unique_embeddings: list[np.ndarray] = []
    for start in range(0, len(unique_smiles), batch_size):
        batch = unique_smiles[start : start + batch_size]
        encoded = tokenizer(
            batch,
            return_tensors="pt",
            padding=True,
            truncation=True,
        )
        input_ids = encoded["input_ids"].to(device)
        attention_mask = encoded["attention_mask"].to(device)

        with torch.inference_mode():
            hidden = model(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state

        hidden = hidden.float().cpu().numpy()
        attn = attention_mask.cpu().numpy()
        for idx in range(hidden.shape[0]):
            seq_len = int(attn[idx].sum())
            vec = hidden[idx][: max(seq_len - 1, 1)].mean(axis=0)
            unique_embeddings.append(vec.astype(np.float32))

    unique_arr = np.stack(unique_embeddings).astype(np.float32)
    return unique_arr[np.asarray(row_to_unique_idx, dtype=np.int64)]


def get_maccs(smiles_list: list[str]) -> np.ndarray:
    fps: list[np.ndarray] = []
    for smiles in smiles_list:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            raise RuntimeError(f"Invalid substrate SMILES during MACCS generation: {smiles}")
        bitstr = MACCSkeys.GenMACCSKeys(mol).ToBitString()
        fps.append(np.array([int(c) for c in bitstr], dtype=np.float32))
    return np.stack(fps).astype(np.float32)


def _coerce_bool(value: object, default: bool = False) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "on"}
    if isinstance(value, (int, float)):
        return bool(value)
    return default


def _resolve_substrate_text(
    value: object,
    *,
    canonicalize_substrates: bool = True,
) -> str | None:
    text = str(value).strip()
    if not text:
        return None

    mol_from_smiles = Chem.MolFromSmiles(text)
    if mol_from_smiles is not None:
        if not canonicalize_substrates:
            return text
        return Chem.MolToSmiles(mol_from_smiles)

    mol = Chem.MolFromInchi(text)
    if mol is None:
        return None
    return Chem.MolToSmiles(mol, canonical=canonicalize_substrates)


def _build_model(task_type: str, device: torch.device):
    if task_type == "KCAT":
        model = KcatModel(device=str(device))
        subdir = "kcat_models"
    elif task_type == "KM":
        model = KmModel(device=str(device))
        subdir = "Km_models"
    elif task_type == "KCAT/KM":
        # ActivityModel (act_model.py) instantiates its own KcatModel/KmModel
        # without passing a device, so they default to "cuda:0" inside __init__.
        # Patch act_model's local class defaults to the resolved device first.
        import act_model as _act_mod
        _act_classes = (_act_mod.KcatModel, _act_mod.KmModel)
        _saved = {cls: cls.__init__.__defaults__ for cls in _act_classes}
        for cls in _act_classes:
            cls.__init__.__defaults__ = cls.__init__.__defaults__[:-1] + (str(device),)
        try:
            model = ActivityModel(device=str(device))
        finally:
            for cls, d in _saved.items():
                cls.__init__.__defaults__ = d
        subdir = "act_models"
    else:
        raise RuntimeError(f"Unsupported task type: {task_type}")
    return model, subdir


def _predict_fold(
    model,
    dataloader: DataLoader,
    task_type: str,
    device: torch.device,
) -> np.ndarray:
    model.eval()
    preds: list[np.ndarray] = []
    with torch.inference_mode():
        for batch in dataloader:
            batch = batch.to(device)
            ezy_feats = batch[:, :1024]
            sbt_feats = batch[:, 1024:]
            if task_type == "KCAT/KM":
                pred = model(ezy_feats, sbt_feats)[-1]
            else:
                pred = model(ezy_feats, sbt_feats)
            preds.append(pred.float().cpu().numpy().ravel())
    return np.concatenate(preds, axis=0)


def run_catapro(
    sequences: list[str],
    smiles: list[str],
    kinetics_type: str,
    model_root: Path,
    prott5_model_path: Path,
    molt5_model_path: Path,
    seq_cache_dir: Path,
    seqmap_cli: Path,
    seqmap_db: Path,
    progress_callback: Callable[[int, int], None] | None = None,
) -> list[float]:
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    seq_embeddings = get_prott5_embeddings(
        sequences=sequences,
        prott5_model_path=prott5_model_path,
        cache_dir=seq_cache_dir,
        seqmap_cli=seqmap_cli,
        seqmap_db=seqmap_db,
    )
    mol_embeddings = get_molt5_embeddings(smiles, molt5_model_path, device=device)
    maccs = get_maccs(smiles)

    feats = np.concatenate([seq_embeddings, mol_embeddings, maccs], axis=1)
    dataloader = DataLoader(FeatureDataset(feats), batch_size=32, shuffle=False)
    total_predictions = int(feats.shape[0])

    _SUBDIR = {"KCAT": "kcat_models", "KM": "Km_models", "KCAT/KM": "act_models"}
    fold_models = []
    model_dir = model_root / "models" / _SUBDIR[kinetics_type]
    if not model_dir.exists():
        raise RuntimeError(f"CataPro model directory not found: {model_dir}")

    for fold in range(10):
        model, _ = _build_model(kinetics_type, device)
        ckpt = model_dir / f"{fold}_bestmodel.pth"
        if not ckpt.exists():
            raise RuntimeError(f"Missing CataPro checkpoint: {ckpt}")
        state = torch.load(str(ckpt), map_location=device)
        model.load_state_dict(state)
        model.to(device)
        model.eval()
        fold_models.append(model)

    predictions: list[float] = []
    with torch.inference_mode():
        for batch in dataloader:
            batch = batch.to(device)
            ezy_feats = batch[:, :1024]
            sbt_feats = batch[:, 1024:]
            batch_fold_preds: list[np.ndarray] = []
            for model in fold_models:
                if kinetics_type == "KCAT/KM":
                    pred = model(ezy_feats, sbt_feats)[-1]
                else:
                    pred = model(ezy_feats, sbt_feats)
                batch_fold_preds.append(pred.float().cpu().numpy().ravel())

            batch_log10 = np.mean(np.stack(batch_fold_preds, axis=0), axis=0)
            batch_linear = np.power(10.0, batch_log10.astype(np.float64))
            predictions.extend(batch_linear.astype(float).tolist())

            if progress_callback is not None:
                progress_callback(len(predictions), total_predictions)

    del fold_models
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return predictions


def _kinetics_type_from_payload(payload: dict) -> str:
    params = payload.get("params") or {}
    ktype = str(params.get("kinetics_type", "")).upper().strip()
    if ktype:
        return ktype

    target = str(payload.get("target", "")).strip()
    if target == "kcat":
        return "KCAT"
    if target == "Km":
        return "KM"
    if target == "kcat/Km":
        return "KCAT/KM"
    raise RuntimeError("Missing kinetics type and unknown target in input payload.")


def run_from_payload(payload: dict) -> dict:
    rows = payload.get("rows") or []
    if not isinstance(rows, list):
        raise RuntimeError("'rows' must be a list in input payload.")

    model_root = _env_path("CATAPRO_MODEL_ROOT", _default_model_root())
    media_path = _env_path("CATAPRO_MEDIA_PATH", _default_media_path())
    tools_path = _env_path("CATAPRO_TOOLS_PATH", _default_tools_path())

    prott5_model_path = _resolve_model_path(
        "CATAPRO_PROTT5_MODEL",
        [
            model_root / "models" / "prot_t5_xl_uniref50",
            model_root / "prot_t5_xl_uniref50",
            _repo_root() / "models" / "UniKP-main" / "models" / "protT5_xl" / "prot_t5_xl_uniref50",
        ],
        "ProtT5",
    )
    molt5_model_path = _resolve_model_path(
        "CATAPRO_MOLT5_MODEL",
        [
            model_root / "models" / "molt5-base-smiles2caption",
            model_root / "molt5-base-smiles2caption",
        ],
        "MolT5",
    )

    seq_cache_dir = media_path / "sequence_info" / "prot_t5_last" / "mean_vecs"
    seqmap_cli = tools_path / "seqmap" / "main.py"
    seqmap_db = media_path / "sequence_info" / "seqmap.sqlite3"

    if not seqmap_cli.exists():
        raise RuntimeError(f"seqmap CLI not found: {seqmap_cli}")
    if not os.environ.get("SEQMAP_MANIFEST_PATH") and not seqmap_db.exists():
        raise RuntimeError(f"seqmap DB not found: {seqmap_db}")

    kinetics_type = _kinetics_type_from_payload(payload)
    canonicalize_substrates = _coerce_bool(
        (payload.get("params") or {}).get("canonicalize_substrates"),
        default=True,
    )

    predictions: list[float | None] = [None] * len(rows)
    invalid_indices: list[int] = []
    valid_indices: list[int] = []
    valid_sequences: list[str] = []
    valid_smiles: list[str] = []

    for idx, row in enumerate(rows):
        seq = str(row.get("sequence", "")).strip()
        substrate = _resolve_substrate_text(
            row.get("substrates", row.get("substrate", "")),
            canonicalize_substrates=canonicalize_substrates,
        )
        if not seq or substrate is None:
            invalid_indices.append(idx)
            continue
        valid_indices.append(idx)
        valid_sequences.append(seq)
        valid_smiles.append(substrate)

    if valid_indices:
        total = len(rows)
        base_done = len(invalid_indices)

        def _emit_progress(done_valid: int, _total_valid: int) -> None:
            done_total = min(total, base_done + done_valid)
            print(f"Progress: {done_total}/{total}", flush=True)

        valid_preds = run_catapro(
            sequences=valid_sequences,
            smiles=valid_smiles,
            kinetics_type=kinetics_type,
            model_root=model_root,
            prott5_model_path=prott5_model_path,
            molt5_model_path=molt5_model_path,
            seq_cache_dir=seq_cache_dir,
            seqmap_cli=seqmap_cli,
            seqmap_db=seqmap_db,
            progress_callback=_emit_progress,
        )
        for local_idx, pred in enumerate(valid_preds):
            predictions[valid_indices[local_idx]] = float(pred)
    else:
        total = len(rows)
        for i in range(total):
            print(f"Progress: {i + 1}/{total}", flush=True)

    return {
        "predictions": predictions,
        "invalid_indices": sorted(set(invalid_indices)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run CataPro predictions in batch.")
    parser.add_argument("--input", required=True, help="Input JSON path.")
    parser.add_argument("--output", required=True, help="Output JSON path.")
    args = parser.parse_args()

    with open(args.input, "r", encoding="utf-8") as f:
        payload = json.load(f)

    result = run_from_payload(payload)

    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(result, f)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        from tools.executor.policy import propagate_resource_error
        propagate_resource_error(exc)
        print(f"[CataPro] ERROR: {exc}", file=sys.stderr, flush=True)
        raise
