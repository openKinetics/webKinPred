#!/usr/bin/env python3
"""Batched IECata ProtT5 residue embedding staging worker.

Writes ephemeral IECata cache files:
  media/sequence_info/iecata_prot_t5_residues/{seq_id}.npy

Each file contains a CPU float32 matrix of shape [seq_len, 1024].
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path

import numpy as np
import torch
import sys as _execution_sys
_execution_sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from tools.executor.policy import require_cuda
from transformers import T5EncoderModel, T5Tokenizer

_REPO_ROOT = Path(__file__).resolve().parents[2]
_REPO_ROOT_STR = str(_REPO_ROOT)
if _REPO_ROOT_STR in sys.path:
    sys.path.remove(_REPO_ROOT_STR)
sys.path.insert(0, _REPO_ROOT_STR)

from tools.gpu_embed_service.cache_io import SpoolAsyncCommitter, resolve_missing_ids


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Batched IECata ProtT5 residue embedding worker.")
    parser.add_argument("--seq-id-to-seq-file", required=True, type=str)
    parser.add_argument("--cache-dir", required=True, type=str)
    parser.add_argument("--model-path", default="", type=str)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--async-workers", type=int, default=8)
    return parser.parse_args()


def _default_model_path() -> str:
    for env_var in ("IECATA_PROTT5_LOCAL_PATH", "KINFORM_T5_MODEL_PATH"):
        path = str(os.environ.get(env_var, "")).strip()
        if path:
            return path
    return str(os.environ.get("IECATA_PROTT5_NAME", "Rostlab/prot_t5_xl_uniref50")).strip()


def _clean_sequence(sequence: str) -> list[str]:
    compact = "".join(str(sequence).split())
    return list(re.sub(r"[UZOB]", "X", compact))


def main() -> int:
    require_cuda(torch)
    args = _parse_args()
    raw_map = json.loads(Path(args.seq_id_to_seq_file).read_text(encoding="utf-8"))
    if not isinstance(raw_map, dict) or not raw_map:
        print("No sequences provided; nothing to do.")
        return 0

    seq_id_to_seq = {
        str(seq_id).strip(): str(sequence).strip()
        for seq_id, sequence in raw_map.items()
        if str(seq_id).strip() and str(sequence).strip()
    }
    if not seq_id_to_seq:
        print("No non-empty sequences provided; nothing to do.")
        return 0

    cache_dir = Path(args.cache_dir).resolve()
    cache_dir.mkdir(parents=True, exist_ok=True)
    ordered_ids = list(seq_id_to_seq.keys())
    missing_ids, ready_ids = resolve_missing_ids(
        ordered_ids,
        cache_dir=cache_dir,
        suffix=".npy",
    )

    if not missing_ids:
        print(f"All {len(ready_ids)} IECata ProtT5 residue embeddings already staged.")
        return 0

    model_path = str(args.model_path).strip() or _default_model_path()
    local_only = Path(model_path).exists()

    print(f"IECata ProtT5 residues: computing {len(missing_ids)} missing embedding(s).")
    tokenizer = T5Tokenizer.from_pretrained(
        model_path,
        do_lower_case=False,
        local_files_only=local_only,
    )
    device = torch.device("cuda")
    dtype = torch.float16 if device.type == "cuda" else torch.float32
    model = T5EncoderModel.from_pretrained(
        model_path,
        local_files_only=local_only,
        torch_dtype=dtype,
    ).to(device)
    model.eval()
    print(f"IECata ProtT5 residues device={device} batch_size={max(1, int(args.batch_size))}")

    batch_size = max(1, int(args.batch_size))
    async_workers = max(1, int(args.async_workers))
    spool_dir = Path(os.environ.get("GPU_EMBED_CACHE_SPOOL_DIR", "/dev/shm/webkinpred-gpu-cache"))
    spool_fallback = Path(
        os.environ.get("GPU_EMBED_CACHE_SPOOL_FALLBACK_DIR", "/tmp/webkinpred-gpu-cache")
    )

    with SpoolAsyncCommitter(
        max_workers=async_workers,
        spool_dir=spool_dir,
        spool_fallback_dir=spool_fallback,
    ) as committer:
        for start in range(0, len(missing_ids), batch_size):
            batch_ids = missing_ids[start : start + batch_size]
            batch_tokens = [_clean_sequence(seq_id_to_seq[seq_id]) for seq_id in batch_ids]
            encoded = tokenizer.batch_encode_plus(
                batch_tokens,
                add_special_tokens=True,
                padding=True,
                is_split_into_words=True,
                return_tensors="pt",
            )
            encoded = {key: value.to(device) for key, value in encoded.items()}
            attention_mask = encoded["attention_mask"]

            with torch.inference_mode():
                hidden = model(input_ids=encoded["input_ids"], attention_mask=attention_mask).last_hidden_state

            hidden_np = hidden.float().cpu().numpy()
            seq_lens = attention_mask.sum(dim=1).cpu().numpy()
            for row_idx, seq_id in enumerate(batch_ids):
                seq_len = int(seq_lens[row_idx])
                token_count = max(seq_len - 1, 1)  # exclude final eos token
                residue = hidden_np[row_idx, :token_count].astype(np.float32, copy=False)
                committer.submit_numpy(cache_dir=cache_dir, seq_id=seq_id, array=residue)

            if device.type == "cuda":
                torch.cuda.empty_cache()

    print(f"IECata ProtT5 residues: committed {len(missing_ids)} embedding(s) to {cache_dir}.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
