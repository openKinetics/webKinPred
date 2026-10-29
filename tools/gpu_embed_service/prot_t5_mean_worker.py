#!/usr/bin/env python3
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
    parser = argparse.ArgumentParser(description="Batched ProtT5 mean embedding worker.")
    parser.add_argument("--seq-id-to-seq-file", required=True, type=str)
    parser.add_argument("--cache-dir", required=True, type=str)
    parser.add_argument("--model-path", default="", type=str)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--async-workers", type=int, default=8)
    return parser.parse_args()


def _default_model_path() -> str:
    env_model = str(os.environ.get("KINFORM_T5_MODEL_PATH", "")).strip()
    if env_model:
        return env_model
    return "Rostlab/prot_t5_xl_uniref50"


def main() -> int:
    require_cuda(torch)
    args = _parse_args()
    seq_map = json.loads(Path(args.seq_id_to_seq_file).read_text(encoding="utf-8"))
    if not isinstance(seq_map, dict) or not seq_map:
        print("No sequences provided; nothing to do.")
        return 0

    cache_dir = Path(args.cache_dir).resolve()
    cache_dir.mkdir(parents=True, exist_ok=True)
    ordered_ids = [str(seq_id).strip() for seq_id in seq_map.keys() if str(seq_id).strip()]
    missing_ids, ready_ids = resolve_missing_ids(
        ordered_ids,
        cache_dir=cache_dir,
        suffix=".npy",
    )

    if not missing_ids:
        print(f"All {len(ready_ids)} ProtT5 mean embeddings already ready in cache.")
        return 0

    model_path = str(args.model_path).strip() or _default_model_path()
    local_only = bool(os.environ.get("KINFORM_MEDIA_PATH")) or Path(model_path).exists()

    print(f"ProtT5 mean: computing {len(missing_ids)} missing sequence embedding(s).")
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
    print(f"ProtT5 mean device={device} batch_size={max(1, int(args.batch_size))}")

    batch_size = max(1, int(args.batch_size))
    async_workers = max(1, int(args.async_workers))
    spool_dir = Path(os.environ.get("GPU_EMBED_CACHE_SPOOL_DIR", "/dev/shm/webkinpred-gpu-cache"))
    spool_fallback = Path(os.environ.get("GPU_EMBED_CACHE_SPOOL_FALLBACK_DIR", "/tmp/webkinpred-gpu-cache"))

    with SpoolAsyncCommitter(
        max_workers=async_workers,
        spool_dir=spool_dir,
        spool_fallback_dir=spool_fallback,
    ) as committer:
        for start in range(0, len(missing_ids), batch_size):
            batch_ids = missing_ids[start : start + batch_size]
            batch_strs = []
            for seq_id in batch_ids:
                seq = str(seq_map[seq_id]).strip()
                spaced = " ".join(list(seq))
                spaced = re.sub(r"[UZOB]", "X", spaced)
                batch_strs.append(spaced)

            token_data = tokenizer(
                batch_strs,
                return_tensors="pt",
                padding=True,
                add_special_tokens=True,
            )
            input_ids = token_data["input_ids"].to(device)
            attention_mask = token_data["attention_mask"].to(device)

            with torch.inference_mode():
                hidden = model(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state

            hidden_np = hidden.float().cpu().numpy()
            seq_lens = attention_mask.sum(dim=1).cpu().numpy()
            for row_idx, seq_id in enumerate(batch_ids):
                seq_len = int(seq_lens[row_idx])
                token_count = max(seq_len - 1, 1)  # exclude eos token
                mean_vec = hidden_np[row_idx, :token_count].mean(axis=0).astype(np.float32, copy=False)
                committer.submit_numpy(cache_dir=cache_dir, seq_id=seq_id, array=mean_vec)

    print(f"ProtT5 mean: committed {len(missing_ids)} embedding(s) to {cache_dir}.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
