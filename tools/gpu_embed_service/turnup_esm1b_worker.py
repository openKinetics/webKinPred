#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch

import sys as _execution_sys
_execution_sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from tools.executor.policy import require_cuda
_REPO_ROOT = Path(__file__).resolve().parents[2]
_REPO_ROOT_STR = str(_REPO_ROOT)
if _REPO_ROOT_STR in sys.path:
    sys.path.remove(_REPO_ROOT_STR)
sys.path.insert(0, _REPO_ROOT_STR)

from tools.gpu_embed_service.cache_io import SpoolAsyncCommitter, resolve_missing_ids


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Batched TurNup ESM1b embedding worker.")
    parser.add_argument("--seq-id-to-seq-file", required=True, type=str)
    parser.add_argument("--cache-dir", required=True, type=str)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--async-workers", type=int, default=8)
    return parser.parse_args()


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
        print(f"All {len(ready_ids)} TurNup ESM1b embeddings already ready in cache.")
        return 0

    print(f"TurNup ESM1b: computing {len(missing_ids)} missing sequence embedding(s).")

    import esm  # Imported lazily only when missing IDs exist.

    model, alphabet = esm.pretrained.esm1b_t33_650M_UR50S()
    batch_converter = alphabet.get_batch_converter()
    model.eval()
    device = torch.device("cuda")
    model = model.to(device)
    print(f"TurNup ESM1b device={device} batch_size={max(1, int(args.batch_size))}")

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
            data = []
            for seq_id in batch_ids:
                seq = str(seq_map[seq_id]).strip()
                seq = seq[:1022]  # ESM1b token context limit.
                data.append((seq_id, seq))

            _, _, batch_tokens = batch_converter(data)
            batch_tokens = batch_tokens.to(device)
            with torch.inference_mode():
                out = model(batch_tokens, repr_layers=[33], return_contacts=False)
            reps = (
                out["representations"][33][:, 0, :]
                .detach()
                .float()
                .cpu()
                .numpy()
                .astype(np.float32, copy=False)
            )
            for idx, seq_id in enumerate(batch_ids):
                committer.submit_numpy(cache_dir=cache_dir, seq_id=seq_id, array=reps[idx])

    print(f"TurNup ESM1b: committed {len(missing_ids)} embedding(s) to {cache_dir}.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
