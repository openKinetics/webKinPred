#!/usr/bin/env python3
"""Batched OmniESI ESM2 embedding staging worker.

Writes ephemeral OmniESI staging files:
  media/sequence_info/omniesi_esm2/{seq_id}.pt

Each file contains a CPU float32 tensor of shape [seq_len, 1280], matching
models/OmniESI/batch_predict.py. The prediction subprocess deletes these
files after consuming them.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
_REPO_ROOT_STR = str(_REPO_ROOT)
if _REPO_ROOT_STR in sys.path:
    sys.path.remove(_REPO_ROOT_STR)
sys.path.insert(0, _REPO_ROOT_STR)

from tools.gpu_embed_service.cache_io import SpoolAsyncCommitter, resolve_missing_ids


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Batched OmniESI ESM2 embedding worker.")
    parser.add_argument("--seq-id-to-seq-file", required=True, type=str)
    parser.add_argument("--cache-dir", required=True, type=str)
    parser.add_argument("--token-budget", type=int, default=0)
    parser.add_argument("--max-batch", type=int, default=0)
    parser.add_argument("--async-workers", type=int, default=8)
    return parser.parse_args()


def _env_int(name: str, default: int) -> int:
    raw = str(os.environ.get(name, "")).strip()
    if not raw:
        return int(default)
    try:
        value = int(raw)
    except ValueError:
        return int(default)
    return value if value > 0 else int(default)


def _make_batches(
    *,
    seq_id_to_seq: dict[str, str],
    seq_ids: list[str],
    token_budget: int,
    max_batch: int,
) -> list[list[str]]:
    ordered = sorted(seq_ids, key=lambda sid: len(seq_id_to_seq[sid]), reverse=True)
    batches: list[list[str]] = []
    current: list[str] = []
    current_tokens = 0

    for seq_id in ordered:
        token_len = min(len(seq_id_to_seq[seq_id]) + 2, 1024)
        if current and (len(current) >= max_batch or current_tokens + token_len > token_budget):
            batches.append(current)
            current = []
            current_tokens = 0
        current.append(seq_id)
        current_tokens += token_len
    if current:
        batches.append(current)
    return batches


def main() -> int:
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
        suffix=".pt",
    )
    if not missing_ids:
        print(f"All {len(ready_ids)} OmniESI ESM2 embeddings already staged.")
        return 0

    import esm
    import torch

    from tools.executor.policy import require_cuda
    require_cuda(torch)
    device = torch.device("cuda")
    print(f"OmniESI ESM2: computing {len(missing_ids)} missing embedding(s) on {device}.")

    model, alphabet = esm.pretrained.esm2_t33_650M_UR50D()
    batch_converter = alphabet.get_batch_converter()
    model = model.to(device)
    model.eval()

    token_budget = int(args.token_budget) if args.token_budget > 0 else _env_int(
        "OMNIESI_GPU_ESM_TOKEN_BUDGET",
        6000 if device.type == "cuda" else 2000,
    )
    max_batch = int(args.max_batch) if args.max_batch > 0 else _env_int(
        "OMNIESI_GPU_ESM_MAX_BATCH",
        8 if device.type == "cuda" else 2,
    )
    batches = _make_batches(
        seq_id_to_seq=seq_id_to_seq,
        seq_ids=missing_ids,
        token_budget=token_budget,
        max_batch=max_batch,
    )

    async_workers = max(1, int(args.async_workers))
    spool_dir = Path(os.environ.get("GPU_EMBED_CACHE_SPOOL_DIR", "/dev/shm/webkinpred-gpu-cache"))
    spool_fallback = Path(os.environ.get("GPU_EMBED_CACHE_SPOOL_FALLBACK_DIR", "/tmp/webkinpred-gpu-cache"))

    def run_batch(batch_ids: list[str], committer: SpoolAsyncCommitter) -> None:
        data = [(seq_id, seq_id_to_seq[seq_id]) for seq_id in batch_ids]
        _labels, _seqs, batch_tokens = batch_converter(data)
        if batch_tokens.shape[1] > 1024:
            batch_tokens = batch_tokens[:, :1024]
        batch_tokens = batch_tokens.to(device)

        with torch.inference_mode():
            results = model(batch_tokens, repr_layers=[33], return_contacts=False)
        token_repr = results["representations"][33].detach().float().cpu()

        for row_idx, seq_id in enumerate(batch_ids):
            seq_len = len(seq_id_to_seq[seq_id])
            max_residues = token_repr.shape[1] - 1
            valid_len = min(seq_len, max_residues)
            residue = token_repr[row_idx, 1 : 1 + valid_len].contiguous()
            committer.submit_torch_tensor(cache_dir=cache_dir, seq_id=seq_id, tensor=residue)

    def run_batch_with_retry(batch_ids: list[str], committer: SpoolAsyncCommitter) -> None:
        try:
            run_batch(batch_ids, committer)
        except RuntimeError as exc:
            if "out of memory" in str(exc).lower() and device.type == "cuda" and len(batch_ids) > 1:
                torch.cuda.empty_cache()
                mid = len(batch_ids) // 2
                run_batch_with_retry(batch_ids[:mid], committer)
                run_batch_with_retry(batch_ids[mid:], committer)
                return
            raise

    with SpoolAsyncCommitter(
        max_workers=async_workers,
        spool_dir=spool_dir,
        spool_fallback_dir=spool_fallback,
    ) as committer:
        for batch_ids in batches:
            run_batch_with_retry(batch_ids, committer)
            if device.type == "cuda":
                torch.cuda.empty_cache()

    print(f"OmniESI ESM2: committed {len(missing_ids)} embedding(s) to {cache_dir}.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
