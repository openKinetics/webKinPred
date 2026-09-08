#!/usr/bin/env python3
"""CatRange ESM-C final-layer mean embedding staging worker.

Writes persistent CatRange cache files:
  media/sequence_info/catrange_esmc/{seq_id}.npy

Each file contains a CPU float32 vector of shape [1152], the mean over
residue positions of the ESM-C 600M *final trunk embeddings*
(``logits(...).embeddings``), excluding the leading CLS token.

This must stay numerically identical to CatRangeInference.embed_sequence
(models/CatRange/inference/catrange_inference.py); the CatRange classifier
and its standardization stats were fit on exactly this representation. It is
deliberately NOT the KinForm ESM-C cache (esmc_layer_24 / esmc_layer_32),
which stores different hidden-state layers.

Runs in the shared ``esmc`` conda env. Used both by the GPU embed service
(tools/gpu_embed_service/run_step.py) and by CatRange's local CPU fallback
(models/CatRange/predict.py).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[2]
_REPO_ROOT_STR = str(_REPO_ROOT)
if _REPO_ROOT_STR in sys.path:
    sys.path.remove(_REPO_ROOT_STR)
sys.path.insert(0, _REPO_ROOT_STR)

from tools.gpu_embed_service.cache_io import SpoolAsyncCommitter, resolve_missing_ids


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="CatRange ESM-C final-layer mean embedding worker."
    )
    parser.add_argument("--seq-id-to-seq-file", required=True, type=str)
    parser.add_argument("--cache-dir", required=True, type=str)
    parser.add_argument("--async-workers", type=int, default=8)
    return parser.parse_args()


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
        suffix=".npy",
    )
    if not missing_ids:
        print(f"All {len(ready_ids)} CatRange ESM-C embeddings already staged.")
        return 0

    import torch
    from esm.models.esmc import ESMC
    from esm.sdk.api import ESMProtein, LogitsConfig

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"CatRange ESM-C: computing {len(missing_ids)} missing embedding(s) on {device}.")

    model = ESMC.from_pretrained("esmc_600m").to(device)
    model.eval()
    # Must match CatRangeInference.embed_sequence exactly.
    config = LogitsConfig(sequence=True, structure=True, return_embeddings=True)

    async_workers = max(1, int(args.async_workers))

    @torch.no_grad()
    def embed_one(sequence: str) -> np.ndarray:
        sequence = str(sequence).strip().upper()
        protein = ESMProtein(sequence=sequence)
        tokens = model.encode(protein)
        out = model.logits(tokens, config)
        reps = out.embeddings[0, 1 : len(sequence) + 1].float()
        return reps.mean(dim=0).cpu().numpy().astype(np.float32, copy=False)

    with SpoolAsyncCommitter(max_workers=async_workers) as committer:
        for seq_id in missing_ids:
            mean_vec = embed_one(seq_id_to_seq[seq_id])
            committer.submit_numpy(cache_dir=cache_dir, seq_id=seq_id, array=mean_vec)
            if device.type == "cuda":
                torch.cuda.empty_cache()

    print(f"CatRange ESM-C: committed {len(missing_ids)} embedding(s) to {cache_dir}.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
