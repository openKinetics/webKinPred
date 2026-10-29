"""CatPred GPU embedding precomputation script.

Runs on the GPU server.  Given a JSON mapping of {seq_id: sequence}, this
script produces the per-checkpoint attentively-pooled protein representations
required by the CatPred prediction pipeline and writes them to the shared
disk cache.  The production adapter (webkinpred_adapter.py) skips the
embedding step entirely whenever all cache files are already present.

Pipeline (mirrors webkinpred_adapter._prepare_seq_pooled_cache):
  1. Discover all model.pt checkpoint files under
     <checkpoint_root>/<parameter>/.
  2. Load ESM2 once and compute a per-residue representation for every
     sequence that is missing at least one cached file. CUDA is required.
  3. For each checkpoint model:
       a. Load the model onto the same device.
       b. Run seq_embedder → rotary embeds → multi-head attention →
          (optional ESM2 concatenation) → attentive pooling.
       c. Save the resulting pooled tensor as <cache_root>/<parameter>/
          <checkpoint_key>/<seq_id>.pt.

Usage
─────
    python catpred_embed_gpu.py \\
        --seq-id-to-seq-file /tmp/seq_map.json \\
        --parameter          kcat \\
        --checkpoint-root    /path/to/production_checkpoints \\
        --cache-root         /mnt/media/sequence_info/catpred_esm2

The script is idempotent: already-cached files are never recomputed.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

_THIS_FILE = Path(__file__).resolve()
_CATPRED_ROOT = _THIS_FILE.parents[2]


def _discover_repo_root(start: Path) -> Path:
    for candidate in [start] + list(start.parents):
        if (candidate / "tools" / "gpu_embed_service" / "cache_io.py").exists():
            return candidate
    # Fallback for expected source layout.
    return _THIS_FILE.parents[4]


_REPO_ROOT = _discover_repo_root(_THIS_FILE.parent)
_REPO_ROOT_STR = str(_REPO_ROOT)
_CATPRED_ROOT_STR = str(_CATPRED_ROOT)
for path_str in (_CATPRED_ROOT_STR, _REPO_ROOT_STR):
    if path_str in sys.path:
        sys.path.remove(path_str)
# Keep repo root first (for `tools.*` imports), catpred root second.
sys.path.insert(0, _CATPRED_ROOT_STR)
sys.path.insert(0, _REPO_ROOT_STR)

import torch

from catpred.utils import load_checkpoint
from tools.gpu_embed_service.cache_io import SpoolAsyncCommitter, resolve_missing_ids

_POOL_BATCH = 50  # sequences processed per sub-batch inside each loaded checkpoint model


# ── Checkpoint discovery ──────────────────────────────────────────────────────

def _checkpoint_cache_key(checkpoint_path: Path) -> str:
    checkpoint_path = checkpoint_path.resolve()
    parent = checkpoint_path.parent.name
    grandparent = checkpoint_path.parent.parent.name
    return f"{grandparent}__{parent}" if grandparent else parent


def _discover_checkpoint_models(checkpoint_dir: Path) -> list[tuple[str, Path]]:
    """Return [(cache_key, model_pt_path), ...] sorted by path."""
    checkpoint_files = sorted(checkpoint_dir.rglob("model.pt"))
    if not checkpoint_files:
        raise RuntimeError(f"No CatPred model checkpoints found under: {checkpoint_dir}")

    entries: list[tuple[str, Path]] = []
    seen_keys: set[str] = set()
    for cp in checkpoint_files:
        key = _checkpoint_cache_key(cp)
        if key in seen_keys:
            raise RuntimeError(
                f"Duplicate checkpoint cache key '{key}' under {checkpoint_dir}"
            )
        seen_keys.add(key)
        entries.append((key, cp.resolve()))
    return entries


# ── Per-sequence amino-acid index tensor ─────────────────────────────────────

_AA_TO_INDEX: dict[str, int] = {
    "A": 0, "R": 1, "N": 2, "D": 3, "C": 4, "Q": 5, "E": 6, "G": 7,
    "H": 8, "I": 9, "L": 10, "K": 11, "M": 12, "F": 13, "P": 14,
    "S": 15, "T": 16, "W": 17, "Y": 18, "V": 19,
}


def _sequence_to_tensor(sequence: str, device: torch.device) -> torch.Tensor:
    return torch.as_tensor(
        [_AA_TO_INDEX[aa] for aa in sequence],
        device=device,
        dtype=torch.long,
    ).unsqueeze(0)  # (1, seq_len)


# ── Attentive pooling ─────────────────────────────────────────────────────────

def _compute_seq_pooled_output(
    model: object,
    sequence: str,
    seq_id: str,
    esm_feature: torch.Tensor,
    device: torch.device,
) -> torch.Tensor:
    """Run the CatPred sequence encoder and return the pooled representation.

    Returns a CPU tensor of shape (hidden_dim,) ready to be saved with
    torch.save().  Mirrors webkinpred_adapter._compute_seq_pooled_output
    exactly so the cached tensors are interchangeable.
    """
    seq_arr = _sequence_to_tensor(sequence, device)
    esm_feature_arr = esm_feature.to(device).unsqueeze(0)  # (1, seq_len, esm_dim)

    # Align lengths when ESM2 truncated the sequence differently.
    if seq_arr.shape[1] != esm_feature_arr.shape[1]:
        common_len = min(seq_arr.shape[1], esm_feature_arr.shape[1])
        seq_arr = seq_arr[:, :common_len]
        esm_feature_arr = esm_feature_arr[:, :common_len]

    seq_outs = model.seq_embedder(seq_arr)
    q = model.rotary_embedder.rotate_queries_or_keys(seq_outs, seq_dim=1)
    k = model.rotary_embedder.rotate_queries_or_keys(seq_outs, seq_dim=1)
    seq_outs, _ = model.multihead_attn(q, k, seq_outs)

    if model.args.add_esm_feats:
        seq_outs = torch.cat([esm_feature_arr, seq_outs], dim=-1)

    if not model.args.skip_attentive_pooling:
        seq_pooled_outs, _ = model.attentive_pooler(seq_outs)
    else:
        seq_pooled_outs = seq_outs.mean(dim=1)

    if model.args.add_pretrained_egnn_feats:
        if seq_id in model.pretrained_egnn_feats_dict:
            pretrained_egnn = (
                model.pretrained_egnn_feats_dict[seq_id].to(device).unsqueeze(0)
            )
        else:
            pretrained_egnn = model.pretrained_egnn_feats_avg.to(device).unsqueeze(0)
        seq_pooled_outs = torch.cat([seq_pooled_outs, pretrained_egnn], dim=-1)

    return seq_pooled_outs.squeeze(0).detach().cpu()


def _env_int(name: str, default: int) -> int:
    raw = str(os.environ.get(name, "")).strip()
    if not raw:
        return int(default)
    try:
        value = int(raw)
    except ValueError:
        return int(default)
    return value if value > 0 else int(default)


def _compute_esm_reprs_batched(
    *,
    seq_id_to_seq: dict[str, str],
    seq_ids: list[str],
    device: torch.device,
) -> dict[str, torch.Tensor]:
    """Compute ESM2 residue representations using token-budget batching.

    This preserves output semantics of get_single_esm_repr() while avoiding
    one-forward-pass-per-sequence overhead.
    """
    from catpred.data import esm_utils

    if esm_utils.PROTEIN_EMBED_USE_CPU:
        raise RuntimeError("GPU_UNAVAILABLE: CatPred embedding requires CUDA execution")
    esm_utils.init_esm()
    model, batch_converter = esm_utils.GLOBAL_VARIABLES["model"]
    use_cpu = bool(esm_utils.PROTEIN_EMBED_USE_CPU)
    max_length = int(esm_utils.ESM_MAX_LENGTH)

    default_token_budget = 2000 if use_cpu else 6000
    token_budget = _env_int("CATPRED_GPU_ESM_TOKEN_BUDGET", default_token_budget)
    max_batch = _env_int("CATPRED_GPU_ESM_MAX_BATCH", 8)

    ordered = sorted(seq_ids, key=lambda sid: len(seq_id_to_seq[sid]), reverse=True)
    batches: list[list[str]] = []
    current: list[str] = []
    current_tokens = 0

    for seq_id in ordered:
        seq_len = len(seq_id_to_seq[seq_id])
        token_len = min(seq_len + 2, max_length)
        if current and (len(current) >= max_batch or (current_tokens + token_len) > token_budget):
            batches.append(current)
            current = []
            current_tokens = 0
        current.append(seq_id)
        current_tokens += token_len
    if current:
        batches.append(current)

    out: dict[str, torch.Tensor] = {}

    def run_batch(batch_ids: list[str]) -> None:
        data = [("protein", seq_id_to_seq[seq_id]) for seq_id in batch_ids]
        _labels, _seqs, batch_tokens = batch_converter(data)
        if batch_tokens.shape[1] > max_length:
            batch_tokens = batch_tokens[:, :max_length]
        if not use_cpu:
            batch_tokens = batch_tokens.to(next(model.parameters()).device, non_blocking=True)

        with torch.inference_mode():
            results = model(batch_tokens, repr_layers=[33])
        token_representations = results["representations"][33].detach().cpu()

        for row_idx, seq_id in enumerate(batch_ids):
            # Match get_single_esm_repr() semantics under truncation.
            seq_len = len(seq_id_to_seq[seq_id])
            max_residues = token_representations.shape[1] - 1
            valid_len = min(seq_len, max_residues)
            out[seq_id] = token_representations[row_idx, 1 : 1 + valid_len].contiguous()

    def run_batch_with_retry(batch_ids: list[str]) -> None:
        try:
            run_batch(batch_ids)
        except RuntimeError as exc:
            if (
                "out of memory" in str(exc).lower()
                and not use_cpu
                and len(batch_ids) > 1
            ):
                torch.cuda.empty_cache()
                mid = len(batch_ids) // 2
                run_batch_with_retry(batch_ids[:mid])
                run_batch_with_retry(batch_ids[mid:])
            else:
                raise

    for batch_ids in batches:
        run_batch_with_retry(batch_ids)

    return out


# ── Cache path helper ─────────────────────────────────────────────────────────

def _cache_path(cache_root: Path, parameter: str, model_key: str, seq_id: str) -> Path:
    return (cache_root / parameter / model_key / f"{seq_id}.pt").resolve()


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    from tools.executor.policy import require_cuda

    require_cuda(torch)
    parser = argparse.ArgumentParser(
        description="Precompute CatPred attentively-pooled embeddings on GPU."
    )
    parser.add_argument(
        "--seq-id-to-seq-file",
        required=True,
        metavar="JSON",
        help="Path to a JSON file mapping {seq_id: amino_acid_sequence}.",
    )
    parser.add_argument(
        "--parameter",
        required=True,
        choices=["kcat", "km"],
        help="Kinetic parameter — determines which checkpoint sub-directory to use.",
    )
    parser.add_argument(
        "--checkpoint-root",
        required=True,
        metavar="DIR",
        help="Root directory whose <parameter>/ sub-directory contains model.pt files.",
    )
    parser.add_argument(
        "--cache-root",
        required=True,
        metavar="DIR",
        help=(
            "Root of the shared embedding cache "
            "(media/sequence_info/catpred_esm2 on the production server)."
        ),
    )
    args = parser.parse_args()

    seq_id_to_seq: dict[str, str] = json.loads(
        Path(args.seq_id_to_seq_file).read_text(encoding="utf-8")
    )
    parameter    = args.parameter
    checkpoint_root = Path(args.checkpoint_root).resolve()
    cache_root   = Path(args.cache_root).resolve()
    checkpoint_dir  = checkpoint_root / parameter

    if not seq_id_to_seq:
        print("No sequences provided — nothing to do.")
        return

    device = torch.device("cuda")
    print(f"CatPred embedding precompute: device={device}, parameter={parameter}, "
          f"sequences={len(seq_id_to_seq)}")

    checkpoint_models = _discover_checkpoint_models(checkpoint_dir)
    print(f"Discovered {len(checkpoint_models)} checkpoint(s) under {checkpoint_dir}.")

    # ── Determine which (seq_id, model_key) pairs still need computation ──────
    seq_ids = list(seq_id_to_seq.keys())
    cache_dir_by_model: dict[str, Path] = {
        model_key: (cache_root / parameter / model_key).resolve()
        for model_key, _ in checkpoint_models
    }
    missing_by_model: dict[str, list[str]] = {}
    for model_key, _ in checkpoint_models:
        missing_ids, _ready_ids = resolve_missing_ids(
            seq_ids,
            cache_dir=cache_dir_by_model[model_key],
            suffix=".pt",
        )
        if missing_ids:
            missing_by_model[model_key] = missing_ids

    missing_seq_ids: set[str] = {
        sid for sids in missing_by_model.values() for sid in sids
    }

    if not missing_seq_ids:
        print("All embeddings are already cached — nothing to do.")
        return

    print(f"Computing ESM2 representations for {len(missing_seq_ids)} sequence(s)...")
    esm_by_seq_id = _compute_esm_reprs_batched(
        seq_id_to_seq=seq_id_to_seq,
        seq_ids=sorted(missing_seq_ids),
        device=device,
    )

    # ── Per-checkpoint attentive pooling ──────────────────────────────────────
    async_workers = max(1, int(os.environ.get("GPU_EMBED_CACHE_ASYNC_WORKERS", "8")))
    spool_dir = Path(os.environ.get("GPU_EMBED_CACHE_SPOOL_DIR", "/dev/shm/webkinpred-gpu-cache"))
    spool_fallback = Path(os.environ.get("GPU_EMBED_CACHE_SPOOL_FALLBACK_DIR", "/tmp/webkinpred-gpu-cache"))
    with SpoolAsyncCommitter(
        max_workers=async_workers,
        spool_dir=spool_dir,
        spool_fallback_dir=spool_fallback,
    ) as committer:
        for model_key, checkpoint_path in checkpoint_models:
            pending = sorted(set(missing_by_model.get(model_key, [])))
            if not pending:
                print(f"  [{model_key}] all cached — skipping model load.")
                continue

            print(f"  [{model_key}] computing pooled embeddings for {len(pending)} sequence(s)...")
            model = load_checkpoint(str(checkpoint_path), device=device)
            model.eval()
            model_cache_dir = cache_dir_by_model[model_key]

            for batch_start in range(0, len(pending), _POOL_BATCH):
                batch = pending[batch_start : batch_start + _POOL_BATCH]
                with torch.no_grad():
                    for seq_id in batch:
                        pooled = _compute_seq_pooled_output(
                            model=model,
                            sequence=seq_id_to_seq[seq_id],
                            seq_id=seq_id,
                            esm_feature=esm_by_seq_id[seq_id],
                            device=device,
                        )
                        committer.submit_torch_tensor(
                            cache_dir=model_cache_dir,
                            seq_id=seq_id,
                            tensor=pooled,
                        )
                if device.type == "cuda":
                    torch.cuda.empty_cache()

            del model
            if device.type == "cuda":
                torch.cuda.empty_cache()

    total_saved = sum(len(v) for v in missing_by_model.values())
    print(f"Done. Saved {total_saved} embedding file(s) to {cache_root / parameter}.")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"[catpred_embed_gpu] ERROR: {exc}", file=sys.stderr, flush=True)
        raise SystemExit(1) from exc
