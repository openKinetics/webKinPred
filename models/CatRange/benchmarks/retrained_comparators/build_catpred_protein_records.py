#!/usr/bin/env python3
"""Precompute CatPred ESM2 features and write per-set protein record files."""

import argparse
import gzip
import json
from pathlib import Path

import esm
import pandas as pd
import torch


ESM_MAX_LENGTH = 2048


def parse_args():
    parser = argparse.ArgumentParser(description="Build CatPred protein record files with cached ESM2 tensors.")
    parser.add_argument("--suite-dir", required=True, help="Benchmark suite directory with suite_manifest.csv.")
    parser.add_argument(
        "--cache-dir",
        default=None,
        help="Optional directory for saved ESM2 tensors. Default: <suite-dir>/catpred_esm2_cache",
    )
    parser.add_argument("--device", default="auto", help="Device for ESM2 embedding: auto, cuda, or cpu.")
    parser.add_argument("--max-batch-size", type=int, default=4, help="Maximum number of sequences per batch.")
    parser.add_argument("--max-batch-tokens", type=int, default=6000, help="Maximum padded tokens per batch.")
    return parser.parse_args()


def _choose_device(raw_device: str) -> torch.device:
    if raw_device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(raw_device)


def _batch_records(records, max_batch_size: int, max_batch_tokens: int):
    sorted_records = sorted(records, key=lambda row: len(row["sequence"]))
    batch = []
    padded_len = 0

    for record in sorted_records:
        seq_tokens = min(len(record["sequence"]) + 2, ESM_MAX_LENGTH)
        next_padded_len = max(padded_len, seq_tokens)
        next_batch_size = len(batch) + 1
        next_cost = next_padded_len * next_batch_size

        if batch and (next_batch_size > max_batch_size or next_cost > max_batch_tokens):
            yield batch
            batch = []
            padded_len = 0
            next_padded_len = seq_tokens

        batch.append(record)
        padded_len = next_padded_len

    if batch:
        yield batch


def _compute_and_store_batch(batch, model, batch_converter, device: torch.device, cache_dir: Path) -> None:
    data = [(row["sequence_id"], row["sequence"]) for row in batch]
    _, _, batch_tokens = batch_converter(data)
    batch_tokens = batch_tokens[:, :ESM_MAX_LENGTH]
    if device.type == "cuda":
        batch_tokens = batch_tokens.to(device)

    with torch.no_grad():
        results = model(batch_tokens, repr_layers=[33])

    token_representations = results["representations"][33]
    for idx, row in enumerate(batch):
        out_path = cache_dir / f"{row['sequence_id']}.pt"
        seq_len = len(row["sequence"])
        embedding = token_representations[idx][1 : seq_len + 1].detach().cpu()
        torch.save(embedding, out_path)


def _write_records_file(set_dir: Path, catpred_input: pd.DataFrame, cache_dir: Path) -> Path:
    records = {}
    subset = catpred_input[["pdbpath", "sequence"]].drop_duplicates().sort_values("pdbpath")
    for row in subset.itertuples(index=False):
        cache_file = cache_dir / f"{row.pdbpath}.pt"
        if not cache_file.exists():
            raise FileNotFoundError(f"Missing cached ESM2 feature: {cache_file}")
        records[str(row.pdbpath)] = {
            "name": str(row.pdbpath),
            "seq": str(row.sequence),
            "esm2_feats_path": str(cache_file.resolve()),
        }

    out_path = set_dir / "protein_records.json.gz"
    with gzip.open(out_path, "wt", encoding="utf-8") as handle:
        json.dump(records, handle)
    return out_path


def main() -> int:
    args = parse_args()

    suite_dir = Path(args.suite_dir).resolve()
    manifest_path = suite_dir / "suite_manifest.csv"
    if not manifest_path.exists():
        raise FileNotFoundError(f"Suite manifest not found: {manifest_path}")

    cache_dir = Path(args.cache_dir).resolve() if args.cache_dir else (suite_dir / "catpred_esm2_cache").resolve()
    cache_dir.mkdir(parents=True, exist_ok=True)

    manifest_df = pd.read_csv(manifest_path)
    unique_records = {}
    for row in manifest_df.itertuples(index=False):
        catpred_input = pd.read_csv(row.catpred_input_csv)
        for rec in catpred_input[["pdbpath", "sequence"]].drop_duplicates().itertuples(index=False):
            unique_records[str(rec.pdbpath)] = {
                "sequence_id": str(rec.pdbpath),
                "sequence": str(rec.sequence),
            }

    pending = []
    for record in unique_records.values():
        if not (cache_dir / f"{record['sequence_id']}.pt").exists():
            pending.append(record)

    device = _choose_device(args.device)
    print(f"Unique CatPred sequences: {len(unique_records)}")
    print(f"Pending ESM2 cache entries: {len(pending)}")
    print(f"Embedding device: {device}")

    if pending:
        model, alphabet = esm.pretrained.esm2_t33_650M_UR50D()
        batch_converter = alphabet.get_batch_converter()
        model.eval()
        if device.type == "cuda":
            model = model.to(device)

        processed = 0
        for batch in _batch_records(pending, args.max_batch_size, args.max_batch_tokens):
            _compute_and_store_batch(batch, model, batch_converter, device, cache_dir)
            processed += len(batch)
            print(f"Cached {processed}/{len(pending)} ESM2 embeddings")

    protein_record_paths = []
    for row in manifest_df.itertuples(index=False):
        set_dir = Path(row.truth_csv).resolve().parent
        catpred_input = pd.read_csv(row.catpred_input_csv)
        record_path = _write_records_file(set_dir, catpred_input, cache_dir)
        protein_record_paths.append(str(record_path.resolve()))
        print(f"Wrote protein records for {row.set_name}: {record_path}")

    manifest_df["protein_records_json"] = protein_record_paths
    manifest_df.to_csv(manifest_path, index=False)
    print(f"Updated suite manifest: {manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
