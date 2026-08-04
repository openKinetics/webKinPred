#!/usr/bin/env python3
"""
Run the CatPred retrained fold-5 model on the negative-holdout examples.

This script must be run using the catpred_model conda environment:
    /work/ssbio/aosinuga2/envs/catpred_model/bin/python run_catpred_holdout.py \
        --holdout-csv <path> --out-csv <path> --parameter kcat

It will:
1. Compute ESM2 embeddings for any mutant sequences not yet in the cache
2. Run CatPred inference using the retrained fold-5 checkpoint
3. Write predictions to the output CSV
"""

import argparse
import gzip
import hashlib
import json
import os
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

SUITE_DIR = Path(
    "/work/ssbio/aosinuga2/Python_work/EnzymeKinetics_Manuscript_Benchmark"
    "/runs/manuscript_kcat_suite_retrained_same_split"
).resolve()
CATPRED_CHECKPOINT = (
    SUITE_DIR / "catpred_retrained" / "fold5" / "checkpoints" / "fold_0" / "model_0"
).resolve()
CATPRED_CACHE_DIR = (SUITE_DIR / "catpred_esm2_cache").resolve()

CATPRED_ROOT = Path("/work/ssbio/aosinuga2/Python_work/CatPred").resolve()
ESM_MAX_LENGTH = 1024


def _seq_id(sequence: str) -> str:
    return "seq_" + hashlib.sha1(str(sequence).encode()).hexdigest()[:16]


def _ensure_catpred_path():
    if str(CATPRED_ROOT) not in sys.path:
        sys.path.insert(0, str(CATPRED_ROOT))


def _ensure_esm2_embeddings(records, device):
    import torch
    import esm as esm_mod  # type: ignore

    pending = [r for r in records if not (CATPRED_CACHE_DIR / f"{r['sequence_id']}.pt").exists()]
    if not pending:
        print(f"All {len(records)} ESM2 embeddings already cached.")
        return

    print(f"Computing ESM2 embeddings for {len(pending)} sequences …")
    CATPRED_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    model, alphabet = esm_mod.pretrained.esm2_t33_650M_UR50D()
    batch_converter = alphabet.get_batch_converter()
    model.eval()
    if device.type == "cuda":
        model = model.to(device)

    # Process in batches of 1 to avoid OOM on long sequences
    for i, row in enumerate(pending):
        out_path = CATPRED_CACHE_DIR / f"{row['sequence_id']}.pt"
        if out_path.exists():
            continue
        data = [(row["sequence_id"], row["sequence"])]
        _, _, batch_tokens = batch_converter(data)
        batch_tokens = batch_tokens[:, :ESM_MAX_LENGTH]
        if device.type == "cuda":
            batch_tokens = batch_tokens.to(device)
        with torch.no_grad():
            results = model(batch_tokens, repr_layers=[33])
        seq_len = len(row["sequence"])
        embedding = results["representations"][33][0][1:seq_len + 1].detach().cpu()
        torch.save(embedding, out_path)
        print(f"  [{i+1}/{len(pending)}] Cached {row['sequence_id']}")

    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()


def _build_catpred_input_csv(holdout_df: pd.DataFrame, parameter: str, tmp_dir: Path) -> Path:
    """Write a CatPred-compatible input CSV for the holdout rows."""
    target_col = {"kcat": "log10kcat", "km": "log10km"}[parameter]
    rows = []
    for _, row in holdout_df.iterrows():
        rows.append({
            "pair_id": str(row["pair_id"]),
            "SMILES": str(row["smiles"]),
            "sequence": str(row["sequence"]),
            "pdbpath": _seq_id(str(row["sequence"])),
            "substrate_name": str(row.get("substrate_name", "")),
            target_col: 1.0,  # placeholder for mandatory column
        })
    df = pd.DataFrame(rows)
    out_path = tmp_dir / "catpred_input.csv"
    df.to_csv(out_path, index=False)
    return out_path


def _build_records_json(holdout_df: pd.DataFrame, tmp_dir: Path) -> Path:
    """Build protein_records JSON with ESM2 feature paths."""
    records = {}
    for _, row in holdout_df.iterrows():
        seq_id = _seq_id(str(row["sequence"]))
        feat_path = CATPRED_CACHE_DIR / f"{seq_id}.pt"
        if not feat_path.exists():
            raise FileNotFoundError(f"Missing ESM2 cache: {feat_path}")
        records[seq_id] = {
            "name": seq_id,
            "seq": str(row["sequence"]),
            "esm2_feats_path": str(feat_path.resolve()),
        }
    out_path = tmp_dir / "protein_records.json.gz"
    with gzip.open(out_path, "wt", encoding="utf-8") as fh:
        json.dump(records, fh)
    return out_path


def _standardize_catpred_output(raw_df: pd.DataFrame, parameter: str) -> pd.DataFrame:
    """Convert raw CatPred output to standard prediction format."""
    log_candidates = [
        "Prediction_log10",
        "pred_log10kcat",
        "pred_log10km",
        "pred_log10",
        "log10kcat_max",
        "log10kcat",
        "log10km_mean",
        "log10km",
    ]
    out_rows = []
    for _, row in raw_df.iterrows():
        pred_log10 = float("nan")
        for col in log_candidates:
            if col in row.index and pd.notna(row[col]):
                try:
                    pred_log10 = float(row[col])
                    break
                except (TypeError, ValueError):
                    pass
        pred_value = float(10.0 ** pred_log10) if not np.isnan(pred_log10) else float("nan")
        out_rows.append({
            "pair_id": str(row.get("pair_id", "")),
            "model_name": "CatPred",
            "pred_value": pred_value,
            "pred_log10": pred_log10,
            "pred_unit": "s^(-1)" if parameter == "kcat" else "M",
        })
    return pd.DataFrame(out_rows)


def main():
    parser = argparse.ArgumentParser(description="CatPred retrained fold-5 inference on holdout examples.")
    parser.add_argument("--holdout-csv", required=True, help="Input CSV: pair_id, sequence, smiles, substrate_name.")
    parser.add_argument("--out-csv", required=True, help="Output predictions CSV.")
    parser.add_argument("--parameter", default="kcat", choices=["kcat", "km"])
    parser.add_argument("--python-executable",
                        default="/work/ssbio/aosinuga2/envs/catpred_model/bin/python",
                        help="CatPred env python executable.")
    parser.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"])
    args = parser.parse_args()

    import torch
    device_str = args.device
    if device_str == "auto":
        device_str = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device_str)
    print(f"CatPred holdout device: {device}")

    holdout_df = pd.read_csv(args.holdout_csv)
    print(f"Loaded {len(holdout_df)} holdout inputs from {args.holdout_csv}")

    # Build record list for ESM2 caching
    records = [
        {"sequence_id": _seq_id(str(row["sequence"])), "sequence": str(row["sequence"])}
        for _, row in holdout_df.iterrows()
    ]
    _ensure_esm2_embeddings(records, device)

    _ensure_catpred_path()
    from catpred.inference import PredictionRequest, run_prediction_pipeline  # type: ignore

    with tempfile.TemporaryDirectory(prefix="catpred_holdout_") as tmp_str:
        tmp_dir = Path(tmp_str)
        input_csv = _build_catpred_input_csv(holdout_df, args.parameter, tmp_dir)
        records_json = _build_records_json(holdout_df, tmp_dir)
        results_dir = tmp_dir / "results"
        results_dir.mkdir()

        request = PredictionRequest(
            parameter=args.parameter,
            input_file=str(input_csv),
            checkpoint_dir=str(CATPRED_CHECKPOINT),
            use_gpu=(device.type == "cuda"),
            repo_root=str(CATPRED_ROOT),
            python_executable=str(Path(args.python_executable).resolve()),
            protein_records_file=str(records_json),
        )
        produced_output = Path(run_prediction_pipeline(request, results_dir=str(results_dir)))
        raw_df = pd.read_csv(produced_output)

    pred_df = _standardize_catpred_output(raw_df, args.parameter)
    Path(args.out_csv).parent.mkdir(parents=True, exist_ok=True)
    pred_df.to_csv(args.out_csv, index=False)
    print(f"Wrote {len(pred_df)} CatPred predictions → {args.out_csv}")


if __name__ == "__main__":
    main()
