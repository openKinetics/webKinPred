#!/usr/bin/env python3
"""Run CatPred predictions for every set in a benchmark suite."""

import argparse
import gzip
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from benchmark_constants import CATPRED_LINEAR_COLUMNS, CATPRED_LOG_COLUMNS
from benchmark_utils import canonical_protein_mask, make_output_dir, safe_log10, write_json


CATPRED_LOG_COLUMN_ALIASES = {
    "kcat": ("log10kcat",),
    "km": ("log10km",),
    "ki": ("log10ki",),
}


def parse_args():
    parser = argparse.ArgumentParser(description="Run CatPred predictions for a benchmark suite.")
    parser.add_argument("--suite-dir", required=True, help="Benchmark suite directory with suite_manifest.csv.")
    parser.add_argument("--checkpoint-dir", required=True, help="CatPred checkpoint directory.")
    parser.add_argument("--repo-root", required=True, help="Path to the CatPred repository root.")
    parser.add_argument("--parameter", default="kcat", choices=["kcat", "km", "ki"], help="CatPred parameter.")
    parser.add_argument("--use-gpu", action="store_true", help="Run CatPred model inference on GPU if available.")
    parser.add_argument("--overwrite", action="store_true", help="Re-run sets even if standardized outputs already exist.")
    parser.add_argument(
        "--set-name",
        action="append",
        default=[],
        help="Optional set filter. Repeat to run only specific suite sets.",
    )
    return parser.parse_args()


def _convert_catpred_to_realkcat_units(pred_value, pred_log10, parameter):
    if parameter.lower() in {"km", "ki"}:
        return pred_value / 1000.0, pred_log10 - 3.0
    return pred_value, pred_log10


def _standardize(raw_df: pd.DataFrame, parameter: str) -> pd.DataFrame:
    linear_col = CATPRED_LINEAR_COLUMNS[parameter]
    preferred_log_col = CATPRED_LOG_COLUMNS[parameter]
    log_candidates = (preferred_log_col, *CATPRED_LOG_COLUMN_ALIASES.get(parameter, ()))
    log_col = next((col for col in log_candidates if col in raw_df.columns), preferred_log_col)

    if linear_col in raw_df.columns:
        pred_value = pd.to_numeric(raw_df[linear_col], errors="coerce")
    else:
        pred_value = pd.Series(np.power(10.0, pd.to_numeric(raw_df[log_col], errors="coerce")), index=raw_df.index)

    pred_log10 = pd.to_numeric(raw_df[log_col], errors="coerce") if log_col in raw_df.columns else safe_log10(pred_value)
    pred_value, pred_log10 = _convert_catpred_to_realkcat_units(pred_value, pred_log10, parameter)

    return pd.DataFrame(
        {
            "pair_id": raw_df["pair_id"].astype(str),
            "parameter": parameter,
            "model_name": "CatPred",
            "sequence": raw_df["sequence"].astype(str),
            "smiles": raw_df["SMILES"].astype(str),
            "substrate_name": raw_df["substrate_name"].astype(str) if "substrate_name" in raw_df.columns else "",
            "pdbpath": raw_df["pdbpath"].astype(str) if "pdbpath" in raw_df.columns else "",
            "pred_value": pred_value,
            "pred_log10": pred_log10,
            "pred_unit": "s^(-1)" if parameter == "kcat" else "M",
            "sd_total": pd.to_numeric(raw_df["SD_total"], errors="coerce") if "SD_total" in raw_df.columns else np.nan,
            "sd_aleatoric": pd.to_numeric(raw_df["SD_aleatoric"], errors="coerce") if "SD_aleatoric" in raw_df.columns else np.nan,
            "sd_epistemic": pd.to_numeric(raw_df["SD_epistemic"], errors="coerce") if "SD_epistemic" in raw_df.columns else np.nan,
        }
    )


def _prepare_catpred_subset(
    input_csv: Path,
    protein_records_json: Path,
    raw_dir: Path,
    pred_dir: Path,
):
    input_df = pd.read_csv(input_csv)
    valid_mask = canonical_protein_mask(input_df["sequence"].astype(str))
    valid_df = input_df.loc[valid_mask].copy()
    skipped_df = input_df.loc[~valid_mask].copy()

    filtered_input_csv = raw_dir / "catpred_input_valid.csv"
    filtered_records_json = raw_dir / "protein_records_valid.json.gz"
    skipped_csv = pred_dir / "catpred_skipped_rows.csv"

    valid_df.to_csv(filtered_input_csv, index=False)
    if skipped_df.empty:
        if skipped_csv.exists():
            skipped_csv.unlink()
    else:
        skipped_df.to_csv(skipped_csv, index=False)

    with gzip.open(protein_records_json, "rt", encoding="utf-8") as handle:
        all_records = json.load(handle)
    keep_pdbpaths = set(valid_df["pdbpath"].astype(str))
    subset_records = {key: value for key, value in all_records.items() if key in keep_pdbpaths}
    with gzip.open(filtered_records_json, "wt", encoding="utf-8") as handle:
        json.dump(subset_records, handle)

    skipped_examples = []
    if not skipped_df.empty:
        preview_cols = [col for col in ["pair_id", "pdbpath", "sequence"] if col in skipped_df.columns]
        skipped_examples = skipped_df[preview_cols].head(10).to_dict(orient="records")

    coverage = {
        "total_rows": int(len(input_df)),
        "valid_rows": int(len(valid_df)),
        "skipped_rows": int(len(skipped_df)),
        "skipped_fraction": float(len(skipped_df) / len(input_df)) if len(input_df) else 0.0,
        "skipped_examples": skipped_examples,
    }
    write_json(pred_dir / "catpred_coverage.json", coverage)

    return valid_df, filtered_input_csv, filtered_records_json


def main() -> int:
    args = parse_args()

    suite_dir = Path(args.suite_dir).resolve()
    manifest_path = suite_dir / "suite_manifest.csv"
    if not manifest_path.exists():
        raise FileNotFoundError(f"Suite manifest not found: {manifest_path}")

    repo_root = Path(args.repo_root).resolve()
    if not repo_root.exists():
        raise FileNotFoundError(f"CatPred repo not found: {repo_root}")

    sys.path.insert(0, str(repo_root))
    from catpred.inference import PredictionRequest, run_prediction_pipeline  # pylint: disable=import-error

    manifest_df = pd.read_csv(manifest_path)
    if "protein_records_json" not in manifest_df.columns:
        raise ValueError(
            "suite_manifest.csv is missing protein_records_json. "
            "Run build_catpred_protein_records.py first."
        )
    if args.set_name:
        keep = {str(name) for name in args.set_name}
        manifest_df = manifest_df[manifest_df["set_name"].astype(str).isin(keep)].copy()
        if manifest_df.empty:
            raise ValueError(f"No suite rows matched --set-name filters: {sorted(keep)}")

    for row in manifest_df.itertuples(index=False):
        set_dir = Path(row.truth_csv).resolve().parent
        pred_dir = make_output_dir(set_dir / "predictions")
        raw_dir = make_output_dir(set_dir / "catpred_raw")
        raw_output = raw_dir / "catpred_output.csv"
        standardized_output = pred_dir / "catpred_standardized.csv"

        if standardized_output.exists() and not args.overwrite:
            print(f"Skipping {row.set_name}: existing standardized predictions found")
            continue

        valid_df, filtered_input_csv, filtered_records_json = _prepare_catpred_subset(
            Path(row.catpred_input_csv).resolve(),
            Path(row.protein_records_json).resolve(),
            raw_dir,
            pred_dir,
        )
        if valid_df.empty:
            pd.DataFrame(
                columns=[
                    "pair_id",
                    "parameter",
                    "model_name",
                    "sequence",
                    "smiles",
                    "substrate_name",
                    "pdbpath",
                    "pred_value",
                    "pred_log10",
                    "pred_unit",
                    "sd_total",
                    "sd_aleatoric",
                    "sd_epistemic",
                ]
            ).to_csv(standardized_output, index=False)
            print(f"Skipping {row.set_name}: no CatPred-compatible sequences in this set")
            continue

        request = PredictionRequest(
            parameter=args.parameter,
            input_file=str(filtered_input_csv),
            checkpoint_dir=str(Path(args.checkpoint_dir).resolve()),
            use_gpu=bool(args.use_gpu),
            repo_root=str(repo_root),
            python_executable=sys.executable,
            protein_records_file=str(filtered_records_json),
        )
        produced_output = Path(run_prediction_pipeline(request, results_dir=str(raw_dir)))
        if not produced_output.exists():
            raise FileNotFoundError(f"CatPred did not create the expected output: {produced_output}")

        raw_df = pd.read_csv(produced_output)
        raw_df.to_csv(raw_output, index=False)
        standardized_df = _standardize(raw_df, args.parameter)
        standardized_df.to_csv(standardized_output, index=False)
        print(f"Wrote CatPred predictions for {row.set_name}: {standardized_output}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
