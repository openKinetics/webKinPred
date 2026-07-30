#!/usr/bin/env python3
"""Evaluate standardized external-model predictions with CatRange-style binning."""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from benchmark_constants import PARAMETER_BIN_EDGES
from benchmark_metrics import confusion_matrix_table, evaluate_classification
from benchmark_utils import bin_continuous_values, make_output_dir, read_table, slugify


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate CatPred/DLKcat predictions with CatRange-style metrics.")
    parser.add_argument("--truth-csv", required=True, help="Standardized truth CSV from prepare_benchmark_inputs.py.")
    parser.add_argument(
        "--pred-csv",
        required=True,
        action="append",
        help="Standardized prediction CSV. Repeat for multiple models.",
    )
    parser.add_argument("--output-dir", required=True, help="Directory for metric outputs.")
    parser.add_argument(
        "--split",
        action="append",
        default=[],
        help="Optional split filter applied to truth rows, e.g. --split test. Repeat to include multiple splits.",
    )
    return parser.parse_args()


def load_truth(path):
    truth_df = read_table(path)
    required = ["pair_id", "parameter", "true_value"]
    missing = [col for col in required if col not in truth_df.columns]
    if missing:
        raise ValueError(f"Truth file is missing required columns: {', '.join(missing)}")
    truth_df = truth_df.copy()
    truth_df["true_bin"] = bin_continuous_values(truth_df["true_value"], truth_df["parameter"].iloc[0])
    return truth_df


def load_prediction(path):
    pred_df = read_table(path)
    required = ["pair_id", "parameter"]
    missing = [col for col in required if col not in pred_df.columns]
    if missing:
        raise ValueError(f"Prediction file {path} is missing required columns: {', '.join(missing)}")
    pred_df = pred_df.copy()
    if "pred_bin" in pred_df.columns:
        pred_df["pred_bin"] = pd.to_numeric(pred_df["pred_bin"], errors="coerce").astype("Float64")
    elif "pred_value" in pred_df.columns:
        pred_df["pred_bin"] = bin_continuous_values(pred_df["pred_value"], pred_df["parameter"].iloc[0])
    else:
        raise ValueError(
            f"Prediction file {path} must contain either 'pred_value' or 'pred_bin'."
        )
    return pred_df


def main() -> int:
    args = parse_args()

    outdir = make_output_dir(args.output_dir)
    matched_dir = make_output_dir(outdir / "matched_tables")
    conf_dir = make_output_dir(outdir / "confusion_matrices")

    truth_df = load_truth(args.truth_csv)
    if args.split:
        if "split" not in truth_df.columns:
            raise ValueError("Requested --split filtering but truth CSV has no 'split' column.")
        split_values = [value.lower() for value in args.split]
        truth_df = truth_df[truth_df["split"].astype(str).str.lower().isin(split_values)].copy()
        if truth_df.empty:
            raise ValueError(f"No truth rows remain after split filtering: {split_values}")
    parameter = str(truth_df["parameter"].iloc[0]).lower()
    if parameter not in PARAMETER_BIN_EDGES:
        raise ValueError(f"Unsupported parameter in truth file: {parameter}")

    summary_rows = []

    for pred_path in args.pred_csv:
        pred_df = load_prediction(pred_path)
        pred_parameter = str(pred_df["parameter"].iloc[0]).lower()
        if pred_parameter != parameter:
            raise ValueError(
                f"Parameter mismatch for {pred_path}: truth is {parameter}, predictions are {pred_parameter}"
            )

        merged = truth_df.merge(
            pred_df,
            on="pair_id",
            how="inner",
            suffixes=("_truth", "_pred"),
        )
        required_merge_cols = ["true_bin", "pred_bin"]
        if "true_value" in merged.columns:
            required_merge_cols.append("true_value")
        if "pred_value" in merged.columns:
            required_merge_cols.append("pred_value")
        merged = merged.dropna(subset=required_merge_cols).copy()

        model_name = (
            str(pred_df["model_name"].dropna().iloc[0])
            if "model_name" in pred_df.columns and pred_df["model_name"].notna().any()
            else Path(pred_path).stem
        )
        model_slug = slugify(model_name)

        matched_rows = len(merged)
        truth_rows = len(truth_df)
        pred_rows = len(pred_df)

        if matched_rows == 0:
            summary_rows.append(
                {
                    "model_name": model_name,
                    "parameter": parameter,
                    "matched_rows": 0,
                    "truth_rows": truth_rows,
                    "prediction_rows": pred_rows,
                    "coverage_vs_truth": 0.0 if truth_rows else np.nan,
                    "unmatched_truth_rows": truth_rows,
                    "extra_prediction_rows": pred_rows,
                    "accuracy": np.nan,
                    "precision": np.nan,
                    "recall": np.nan,
                    "f1": np.nan,
                    "mcc": np.nan,
                    "auc_pr": np.nan,
                    "e_accuracy": np.nan,
                    "n_samples": 0,
                }
            )
            print("Evaluated {}: matched 0 rows".format(model_name))
            continue

        merged["true_bin"] = merged["true_bin"].astype(int)
        merged["pred_bin"] = merged["pred_bin"].astype(int)

        proba_cols = []
        if len(PARAMETER_BIN_EDGES[parameter]) > 1:
            proba_cols = [f"prob_class_{i}" for i in range(len(PARAMETER_BIN_EDGES[parameter]) - 1) if f"prob_class_{i}" in merged.columns]
        y_proba = merged[proba_cols].to_numpy(dtype=float) if proba_cols else None

        metrics = evaluate_classification(
            y_true=merged["true_bin"].to_numpy(dtype=int),
            y_pred=merged["pred_bin"].to_numpy(dtype=int),
            y_proba=y_proba,
        )

        summary_rows.append(
            {
                "model_name": model_name,
                "parameter": parameter,
                "matched_rows": matched_rows,
                "truth_rows": truth_rows,
                "prediction_rows": pred_rows,
                "coverage_vs_truth": matched_rows / truth_rows if truth_rows else np.nan,
                "unmatched_truth_rows": truth_rows - matched_rows,
                "extra_prediction_rows": pred_rows - matched_rows,
                **metrics,
            }
        )

        merged_path = matched_dir / f"{model_slug}_matched.csv"
        merged.to_csv(merged_path, index=False)

        conf_path = conf_dir / f"{model_slug}_confusion_matrix.csv"
        confusion_matrix_table(
            merged["true_bin"].to_numpy(dtype=int),
            merged["pred_bin"].to_numpy(dtype=int),
        ).to_csv(conf_path)

        print(f"Evaluated {model_name}: matched {matched_rows} rows")

    summary_df = pd.DataFrame(summary_rows).sort_values(
        by=["e_accuracy", "accuracy", "mcc"],
        ascending=False,
        na_position="last",
    )

    summary_csv = outdir / "summary_metrics.csv"
    summary_txt = outdir / "summary_metrics.txt"
    summary_df.to_csv(summary_csv, index=False)
    summary_txt.write_text(summary_df.to_string(index=False) + "\n", encoding="utf-8")

    print(f"Saved summary table: {summary_csv}")
    print(f"Saved text summary:  {summary_txt}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
