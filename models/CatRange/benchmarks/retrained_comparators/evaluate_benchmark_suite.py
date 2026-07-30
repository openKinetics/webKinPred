#!/usr/bin/env python3
"""Evaluate every prediction set in a benchmark suite and aggregate manuscript tables."""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from benchmark_constants import PARAMETER_BIN_EDGES
from benchmark_metrics import confusion_matrix_table, evaluate_classification
from benchmark_utils import make_output_dir, read_table, slugify
from evaluate_benchmark import load_prediction, load_truth


SUMMARY_METRICS = ["accuracy", "e_accuracy", "precision", "recall", "f1", "mcc", "auc_pr"]
MODEL_NAME_ALIASES = {
    "RealKcat": "CatRange",
}
PREFERRED_PREDICTION_FILES = [
    "realkcat_standardized.csv",
    "catpred_standardized.csv",
    "dlkcat_standardized.csv",
    "unikp_standardized.csv",
    "eitlem_standardized.csv",
]


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate a full benchmark suite.")
    parser.add_argument("--suite-dir", required=True, help="Benchmark suite directory with suite_manifest.csv.")
    return parser.parse_args()


def _display_model_name(model_name: str) -> str:
    return MODEL_NAME_ALIASES.get(str(model_name), str(model_name))


def _evaluate_set(truth_csv: Path, pred_paths, outdir: Path):
    truth_df = load_truth(truth_csv)
    parameter = str(truth_df["parameter"].iloc[0]).lower()
    matched_dir = make_output_dir(outdir / "matched_tables")
    conf_dir = make_output_dir(outdir / "confusion_matrices")
    evaluable = []
    common_pair_ids = set(truth_df["pair_id"].astype(str))

    for pred_path in pred_paths:
        pred_df = load_prediction(pred_path)
        pred_parameter = str(pred_df["parameter"].iloc[0]).lower()
        if pred_parameter != parameter:
            raise ValueError(
                f"Parameter mismatch for {pred_path}: truth is {parameter}, predictions are {pred_parameter}"
            )
        model_name = (
            str(pred_df["model_name"].dropna().iloc[0])
            if "model_name" in pred_df.columns and pred_df["model_name"].notna().any()
            else Path(pred_path).stem
        )
        model_name = _display_model_name(model_name)
        merged = truth_df.merge(pred_df, on="pair_id", how="inner", suffixes=("_truth", "_pred"))
        required_merge_cols = ["true_bin", "pred_bin"]
        if "true_value" in merged.columns:
            required_merge_cols.append("true_value")
        if "pred_value" in merged.columns:
            required_merge_cols.append("pred_value")
        merged = merged.dropna(subset=required_merge_cols).copy()
        common_pair_ids &= set(merged["pair_id"].astype(str))
        evaluable.append(
            {
                "pred_path": pred_path,
                "pred_df": pred_df,
                "model_name": model_name,
                "merged": merged,
                "prediction_rows": int(len(pred_df)),
                "evaluable_rows": int(len(merged)),
            }
        )

    comparable_truth = truth_df[truth_df["pair_id"].astype(str).isin(common_pair_ids)].copy()
    comparable_truth.to_csv(outdir / "comparison_truth.csv", index=False)

    summary_rows = []
    truth_rows = len(truth_df)
    comparison_rows = len(comparable_truth)

    for item in evaluable:
        model_name = item["model_name"]
        model_slug = slugify(model_name)
        pred_df = item["pred_df"]
        merged = item["merged"][item["merged"]["pair_id"].astype(str).isin(common_pair_ids)].copy()
        pred_rows = item["prediction_rows"]
        evaluable_rows = item["evaluable_rows"]

        if comparison_rows == 0 or merged.empty:
            summary_rows.append(
                {
                    "model_name": model_name,
                    "parameter": parameter,
                    "matched_rows": 0,
                    "truth_rows": truth_rows,
                    "comparison_rows": comparison_rows,
                    "prediction_rows": pred_rows,
                    "raw_evaluable_rows": evaluable_rows,
                    "coverage_vs_truth": evaluable_rows / truth_rows if truth_rows else np.nan,
                    "comparison_fraction": comparison_rows / truth_rows if truth_rows else np.nan,
                    "unmatched_truth_rows": truth_rows - evaluable_rows,
                    "extra_prediction_rows": pred_rows - evaluable_rows,
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
            continue

        merged["true_bin"] = merged["true_bin"].astype(int)
        merged["pred_bin"] = merged["pred_bin"].astype(int)

        proba_cols = [
            f"prob_class_{idx}"
            for idx in range(len(PARAMETER_BIN_EDGES[parameter]) - 1)
            if f"prob_class_{idx}" in merged.columns
        ]
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
                "matched_rows": len(merged),
                "truth_rows": truth_rows,
                "comparison_rows": comparison_rows,
                "prediction_rows": pred_rows,
                "raw_evaluable_rows": evaluable_rows,
                "coverage_vs_truth": evaluable_rows / truth_rows if truth_rows else np.nan,
                "comparison_fraction": comparison_rows / truth_rows if truth_rows else np.nan,
                "unmatched_truth_rows": truth_rows - evaluable_rows,
                "extra_prediction_rows": pred_rows - evaluable_rows,
                **metrics,
            }
        )

        merged.to_csv(matched_dir / f"{model_slug}_matched.csv", index=False)
        confusion_matrix_table(
            merged["true_bin"].to_numpy(dtype=int),
            merged["pred_bin"].to_numpy(dtype=int),
        ).to_csv(conf_dir / f"{model_slug}_confusion_matrix.csv")

    summary_df = pd.DataFrame(summary_rows).sort_values(
        by=["e_accuracy", "accuracy", "mcc"],
        ascending=False,
        na_position="last",
    )
    summary_df.to_csv(outdir / "summary_metrics.csv", index=False)
    (outdir / "summary_metrics.txt").write_text(summary_df.to_string(index=False) + "\n", encoding="utf-8")
    return summary_df


def _aggregate_mean_std(df: pd.DataFrame, group_name: str) -> pd.DataFrame:
    rows = []
    for model_name, model_df in df.groupby("model_name"):
        for metric in SUMMARY_METRICS:
            rows.append(
                {
                    "group_name": group_name,
                    "model_name": model_name,
                    "metric": metric,
                    "mean": float(model_df[metric].mean()),
                    "std": float(model_df[metric].std(ddof=1)) if len(model_df) > 1 else 0.0,
                    "min": float(model_df[metric].min()),
                    "max": float(model_df[metric].max()),
                    "n_sets": int(len(model_df)),
                }
            )
    return pd.DataFrame(rows)


def _discover_prediction_files(pred_dir: Path):
    preferred = []
    for filename in PREFERRED_PREDICTION_FILES:
        path = pred_dir / filename
        if path.exists():
            preferred.append(path)

    extras = sorted(
        path for path in pred_dir.glob("*_standardized.csv")
        if path.name not in PREFERRED_PREDICTION_FILES
    )
    discovered = preferred + extras
    if not discovered:
        raise FileNotFoundError(
            f"No standardized prediction files found in {pred_dir}. "
            "Expected files ending in '_standardized.csv'."
        )
    return discovered


def main() -> int:
    args = parse_args()

    suite_dir = Path(args.suite_dir).resolve()
    manifest_path = suite_dir / "suite_manifest.csv"
    if not manifest_path.exists():
        raise FileNotFoundError(f"Suite manifest not found: {manifest_path}")

    manifest_df = pd.read_csv(manifest_path)
    suite_results_dir = make_output_dir(suite_dir / "suite_results")

    summary_rows = []
    for row in manifest_df.itertuples(index=False):
        set_dir = Path(row.truth_csv).resolve().parent
        pred_dir = set_dir / "predictions"
        pred_paths = _discover_prediction_files(pred_dir)

        eval_dir = make_output_dir(set_dir / "evaluation")
        summary_df = _evaluate_set(Path(row.truth_csv), pred_paths, eval_dir)
        summary_df["set_name"] = row.set_name
        summary_df["comparison_group"] = row.comparison_group
        summary_df["fold"] = int(row.fold)
        summary_df["seqid_threshold"] = int(row.seqid_threshold) if not pd.isna(row.seqid_threshold) else np.nan
        summary_rows.append(summary_df)
        print(f"Evaluated {row.set_name}: {eval_dir}")

    all_summary = pd.concat(summary_rows, ignore_index=True)
    all_summary.to_csv(suite_results_dir / "all_set_summary.csv", index=False)

    fold_test_df = all_summary[all_summary["comparison_group"] == "fold_test"].copy()
    fold_test_df.to_csv(suite_results_dir / "fold_test_summary.csv", index=False)
    _aggregate_mean_std(fold_test_df, "fold_test").to_csv(
        suite_results_dir / "fold_test_mean_std.csv", index=False
    )

    fold5_test_df = all_summary[all_summary["set_name"] == "fold5_test"].copy()
    fold5_test_df.to_csv(suite_results_dir / "fold5_test_summary.csv", index=False)

    fold5_seqid_df = all_summary[all_summary["comparison_group"] == "fold5_seqid"].copy()
    fold5_seqid_df = fold5_seqid_df.sort_values(["seqid_threshold", "model_name"])
    fold5_seqid_df.to_csv(suite_results_dir / "fold5_seqid_summary.csv", index=False)

    print(f"Wrote suite summaries to {suite_results_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
