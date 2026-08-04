#!/usr/bin/env python3
"""Run CatRange fold models on the benchmark suite sets."""

import argparse
import sys
from pathlib import Path

import pandas as pd
import torch

from benchmark_constants import PARAMETER_BIN_EDGES
from benchmark_utils import bin_continuous_values, make_output_dir


_PROCESSED_SUBDIR = {
    "esmc": "esmc_1152d",
}


def parse_args():
    parser = argparse.ArgumentParser(description="Predict CatRange outputs for a benchmark suite.")
    parser.add_argument("--realkcat-root", required=True, help="Path to the CatRange repository root.")
    parser.add_argument("--suite-dir", required=True, help="Benchmark suite directory with suite_manifest.csv.")
    parser.add_argument("--parameter", default="kcat", choices=["kcat", "km"], help="CatRange parameter.")
    parser.add_argument("--embedding", default="esmc", choices=["esmc"], help="Embedding family.")
    parser.add_argument("--model-name", default="CatRange", help="Model name written to prediction outputs.")
    return parser.parse_args()


def _load_realkcat_modules(realkcat_root: Path):
    sys.path.insert(0, str(realkcat_root))
    from src.data_pipeline import standardize_array  # pylint: disable=import-error
    from src.model_training import load_model, predict_with_model  # pylint: disable=import-error
    from src.utils import safe_load, tensor_to_numpy  # pylint: disable=import-error

    return standardize_array, load_model, predict_with_model, safe_load, tensor_to_numpy


def _dataset_path(realkcat_root: Path, embedding: str, parameter: str, fold: int, threshold: int) -> Path:
    subdir = _PROCESSED_SUBDIR[embedding]
    return (
        realkcat_root
        / "data"
        / "processed"
        / subdir
        / f"fold_{fold}_{parameter}"
        / f"dataset_y1_y2_seq_idleq_{threshold}.pt"
    )


def _model_paths(realkcat_root: Path, parameter: str, embedding: str, fold: int):
    base = realkcat_root / "outputs" / f"{parameter}_{embedding}" / "models"
    return (
        base / f"{parameter}_{embedding}_fold{fold}.pkl",
        base / f"{parameter}_{embedding}_fold{fold}_stats.pt",
    )


def _load_stats(path: Path):
    stats = torch.load(path, map_location="cpu")
    return (
        stats["mean_1"],
        stats["std_1"],
        stats["mean_2"],
        stats["std_2"],
        int(stats["split_dim"]),
    )


def _full_probability_frame(probabilities, classes, n_classes, index):
    columns = {f"prob_class_{class_idx}": [0.0] * len(index) for class_idx in range(n_classes)}
    if probabilities is None:
        return pd.DataFrame(index=index)
    for pos, class_idx in enumerate(classes):
        columns[f"prob_class_{int(class_idx)}"] = probabilities[:, pos]
    return pd.DataFrame(columns, index=index)


def main() -> int:
    args = parse_args()

    realkcat_root = Path(args.realkcat_root).resolve()
    suite_dir = Path(args.suite_dir).resolve()
    manifest_path = suite_dir / "suite_manifest.csv"
    if not manifest_path.exists():
        raise FileNotFoundError(f"Suite manifest not found: {manifest_path}")

    standardize_array, load_model, predict_with_model, safe_load, tensor_to_numpy = _load_realkcat_modules(realkcat_root)

    manifest_df = pd.read_csv(manifest_path)
    n_classes = len(PARAMETER_BIN_EDGES[args.parameter]) - 1

    for row in manifest_df.itertuples(index=False):
        fold = int(row.fold)
        threshold = int(row.seqid_threshold) if not pd.isna(row.seqid_threshold) else 100
        truth_path = Path(row.truth_csv)
        truth_df = pd.read_csv(truth_path)

        dataset_path = _dataset_path(realkcat_root, args.embedding, args.parameter, fold, threshold)
        model_path, stats_path = _model_paths(realkcat_root, args.parameter, args.embedding, fold)
        if not dataset_path.exists():
            raise FileNotFoundError(f"CatRange dataset not found: {dataset_path}")
        if not model_path.exists():
            raise FileNotFoundError(f"CatRange model not found: {model_path}")
        if not stats_path.exists():
            raise FileNotFoundError(f"CatRange stats not found: {stats_path}")

        dataset = safe_load(dataset_path, device=torch.device("cpu"))
        label_index = 1 if args.parameter == "kcat" else 2
        X = tensor_to_numpy(dataset[0])
        y_tensor = tensor_to_numpy(dataset[label_index])

        truth_bins = pd.to_numeric(
            bin_continuous_values(truth_df["true_value"], args.parameter),
            errors="coerce",
        ).to_numpy(dtype=float)
        if len(truth_df) != len(X):
            raise ValueError(
                f"Length mismatch for {row.set_name}: truth has {len(truth_df)} rows but "
                f"CatRange tensor has {len(X)} rows."
            )
        if not (truth_bins == y_tensor.astype(float)).all():
            mismatch_count = int((truth_bins != y_tensor.astype(float)).sum())
            raise ValueError(
                f"Truth/bin alignment failed for {row.set_name}: {mismatch_count} rows differ "
                "between workbook-derived bins and CatRange labels."
            )

        mean_1, std_1, mean_2, std_2, split_dim = _load_stats(stats_path)
        X_std = standardize_array(X, mean_1, std_1, mean_2, std_2, split_dim)

        model = load_model(model_path)
        pred_bins, probabilities = predict_with_model(model, X_std)
        classes = getattr(model, "classes_", list(range(n_classes)))

        pred_df = pd.DataFrame(
            {
                "pair_id": truth_df["pair_id"].astype(str),
                "parameter": args.parameter,
                "model_name": args.model_name,
                "sequence": truth_df["sequence"].astype(str),
                "smiles": truth_df["smiles"].astype(str),
                "substrate_name": truth_df["substrate_name"].astype(str),
                "pred_bin": pred_bins,
            }
        )
        proba_df = _full_probability_frame(probabilities, classes, n_classes, pred_df.index)
        pred_df = pd.concat([pred_df, proba_df], axis=1)

        pred_dir = make_output_dir(truth_path.parent / "predictions")
        outpath = pred_dir / "realkcat_standardized.csv"
        pred_df.to_csv(outpath, index=False)
        print(f"Wrote CatRange predictions for {row.set_name}: {outpath}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
