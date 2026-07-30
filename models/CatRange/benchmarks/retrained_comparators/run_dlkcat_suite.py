#!/usr/bin/env python3
"""Run DLKcat predictions for every set in a benchmark suite."""

import argparse
import shutil
import subprocess
from pathlib import Path

import pandas as pd

from benchmark_constants import DLKCAT_OUTPUT_COLUMN
from benchmark_utils import make_output_dir, safe_log10


def parse_args():
    parser = argparse.ArgumentParser(description="Run DLKcat predictions for a benchmark suite.")
    parser.add_argument("--suite-dir", required=True, help="Benchmark suite directory with suite_manifest.csv.")
    parser.add_argument("--dlkcat-root", required=True, help="Path to the DLKcat repository root.")
    parser.add_argument("--python-executable", required=True, help="Python executable for the DLKcat env.")
    parser.add_argument("--overwrite", action="store_true", help="Re-run sets even if standardized outputs already exist.")
    return parser.parse_args()


def _normalize(input_tsv: Path, output_tsv: Path, output_csv: Path) -> None:
    input_df = pd.read_csv(input_tsv, sep="\t")
    output_df = pd.read_csv(output_tsv, sep="\t")

    if len(input_df) != len(output_df):
        raise ValueError(
            f"DLKcat row mismatch: {len(input_df)} input rows vs {len(output_df)} prediction rows."
        )

    pred_value = pd.to_numeric(output_df[DLKCAT_OUTPUT_COLUMN], errors="coerce")
    standardized = pd.DataFrame(
        {
            "pair_id": input_df["pair_id"].astype(str),
            "parameter": "kcat",
            "model_name": "DLKcat",
            "sequence": input_df["Protein Sequence"].astype(str),
            "smiles": input_df["Substrate SMILES"].astype(str),
            "substrate_name": input_df["Substrate Name"].astype(str),
            "pred_value": pred_value,
            "pred_log10": safe_log10(pred_value),
            "pred_unit": "s^(-1)",
        }
    )
    standardized.to_csv(output_csv, index=False)


def main() -> int:
    args = parse_args()

    suite_dir = Path(args.suite_dir).resolve()
    manifest_path = suite_dir / "suite_manifest.csv"
    if not manifest_path.exists():
        raise FileNotFoundError(f"Suite manifest not found: {manifest_path}")

    dlkcat_root = Path(args.dlkcat_root).resolve()
    example_dir = dlkcat_root / "DeeplearningApproach" / "Code" / "example"
    prediction_script = example_dir / "prediction_for_input.py"
    if not prediction_script.exists():
        raise FileNotFoundError(f"DLKcat prediction script not found: {prediction_script}")

    manifest_df = pd.read_csv(manifest_path)
    for row in manifest_df.itertuples(index=False):
        input_tsv = Path(row.dlkcat_input_tsv)
        set_dir = Path(row.truth_csv).resolve().parent
        pred_dir = make_output_dir(set_dir / "predictions")
        raw_output = pred_dir / "dlkcat_output.tsv"
        standardized_output = pred_dir / "dlkcat_standardized.csv"

        if standardized_output.exists() and not args.overwrite:
            print(f"Skipping {row.set_name}: existing DLKcat predictions found")
            continue

        subprocess.run(
            [args.python_executable, str(prediction_script), str(input_tsv)],
            cwd=str(example_dir),
            check=True,
        )

        generated_output = example_dir / "output.tsv"
        if not generated_output.exists():
            raise FileNotFoundError(f"DLKcat did not create output.tsv in {example_dir}")

        shutil.copy2(generated_output, raw_output)
        _normalize(input_tsv=input_tsv, output_tsv=raw_output, output_csv=standardized_output)
        print(f"Wrote DLKcat predictions for {row.set_name}: {standardized_output}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
