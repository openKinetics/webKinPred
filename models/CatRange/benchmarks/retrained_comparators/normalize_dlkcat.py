#!/usr/bin/env python3
"""Normalize DLKcat output into the shared benchmark schema."""

import argparse
from pathlib import Path

import pandas as pd

from benchmark_constants import DLKCAT_OUTPUT_COLUMN
from benchmark_utils import read_table, resolve_pair_ids, safe_log10


def parse_args():
    parser = argparse.ArgumentParser(description="Normalize DLKcat output for benchmarking.")
    parser.add_argument("--input-tsv", required=True, help="DLKcat input TSV used for prediction.")
    parser.add_argument("--output-tsv", required=True, help="DLKcat prediction output TSV.")
    parser.add_argument("--output-csv", required=True, help="Where to write the standardized CSV.")
    parser.add_argument("--model-name", default="DLKcat", help="Model name written to the standardized output.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    input_df = read_table(args.input_tsv, sep="\t")
    output_df = read_table(args.output_tsv, sep="\t")

    if len(input_df) != len(output_df):
        raise ValueError(
            "DLKcat input/output row counts do not match: "
            f"{len(input_df)} input rows vs {len(output_df)} output rows."
        )

    pair_ids = resolve_pair_ids(
        input_df,
        sequence_col="Protein Sequence",
        smiles_col="Substrate SMILES",
        substrate_name_col="Substrate Name",
        pair_id_col="pair_id",
    )

    pred_value = pd.to_numeric(output_df[DLKCAT_OUTPUT_COLUMN], errors="coerce")

    standardized = pd.DataFrame(
        {
            "pair_id": pair_ids,
            "parameter": "kcat",
            "model_name": args.model_name,
            "sequence": input_df["Protein Sequence"].astype(str),
            "smiles": input_df["Substrate SMILES"].astype(str),
            "substrate_name": input_df["Substrate Name"].astype(str),
            "pred_value": pred_value,
            "pred_log10": safe_log10(pred_value),
            "pred_unit": "s^(-1)",
        }
    )

    output_path = Path(args.output_csv)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    standardized.to_csv(output_path, index=False)
    print(f"Wrote standardized DLKcat output: {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
