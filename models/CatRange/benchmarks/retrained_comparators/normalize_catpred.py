#!/usr/bin/env python3
"""Normalize CatPred output into the shared benchmark schema."""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from benchmark_constants import CATPRED_LINEAR_COLUMNS, CATPRED_LOG_COLUMNS
from benchmark_utils import make_output_dir, read_table, resolve_pair_ids, safe_log10


_REALKCAT_TARGET_UNITS = {
    "kcat": "s^(-1)",
    "km": "M",
    "ki": "M",
}


def _convert_catpred_to_realkcat_units(pred_value, pred_log10, parameter):
    """CatPred reports km/ki in mM; CatRange bins km in M."""
    parameter = parameter.lower()
    if parameter in {"km", "ki"}:
        return pred_value / 1000.0, pred_log10 - 3.0
    return pred_value, pred_log10


def parse_args():
    parser = argparse.ArgumentParser(description="Normalize CatPred output for benchmarking.")
    parser.add_argument("--input-csv", required=True, help="CatPred output CSV.")
    parser.add_argument("--output-csv", required=True, help="Where to write the standardized CSV.")
    parser.add_argument("--parameter", required=True, choices=["kcat", "km", "ki"], help="Predicted parameter.")
    parser.add_argument("--pair-id-column", default="pair_id", help="Existing pair_id column if present.")
    parser.add_argument("--sequence-column", default="sequence", help="Sequence column in CatPred output.")
    parser.add_argument("--smiles-column", default="SMILES", help="SMILES column in CatPred output.")
    parser.add_argument("--substrate-name-column", default="substrate_name", help="Optional substrate-name column.")
    parser.add_argument("--pdbpath-column", default="pdbpath", help="Optional pdbpath column.")
    parser.add_argument("--model-name", default="CatPred", help="Model name written to the standardized output.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    df = read_table(args.input_csv)

    pair_ids = resolve_pair_ids(
        df,
        sequence_col=args.sequence_column,
        smiles_col=args.smiles_column,
        substrate_name_col=args.substrate_name_column,
        pair_id_col=args.pair_id_column,
    )

    linear_col = CATPRED_LINEAR_COLUMNS[args.parameter]
    log_col = CATPRED_LOG_COLUMNS[args.parameter]

    if linear_col in df.columns:
        pred_value = pd.to_numeric(df[linear_col], errors="coerce")
    elif log_col in df.columns:
        pred_value = pd.Series(np.power(10.0, pd.to_numeric(df[log_col], errors="coerce")), index=df.index, dtype=float)
    else:
        raise ValueError(
            f"Could not find CatPred prediction columns for {args.parameter}. "
            f"Expected one of: {linear_col}, {log_col}"
        )

    pred_log10 = pd.to_numeric(df[log_col], errors="coerce") if log_col in df.columns else safe_log10(pred_value)
    pred_value, pred_log10 = _convert_catpred_to_realkcat_units(pred_value, pred_log10, args.parameter)

    out_df = pd.DataFrame(
        {
            "pair_id": pair_ids,
            "parameter": args.parameter,
            "model_name": args.model_name,
            "sequence": df[args.sequence_column].astype(str),
            "smiles": df[args.smiles_column].astype(str),
            "substrate_name": df[args.substrate_name_column].astype(str) if args.substrate_name_column in df.columns else "",
            "pdbpath": df[args.pdbpath_column].astype(str) if args.pdbpath_column in df.columns else "",
            "pred_value": pred_value,
            "pred_log10": pred_log10,
            "pred_unit": _REALKCAT_TARGET_UNITS[args.parameter],
            "sd_total": pd.to_numeric(df["SD_total"], errors="coerce") if "SD_total" in df.columns else np.nan,
            "sd_aleatoric": pd.to_numeric(df["SD_aleatoric"], errors="coerce") if "SD_aleatoric" in df.columns else np.nan,
            "sd_epistemic": pd.to_numeric(df["SD_epistemic"], errors="coerce") if "SD_epistemic" in df.columns else np.nan,
        }
    )

    out_file = Path(args.output_csv)
    make_output_dir(out_file.parent)
    pd.DataFrame(out_df).to_csv(out_file, index=False)
    print(f"Wrote standardized CatPred output: {out_file}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
