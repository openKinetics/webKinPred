#!/usr/bin/env python3
"""Prepare standardized truth + tool input files for CatPred/DLKcat benchmarking."""

import argparse

import pandas as pd

from benchmark_utils import make_output_dir, read_table, resolve_pair_ids, safe_log10


def parse_args():
    parser = argparse.ArgumentParser(description="Prepare benchmark inputs for CatPred and DLKcat.")
    parser.add_argument("--input-csv", required=True, help="Master benchmark CSV with truth values.")
    parser.add_argument("--output-dir", required=True, help="Directory where standardized files will be written.")
    parser.add_argument("--parameter", required=True, choices=["kcat", "km"], help="Benchmark parameter.")
    parser.add_argument("--truth-column", required=True, help="Column containing the continuous ground-truth value.")
    parser.add_argument("--sequence-column", required=True, help="Protein sequence column.")
    parser.add_argument("--smiles-column", required=True, help="Substrate SMILES column.")
    parser.add_argument("--substrate-name-column", default=None, help="Optional substrate-name column.")
    parser.add_argument("--pair-id-column", default=None, help="Optional explicit pair_id column.")
    parser.add_argument("--pdbpath-column", default=None, help="Optional pdbpath column for CatPred.")
    parser.add_argument(
        "--keep-columns",
        nargs="*",
        default=[],
        help="Optional extra columns to retain in truth_standardized.csv.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    df = read_table(args.input_csv)
    outdir = make_output_dir(args.output_dir)

    substrate_name = (
        df[args.substrate_name_column].astype(str)
        if args.substrate_name_column and args.substrate_name_column in df.columns
        else pd.Series("", index=df.index, dtype=str)
    )

    pair_ids = resolve_pair_ids(
        df,
        sequence_col=args.sequence_column,
        smiles_col=args.smiles_column,
        substrate_name_col=args.substrate_name_column,
        pair_id_col=args.pair_id_column,
    )

    truth_df = pd.DataFrame(
        {
            "pair_id": pair_ids,
            "parameter": args.parameter,
            "sequence": df[args.sequence_column].astype(str),
            "smiles": df[args.smiles_column].astype(str),
            "substrate_name": substrate_name,
            "true_value": pd.to_numeric(df[args.truth_column], errors="coerce"),
        }
    )
    truth_df["true_log10"] = safe_log10(truth_df["true_value"])
    truth_df["pdbpath"] = (
        df[args.pdbpath_column].astype(str)
        if args.pdbpath_column and args.pdbpath_column in df.columns
        else truth_df["pair_id"]
    )

    for col in args.keep_columns:
        if col in df.columns and col not in truth_df.columns:
            truth_df[col] = df[col]

    truth_path = outdir / "truth_standardized.csv"
    truth_df.to_csv(truth_path, index=False)

    catpred_df = pd.DataFrame(
        {
            "pair_id": truth_df["pair_id"],
            "SMILES": truth_df["smiles"],
            "sequence": truth_df["sequence"],
            "pdbpath": truth_df["pdbpath"],
            "substrate_name": truth_df["substrate_name"],
        }
    )
    catpred_path = outdir / "catpred_input.csv"
    catpred_df.to_csv(catpred_path, index=False)

    print(f"Wrote standardized truth table: {truth_path}")
    print(f"Wrote CatPred input CSV:       {catpred_path}")

    if args.parameter == "kcat":
        dlkcat_df = pd.DataFrame(
            {
                "Substrate Name": truth_df["substrate_name"],
                "Substrate SMILES": truth_df["smiles"],
                "Protein Sequence": truth_df["sequence"],
                "pair_id": truth_df["pair_id"],
            }
        )
        dlkcat_path = outdir / "dlkcat_input.tsv"
        dlkcat_df.to_csv(dlkcat_path, sep="\t", index=False)
        print(f"Wrote DLKcat input TSV:        {dlkcat_path}")
    else:
        print("Skipped DLKcat input generation because DLKcat is kcat-only.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
