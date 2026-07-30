#!/usr/bin/env python3
"""Import exact CatRange fold partitions into the standalone benchmark workspace."""

import argparse
from pathlib import Path

import pandas as pd

from benchmark_utils import make_output_dir, safe_log10


_PROCESSED_SUBDIR = {
    "esmc": "esmc_1152d",
}

_MASTER_RAW_FILE = "data/raw/WT_MD_database_v1_curated_no_OOD_nodups_esmc.xlsx"


def parse_args():
    parser = argparse.ArgumentParser(description="Import exact CatRange partitions for benchmarking.")
    parser.add_argument("--realkcat-root", required=True, help="Path to the CatRange repository root.")
    parser.add_argument("--parameter", required=True, choices=["kcat", "km"], help="Partition parameter to import.")
    parser.add_argument("--fold", required=True, type=int, choices=[1, 2, 3, 4, 5], help="Fold number.")
    parser.add_argument("--embedding", default="esmc", choices=["esmc"], help="Embedding partition set to use.")
    parser.add_argument("--output-dir", required=True, help="Where benchmark-ready files should be written.")
    return parser.parse_args()


def _partition_workbook(realkcat_root, parameter, fold, embedding):
    subdir = _PROCESSED_SUBDIR[embedding]
    return (
        Path(realkcat_root)
        / "data"
        / "processed"
        / subdir
        / "fold_{}_{}".format(fold, parameter)
        / "fold{}_partitions_and_thresholds_by_length.xlsx".format(fold)
    )


def _load_split_sheet(workbook_path, sheet_name, split_name):
    df = pd.read_excel(workbook_path, sheet_name=sheet_name)
    df["split"] = split_name
    return df


def _build_pair_id(row):
    source = str(row.get("source", "")).strip()
    split = str(row.get("split", "")).strip()
    original_idx = int(row["original_df_idx"])
    return "{}_{}_idx{:06d}".format(split, source or "UNK", original_idx)


def _write_catpred_files(df, outdir):
    all_df = pd.DataFrame(
        {
            "pair_id": df["pair_id"],
            "split": df["split"],
            "SMILES": df["smiles"],
            "sequence": df["sequence"],
            "pdbpath": df["pair_id"],
            "substrate_name": df["substrate_name"],
        }
    )
    all_df.to_csv(outdir / "catpred_input.csv", index=False)
    for split_name in ["train", "val", "test"]:
        split_df = all_df[all_df["split"] == split_name].copy()
        split_df.to_csv(outdir / "catpred_input_{}.csv".format(split_name), index=False)


def _write_dlkcat_files(df, outdir):
    all_df = pd.DataFrame(
        {
            "Substrate Name": df["substrate_name"],
            "Substrate SMILES": df["smiles"],
            "Protein Sequence": df["sequence"],
            "pair_id": df["pair_id"],
            "split": df["split"],
        }
    )
    all_df.to_csv(outdir / "dlkcat_input.tsv", sep="\t", index=False)
    for split_name in ["train", "val", "test"]:
        split_df = all_df[all_df["split"] == split_name].copy()
        split_df.to_csv(outdir / "dlkcat_input_{}.tsv".format(split_name), sep="\t", index=False)


def main():
    args = parse_args()

    realkcat_root = Path(args.realkcat_root)
    workbook_path = _partition_workbook(realkcat_root, args.parameter, args.fold, args.embedding)
    master_path = realkcat_root / _MASTER_RAW_FILE

    if not workbook_path.exists():
        raise FileNotFoundError("CatRange partition workbook not found: {}".format(workbook_path))
    if not master_path.exists():
        raise FileNotFoundError("CatRange master metadata workbook not found: {}".format(master_path))

    outdir = make_output_dir(args.output_dir)

    part_train = _load_split_sheet(workbook_path, "train_partition", "train")
    part_val = _load_split_sheet(workbook_path, "val_partition", "val")
    part_test = _load_split_sheet(workbook_path, "test_partition", "test")
    partitions = pd.concat([part_train, part_val, part_test], ignore_index=True)

    master_df = pd.read_excel(master_path, sheet_name="kcat_km_entries")
    master_df = master_df.reset_index().rename(columns={"index": "original_df_idx"})

    meta_cols = [
        "original_df_idx",
        "ECNumber",
        "Substrate",
        "Organism",
        "Unit_kcat",
        "Unit_km",
        "mutantSites_kcat",
        "mutantSites_km",
        "Type",
        "EnzymeType",
        "ECPathway",
        "pH",
        "Temperature",
        "Source",
        "Smiles",
        "Isomeric_SMILES",
        "sequence",
        "source",
    ]
    meta_cols = [col for col in meta_cols if col in master_df.columns]
    metadata = master_df[meta_cols].copy()

    merged = partitions.merge(metadata, on="original_df_idx", how="left", suffixes=("", "_meta"))
    merged["pair_id"] = merged.apply(_build_pair_id, axis=1)
    merged["parameter"] = args.parameter
    merged["embedding"] = args.embedding
    merged["fold"] = args.fold

    if args.parameter == "kcat":
        merged["true_value"] = pd.to_numeric(merged["kcat_Value"], errors="coerce")
        merged["true_unit"] = merged["Unit_kcat"].fillna("s^(-1)")
    else:
        merged["true_value"] = pd.to_numeric(merged["km_Value"], errors="coerce")
        merged["true_unit"] = merged["Unit_km"].fillna("M")

    merged["true_log10"] = safe_log10(merged["true_value"])

    standardized = pd.DataFrame(
        {
            "pair_id": merged["pair_id"],
            "split": merged["split"],
            "parameter": merged["parameter"],
            "embedding": merged["embedding"],
            "fold": merged["fold"],
            "source_partition": merged["source"],
            "original_df_idx": merged["original_df_idx"],
            "ecnumber": merged["ECNumber"],
            "organism": merged["Organism"],
            "substrate_name": merged["Substrate"],
            "smiles": merged["Isomeric_SMILES"],
            "sequence": merged["sequence"],
            "true_value": merged["true_value"],
            "true_log10": merged["true_log10"],
            "true_unit": merged["true_unit"],
            "enzyme_type": merged["EnzymeType"] if "EnzymeType" in merged.columns else "",
            "assay_source": merged["Source"] if "Source" in merged.columns else "",
            "mutant_sites": merged["mutantSites_kcat"] if args.parameter == "kcat" and "mutantSites_kcat" in merged.columns else (
                merged["mutantSites_km"] if args.parameter == "km" and "mutantSites_km" in merged.columns else ""
            ),
        }
    )

    master_out = outdir / "realkcat_partition_master.csv"
    truth_out = outdir / "truth_standardized.csv"
    standardized.to_csv(master_out, index=False)
    standardized.to_csv(truth_out, index=False)

    _write_catpred_files(standardized, outdir)
    if args.parameter == "kcat":
        _write_dlkcat_files(standardized, outdir)

    print("Imported CatRange partitions from:", workbook_path)
    print("Wrote master table:            ", master_out)
    print("Wrote truth table:             ", truth_out)
    print("Split counts:")
    print(standardized["split"].value_counts().sort_index().to_string())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
