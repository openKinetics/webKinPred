#!/usr/bin/env python3
"""Build the manuscript benchmark suite from exact CatRange fold sheets."""

import argparse
from pathlib import Path

import pandas as pd

from benchmark_utils import make_output_dir, safe_log10, stable_sequence_ids, write_json


_PROCESSED_SUBDIR = {
    "esmc": "esmc_1152d",
}

_MASTER_RAW_FILE = "data/raw/WT_MD_database_v1_curated_no_OOD_nodups_esmc.xlsx"

_SET_SPECS = [
    {"set_name": "fold1_test", "fold": 1, "sheet_name": "leq_100", "comparison_group": "fold_test"},
    {"set_name": "fold2_test", "fold": 2, "sheet_name": "leq_100", "comparison_group": "fold_test"},
    {"set_name": "fold3_test", "fold": 3, "sheet_name": "leq_100", "comparison_group": "fold_test"},
    {"set_name": "fold4_test", "fold": 4, "sheet_name": "leq_100", "comparison_group": "fold_test"},
    {"set_name": "fold5_test", "fold": 5, "sheet_name": "leq_100", "comparison_group": "fold_test"},
    {"set_name": "fold5_seqid_leq_40", "fold": 5, "sheet_name": "leq_40", "comparison_group": "fold5_seqid"},
    {"set_name": "fold5_seqid_leq_60", "fold": 5, "sheet_name": "leq_60", "comparison_group": "fold5_seqid"},
    {"set_name": "fold5_seqid_leq_80", "fold": 5, "sheet_name": "leq_80", "comparison_group": "fold5_seqid"},
    {"set_name": "fold5_seqid_leq_90", "fold": 5, "sheet_name": "leq_90", "comparison_group": "fold5_seqid"},
    {"set_name": "fold5_seqid_leq_99", "fold": 5, "sheet_name": "leq_99", "comparison_group": "fold5_seqid"},
    {"set_name": "fold5_seqid_leq_100", "fold": 5, "sheet_name": "leq_100", "comparison_group": "fold5_seqid"},
]


def parse_args():
    parser = argparse.ArgumentParser(description="Build the CatRange manuscript benchmark suite.")
    parser.add_argument("--realkcat-root", required=True, help="Path to the CatRange repository root.")
    parser.add_argument("--output-dir", required=True, help="Where the benchmark suite should be written.")
    parser.add_argument("--parameter", default="kcat", choices=["kcat", "km"], help="CatRange parameter.")
    parser.add_argument("--embedding", default="esmc", choices=["esmc"], help="Embedding family.")
    return parser.parse_args()


def _partition_workbook(realkcat_root: Path, parameter: str, fold: int, embedding: str) -> Path:
    subdir = _PROCESSED_SUBDIR[embedding]
    return (
        realkcat_root
        / "data"
        / "processed"
        / subdir
        / f"fold_{fold}_{parameter}"
        / f"fold{fold}_partitions_and_thresholds_by_length.xlsx"
    )


def _sheet_threshold(sheet_name: str):
    if sheet_name.startswith("leq_"):
        return int(sheet_name.split("_", 1)[1])
    return None


def _build_pair_id(fold: int, source: str, original_idx: int) -> str:
    return f"fold{fold}_{source}_idx{int(original_idx):06d}"


def _normalize_partition_df(df: pd.DataFrame) -> pd.DataFrame:
    normalized = df.copy()

    if "sequence" not in normalized.columns and "sequence_test" in normalized.columns:
        normalized["sequence"] = normalized["sequence_test"]

    if "smiles" not in normalized.columns:
        if "Isomeric_SMILES" in normalized.columns:
            normalized["smiles"] = normalized["Isomeric_SMILES"]
        elif "Smiles" in normalized.columns:
            normalized["smiles"] = normalized["Smiles"]

    return normalized


def _merge_metadata(part_df: pd.DataFrame, master_df: pd.DataFrame) -> pd.DataFrame:
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
    merged = part_df.merge(metadata, on="original_df_idx", how="left", suffixes=("", "_meta"))

    if "Substrate" in merged.columns:
        merged["substrate_name"] = merged["Substrate"].fillna("")
    else:
        merged["substrate_name"] = ""

    if "smiles" not in merged.columns or merged["smiles"].isna().all():
        if "Isomeric_SMILES" in merged.columns:
            merged["smiles"] = merged["Isomeric_SMILES"]
        elif "Smiles" in merged.columns:
            merged["smiles"] = merged["Smiles"]

    if "sequence" not in merged.columns or merged["sequence"].isna().all():
        if "sequence_meta" in merged.columns:
            merged["sequence"] = merged["sequence_meta"]

    return merged


def _write_catpred_files(df: pd.DataFrame, outdir: Path) -> Path:
    catpred_df = pd.DataFrame(
        {
            "pair_id": df["pair_id"],
            "split": df["split"],
            "SMILES": df["smiles"],
            "sequence": df["sequence"],
            "pdbpath": df["sequence_id"],
            "substrate_name": df["substrate_name"],
        }
    )
    outpath = outdir / "catpred_input.csv"
    catpred_df.to_csv(outpath, index=False)
    return outpath


def _write_dlkcat_files(df: pd.DataFrame, outdir: Path) -> Path:
    dlkcat_df = pd.DataFrame(
        {
            "Substrate Name": df["substrate_name"],
            "Substrate SMILES": df["smiles"],
            "Protein Sequence": df["sequence"],
            "pair_id": df["pair_id"],
            "split": df["split"],
        }
    )
    outpath = outdir / "dlkcat_input.tsv"
    dlkcat_df.to_csv(outpath, sep="\t", index=False)
    return outpath


def _build_set(
    realkcat_root: Path,
    master_df: pd.DataFrame,
    output_dir: Path,
    parameter: str,
    embedding: str,
    spec: dict,
):
    fold = int(spec["fold"])
    sheet_name = str(spec["sheet_name"])
    workbook_path = _partition_workbook(realkcat_root, parameter, fold, embedding)
    if not workbook_path.exists():
        raise FileNotFoundError(f"Workbook not found: {workbook_path}")

    part_df = pd.read_excel(workbook_path, sheet_name=sheet_name)
    part_df = _normalize_partition_df(part_df)
    merged = _merge_metadata(part_df, master_df)

    merged["pair_id"] = [
        _build_pair_id(fold=fold, source=str(src), original_idx=int(idx))
        for src, idx in zip(merged["source"], merged["original_df_idx"])
    ]
    merged["split"] = "test"
    merged["parameter"] = parameter
    merged["embedding"] = embedding
    merged["fold"] = fold
    merged["sheet_name"] = sheet_name
    merged["comparison_group"] = str(spec["comparison_group"])
    merged["set_name"] = str(spec["set_name"])
    merged["seqid_threshold"] = _sheet_threshold(sheet_name)
    merged["sequence_id"] = stable_sequence_ids(merged["sequence"])

    if parameter == "kcat":
        merged["true_value"] = pd.to_numeric(merged["kcat_Value"], errors="coerce")
        merged["true_unit"] = merged["Unit_kcat"].fillna("s^(-1)")
    else:
        merged["true_value"] = pd.to_numeric(merged["km_Value"], errors="coerce")
        merged["true_unit"] = merged["Unit_km"].fillna("M")

    merged["true_log10"] = safe_log10(merged["true_value"])

    truth_df = pd.DataFrame(
        {
            "pair_id": merged["pair_id"],
            "split": merged["split"],
            "parameter": merged["parameter"],
            "embedding": merged["embedding"],
            "fold": merged["fold"],
            "set_name": merged["set_name"],
            "comparison_group": merged["comparison_group"],
            "sheet_name": merged["sheet_name"],
            "seqid_threshold": merged["seqid_threshold"],
            "source_partition": merged["source"],
            "original_df_idx": merged["original_df_idx"],
            "ecnumber": merged["ECNumber"] if "ECNumber" in merged.columns else "",
            "organism": merged["Organism"] if "Organism" in merged.columns else "",
            "substrate_name": merged["substrate_name"],
            "smiles": merged["smiles"],
            "sequence": merged["sequence"],
            "sequence_id": merged["sequence_id"],
            "true_value": merged["true_value"],
            "true_log10": merged["true_log10"],
            "true_unit": merged["true_unit"],
            "enzyme_type": merged["EnzymeType"] if "EnzymeType" in merged.columns else "",
            "assay_source": merged["Source"] if "Source" in merged.columns else "",
            "mutant_sites": merged["mutantSites_kcat"] if parameter == "kcat" and "mutantSites_kcat" in merged.columns else (
                merged["mutantSites_km"] if parameter == "km" and "mutantSites_km" in merged.columns else ""
            ),
        }
    )

    set_dir = make_output_dir(output_dir / "sets" / spec["set_name"])
    truth_path = set_dir / "truth_standardized.csv"
    truth_df.to_csv(truth_path, index=False)

    catpred_input = _write_catpred_files(truth_df, set_dir)
    dlkcat_input = _write_dlkcat_files(truth_df, set_dir) if parameter == "kcat" else None

    metadata = {
        "set_name": spec["set_name"],
        "comparison_group": spec["comparison_group"],
        "fold": fold,
        "sheet_name": sheet_name,
        "seqid_threshold": _sheet_threshold(sheet_name),
        "n_rows": int(len(truth_df)),
        "n_unique_sequences": int(truth_df["sequence_id"].nunique()),
        "truth_csv": str(truth_path.resolve()),
        "catpred_input_csv": str(catpred_input.resolve()),
        "dlkcat_input_tsv": str(dlkcat_input.resolve()) if dlkcat_input is not None else "",
    }
    write_json(set_dir / "set_metadata.json", metadata)
    return metadata


def main() -> int:
    args = parse_args()

    realkcat_root = Path(args.realkcat_root).resolve()
    output_dir = make_output_dir(args.output_dir)
    master_path = realkcat_root / _MASTER_RAW_FILE
    if not master_path.exists():
        raise FileNotFoundError(f"CatRange master workbook not found: {master_path}")

    master_df = pd.read_excel(master_path, sheet_name="kcat_km_entries")
    master_df = master_df.reset_index().rename(columns={"index": "original_df_idx"})

    rows = []
    for spec in _SET_SPECS:
        row = _build_set(
            realkcat_root=realkcat_root,
            master_df=master_df,
            output_dir=output_dir,
            parameter=args.parameter,
            embedding=args.embedding,
            spec=spec,
        )
        rows.append(row)
        print(
            f"Built {row['set_name']}: n={row['n_rows']} rows, "
            f"{row['n_unique_sequences']} unique sequences"
        )

    manifest_df = pd.DataFrame(rows).sort_values(["comparison_group", "fold", "seqid_threshold", "set_name"])
    manifest_path = output_dir / "suite_manifest.csv"
    manifest_df.to_csv(manifest_path, index=False)

    write_json(
        output_dir / "suite_metadata.json",
        {
            "parameter": args.parameter,
            "embedding": args.embedding,
            "n_sets": int(len(manifest_df)),
            "manifest_csv": str(manifest_path.resolve()),
        },
    )

    print(f"Wrote suite manifest: {manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
