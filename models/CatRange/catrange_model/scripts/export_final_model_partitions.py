#!/usr/bin/env python
"""
Export the exact train / val / test partitions used by the final model
(kcat_esmc, Fold 5) to an Excel workbook with sheets:
    Train, Val, Test

Columns per sheet:
    ECNumber, Organism, Smiles, Substrate, Sequence, Value, Unit
"""

import pandas as pd
from pathlib import Path

# ── paths ──────────────────────────────────────────────────────────────
BASE = Path(__file__).resolve().parent.parent          # CatRange/
FOLD5_XLSX = BASE / "data/processed/esmc_1152d/fold_5_kcat/fold5_partitions_and_thresholds_by_length.xlsx"
RAW_XLSX   = BASE / "data/raw/WT_MD_database_v1_curated_no_OOD_nodups_esmc.xlsx"
OUT_XLSX   = BASE / "data/final_model_partitions_kcat_esmc_fold5.xlsx"

# ── load fold-5 partition tables ───────────────────────────────────────
print("Loading fold-5 partition data …")
xls_fold = pd.ExcelFile(FOLD5_XLSX)
df_train = pd.read_excel(xls_fold, "train_partition")
df_val   = pd.read_excel(xls_fold, "val_partition")
df_test  = pd.read_excel(xls_fold, "test_partition")
print(f"  train={len(df_train)}, val={len(df_val)}, test={len(df_test)}")

# ── load master sheet with full metadata ───────────────────────────────
print("Loading master metadata sheet …")
df_master = pd.read_excel(RAW_XLSX, sheet_name="kcat_km_entries")
print(f"  master rows: {len(df_master)}")

# Build a look-up: first occurrence of each (Isomeric_SMILES, sequence)
# keeps ECNumber, Organism, Substrate, Unit_kcat
df_lookup = (
    df_master
    .drop_duplicates(subset=["Isomeric_SMILES", "sequence"], keep="first")
    [["ECNumber", "Organism", "Substrate", "Isomeric_SMILES", "sequence", "Unit_kcat"]]
    .copy()
)
print(f"  unique (SMILES, seq) pairs for look-up: {len(df_lookup)}")


def enrich_and_format(df_part: pd.DataFrame, df_lookup: pd.DataFrame) -> pd.DataFrame:
    """Join partition rows with metadata and format to requested columns."""
    merged = df_part.merge(df_lookup, on=["Isomeric_SMILES", "sequence"], how="left")

    out = pd.DataFrame({
        "ECNumber":  merged["ECNumber"],
        "Organism":  merged["Organism"],
        "Smiles":    merged["Isomeric_SMILES"],
        "Substrate": merged["Substrate"],
        "Sequence":  merged["sequence"],
        "Value":     merged["kcat_Value"],
        "Unit":      merged["Unit_kcat"].fillna("s^(-1)"),
    })
    return out


# ── build output frames ───────────────────────────────────────────────
print("Enriching partitions with metadata …")
out_train = enrich_and_format(df_train, df_lookup)
out_val   = enrich_and_format(df_val,   df_lookup)
out_test  = enrich_and_format(df_test,  df_lookup)

# Sanity check: no missing ECNumbers
for name, df in [("Train", out_train), ("Val", out_val), ("Test", out_test)]:
    n_miss = df["ECNumber"].isna().sum()
    if n_miss:
        print(f"  ⚠  {name}: {n_miss} rows missing ECNumber")
    else:
        print(f"  ✓ {name}: all {len(df)} rows matched")

# ── write workbook ────────────────────────────────────────────────────
print(f"\nWriting workbook → {OUT_XLSX}")
with pd.ExcelWriter(OUT_XLSX, engine="openpyxl") as writer:
    out_train.to_excel(writer, sheet_name="Train", index=False)
    out_val.to_excel(writer,   sheet_name="Val",   index=False)
    out_test.to_excel(writer,  sheet_name="Test",  index=False)

print("Done ✓")
print(f"  Train: {len(out_train)} rows")
print(f"  Val:   {len(out_val)} rows")
print(f"  Test:  {len(out_test)} rows")
print(f"  Total: {len(out_train) + len(out_val) + len(out_test)} rows")
