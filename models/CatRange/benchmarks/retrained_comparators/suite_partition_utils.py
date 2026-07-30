#!/usr/bin/env python3
"""Shared helpers for loading CatRange fold partitions into benchmark runners."""

from pathlib import Path
from typing import Iterable, Optional

import pandas as pd

import build_realkcat_kcat_benchmark_suite as suite_build
from benchmark_utils import safe_log10, stable_sequence_ids


def load_master_metadata(realkcat_root) -> pd.DataFrame:
    """Load the CatRange master workbook with stable original indices."""
    realkcat_root = Path(realkcat_root).resolve()
    master_path = realkcat_root / suite_build._MASTER_RAW_FILE
    if not master_path.exists():
        raise FileNotFoundError(f"CatRange master workbook not found: {master_path}")
    master_df = pd.read_excel(master_path, sheet_name="kcat_km_entries")
    return master_df.reset_index().rename(columns={"index": "original_df_idx"})


def _enrich_partition_df(
    part_df: pd.DataFrame,
    master_df: pd.DataFrame,
    parameter: str,
    fold: int,
    split_name: str,
) -> pd.DataFrame:
    merged = suite_build._normalize_partition_df(part_df)
    merged = suite_build._merge_metadata(merged, master_df)
    merged = merged.copy()
    merged["split"] = split_name
    merged["fold"] = int(fold)
    merged["parameter"] = str(parameter)
    merged["pair_id"] = [
        suite_build._build_pair_id(fold=fold, source=str(src), original_idx=int(idx))
        for src, idx in zip(merged["source"], merged["original_df_idx"])
    ]
    merged["sequence_id"] = stable_sequence_ids(merged["sequence"])

    if parameter == "kcat":
        merged["true_value"] = pd.to_numeric(merged["kcat_Value"], errors="coerce")
        merged["true_unit"] = merged["Unit_kcat"].fillna("s^(-1)")
    else:
        merged["true_value"] = pd.to_numeric(merged["km_Value"], errors="coerce")
        merged["true_unit"] = merged["Unit_km"].fillna("M")

    merged["true_log10"] = safe_log10(merged["true_value"])
    merged["substrate_name"] = merged.get("substrate_name", "").fillna("")
    merged["smiles"] = merged["smiles"].fillna("")
    merged["sequence"] = merged["sequence"].fillna("")
    return merged


def load_partition_sheet(
    realkcat_root,
    master_df: pd.DataFrame,
    parameter: str,
    embedding: str,
    fold: int,
    sheet_name: str,
    split_name: str,
) -> pd.DataFrame:
    """Load and enrich one CatRange workbook sheet."""
    workbook = suite_build._partition_workbook(Path(realkcat_root).resolve(), parameter, fold, embedding)
    if not workbook.exists():
        raise FileNotFoundError(f"CatRange partition workbook not found: {workbook}")
    part_df = pd.read_excel(workbook, sheet_name=sheet_name)
    return _enrich_partition_df(part_df, master_df, parameter, fold, split_name)


def load_fold_splits(
    realkcat_root,
    parameter: str,
    embedding: str,
    fold: int,
    split_names: Iterable[str] = ("train", "val", "test"),
    master_df: Optional[pd.DataFrame] = None,
):
    """Load multiple split sheets for one fold into enriched dataframes."""
    realkcat_root = Path(realkcat_root).resolve()
    master_df = load_master_metadata(realkcat_root) if master_df is None else master_df
    workbook = suite_build._partition_workbook(realkcat_root, parameter, fold, embedding)
    if not workbook.exists():
        raise FileNotFoundError(f"CatRange partition workbook not found: {workbook}")

    sheet_map = {
        "train": "train_partition",
        "val": "val_partition",
        "test": "test_partition",
    }

    loaded = {}
    for split_name in split_names:
        if split_name not in sheet_map:
            raise ValueError(f"Unsupported split name: {split_name}")
        part_df = pd.read_excel(workbook, sheet_name=sheet_map[split_name])
        loaded[split_name] = _enrich_partition_df(part_df, master_df, parameter, fold, split_name)
    return loaded


def combine_splits(split_frames, split_order: Iterable[str]) -> pd.DataFrame:
    """Concatenate selected split dataframes in a stable order."""
    frames = [split_frames[name].copy() for name in split_order if name in split_frames]
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)
