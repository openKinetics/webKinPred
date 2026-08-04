"""Shared helpers for the standalone benchmark workflow."""

import hashlib
import json
from pathlib import Path
from typing import Iterable, Optional

import numpy as np
import pandas as pd

from benchmark_constants import PARAMETER_BIN_EDGES

VALID_PROTEIN_AAS = frozenset("ACDEFGHIKLMNPQRSTVWY")


def ensure_columns(df: pd.DataFrame, required: Iterable[str], table_name: str) -> None:
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(f"{table_name} is missing required columns: {', '.join(missing)}")


def safe_log10(values: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(values, errors="coerce")
    return pd.Series(
        np.where(numeric > 0, np.log10(numeric), np.nan),
        index=values.index,
        dtype=float,
    )


def to_linear(values: pd.Series, value_scale: str) -> pd.Series:
    numeric = pd.to_numeric(values, errors="coerce")
    value_scale = value_scale.lower()

    if value_scale == "linear":
        return numeric.astype(float)
    if value_scale == "log10":
        return pd.Series(np.power(10.0, numeric), index=values.index, dtype=float)
    if value_scale == "log2":
        return pd.Series(np.power(2.0, numeric), index=values.index, dtype=float)

    raise ValueError(f"Unsupported value scale: {value_scale}")


def stable_pair_ids(
    sequence: pd.Series,
    smiles: pd.Series,
    substrate_name: Optional[pd.Series] = None,
) -> pd.Series:
    substrate_name = substrate_name if substrate_name is not None else pd.Series("", index=sequence.index)

    base_ids = []
    for seq, smi, sub in zip(sequence.fillna(""), smiles.fillna(""), substrate_name.fillna("")):
        payload = "||".join([str(seq).strip(), str(smi).strip(), str(sub).strip()])
        base_ids.append(hashlib.sha1(payload.encode("utf-8")).hexdigest()[:16])

    seen = {}
    unique_ids = []
    for base_id in base_ids:
        seen[base_id] = seen.get(base_id, 0) + 1
        count = seen[base_id]
        unique_ids.append(base_id if count == 1 else f"{base_id}__{count}")

    return pd.Series(unique_ids, index=sequence.index, dtype=str)


def stable_sequence_ids(sequence: pd.Series, prefix: str = "seq") -> pd.Series:
    ids = []
    for seq in sequence.fillna(""):
        payload = str(seq).strip()
        token = hashlib.sha1(payload.encode("utf-8")).hexdigest()[:16]
        ids.append(f"{prefix}_{token}")
    return pd.Series(ids, index=sequence.index, dtype=str)


def is_canonical_protein_sequence(sequence) -> bool:
    return isinstance(sequence, str) and bool(sequence) and set(sequence).issubset(VALID_PROTEIN_AAS)


def canonical_protein_mask(values: pd.Series) -> pd.Series:
    return pd.Series(
        [is_canonical_protein_sequence(value) for value in values.tolist()],
        index=values.index,
        dtype=bool,
    )


def resolve_pair_ids(
    df: pd.DataFrame,
    sequence_col: str,
    smiles_col: str,
    substrate_name_col: Optional[str] = None,
    pair_id_col: Optional[str] = None,
) -> pd.Series:
    if pair_id_col and pair_id_col in df.columns:
        return df[pair_id_col].astype(str)

    substrate_series = df[substrate_name_col] if substrate_name_col and substrate_name_col in df.columns else None
    return stable_pair_ids(df[sequence_col], df[smiles_col], substrate_series)


def bin_continuous_values(values: pd.Series, parameter: str) -> pd.Series:
    if parameter not in PARAMETER_BIN_EDGES:
        raise ValueError(f"Unsupported parameter for binning: {parameter}")

    edges = np.asarray(PARAMETER_BIN_EDGES[parameter], dtype=float)
    numeric = pd.to_numeric(values, errors="coerce").to_numpy()

    # CatRange's saved labels place exact edge values in the lower bin.
    bins = np.searchsorted(edges, numeric, side="left") - 1
    bins = np.clip(bins, 0, len(edges) - 2)
    bins = bins.astype(float)
    bins[np.isnan(numeric)] = np.nan

    return pd.Series(bins, index=values.index, dtype="Float64")


def slugify(name: str) -> str:
    return "".join(ch.lower() if ch.isalnum() else "_" for ch in name).strip("_")


def read_table(path, sep=","):
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")
    return pd.read_csv(path, sep=sep)


def make_output_dir(path):
    outdir = Path(path)
    outdir.mkdir(parents=True, exist_ok=True)
    return outdir


def write_json(path, payload) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
