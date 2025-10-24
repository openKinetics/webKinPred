#!/usr/bin/env python3
"""
oversampling_stats.py

Generate before/after band-balance statistics for the KINEFORM/UniKP
band-balanced oversampling procedure on the EITLEM and DLKcat kcat datasets.

Outputs
-------
oversampling_stats.json   – hierarchical JSON (dataset → fold → stats)
oversampling_stats.pkl    – same structure, pickled
oversampling_stats.csv    – flat table: dataset,fold,band,before,after

Notes
-----
* Requires the same supporting files and Conda MMseqs-enabled environment
  as the original training/evaluation scripts.
* The oversampling procedure relies on `mmseqs easy-cluster`, invoked via
  `subprocess`.  Make sure `config.MMSEQS_ENV` and `config.CONDA_EXE`
  are valid on your system.
"""

from __future__ import annotations

import json
import math
import pickle
import subprocess
import tempfile
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold

# ──────────────────────────── local config ────────────────────────── #
from config import (
    CONDA_EXE,      # path to the conda executable
    MMSEQS_ENV,     # name of the conda env that has mmseqs installed
    RAW_DLKCAT,       # DLKcat raw-JSON Path (see original script)
    SEQ_LOOKUP,     # pickle mapping protein-ID → sequence
)

# EITLEM paths (match evaluation script)
from pathlib import Path
import json
import math
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold
from tqdm import tqdm

# Determine repository root relative to this file (utils/ is in code/, so go up two levels)
ROOT = Path(__file__).resolve().parent.parent.parent
EITLEM_DIR  = ROOT / "data/EITLEM_data"
EITLEM_JSON = EITLEM_DIR / "KCAT/kcat_data.json"

# output directory
OUT_DIR = ROOT / "results"
OUT_DIR.mkdir(parents=True, exist_ok=True)

JSON_OUT = OUT_DIR / "oversampling_stats.json"
PKL_OUT  = OUT_DIR / "oversampling_stats.pkl"
CSV_OUT  = OUT_DIR / "oversampling_stats.csv"

# reproducibility
SEED = 42
np.random.seed(SEED)

# ─────────────────── identity-band / oversampling logic ───────────── #
THR_BANDS   = [0.90, 0.40, 0.20]                       # high → low
BAND_LABELS = {0.90: ">=0.90", 0.40: "0.40–0.89", 0.20: "0.20–0.39"}
BANDS_ORDER = [BAND_LABELS[t] for t in THR_BANDS] + ["<0.20"]
HIGH_BAND   = BAND_LABELS[0.90]


def cluster_sequences(seqs: List[str], min_id: float) -> Dict[int, int]:
    """
    Run MMseqs2 easy-cluster on a list of sequences and return a mapping
    from sequence index → cluster representative index.
    """
    if not seqs:
        return {}
    with tempfile.TemporaryDirectory() as td:
        td_path = Path(td)
        fa  = td_path / "seqs.fasta"
        pre = td_path / "clu"

        # write FASTA
        with fa.open("w") as fh:
            for i, s in enumerate(seqs):
                fh.write(f">{i}\n{s}\n")

        # MMseqs call (quiet)
        subprocess.run(
            [
                str(CONDA_EXE),
                "run",
                "-n",
                MMSEQS_ENV,
                "mmseqs",
                "easy-cluster",
                str(fa),
                str(pre),
                str(td_path),
                "--min-seq-id",
                str(min_id),
                "--cov-mode",
                "0",
                "-c",
                "0.8",
            ],
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )

        # read clustering tsv
        mapping: Dict[int, int] = {}
        clu_tsv = pre.with_name(pre.name + "_cluster.tsv")
        with clu_tsv.open() as fh:
            for line in fh:
                rep, mem = map(int, line.rstrip().split("\t"))
                mapping[mem] = rep
                mapping[rep] = rep
    return mapping


def _unique_seqs(seqs: List[str]) -> Tuple[List[str], Dict[int, int]]:
    """
    Return the list of unique sequences in *seqs* (preserving first
    occurrence order) and a mapping row-idx → unique-idx.
    """
    uniq, row2uniq = [], {}
    for i, s in enumerate(seqs):
        if s not in uniq:
            uniq.append(s)
        row2uniq[i] = uniq.index(s)
    return uniq, row2uniq


def _assign_bands(indices: np.ndarray, sequences: List[str]) -> Dict[str, List[int]]:
    """
    Core of the band assignment from the oversampling procedure.
    Returns a dict: band_label → list of *global* row indices falling
    into that band (for the given indices subset).
    """
    uniq_seqs, row2uniq = _unique_seqs([sequences[i] for i in indices])

    # cluster once per threshold
    maps = {thr: cluster_sequences(uniq_seqs, thr) for thr in THR_BANDS}

    # figure out band for each unique sequence
    band_for: Dict[int, str] = {}
    for u in range(len(uniq_seqs)):
        band = "<0.20"
        for thr in THR_BANDS:
            cid = maps[thr][u]
            if sum(1 for v in maps[thr] if maps[thr][v] == cid) > 1:
                band = BAND_LABELS[thr]
                break
        band_for[u] = band

    rows_by_band: Dict[str, List[int]] = defaultdict(list)
    for local_row, global_row in enumerate(indices):
        rows_by_band[band_for[row2uniq[local_row]]].append(global_row)
    return rows_by_band


def oversample_similarity_balanced_indices(
    tr_idx: np.ndarray, sequences: List[str]
) -> np.ndarray:
    """
    Perform band-balanced oversampling (code identical to evaluation
    scripts) and return the re-sampled index array.
    """
    rows_by_band = _assign_bands(tr_idx, sequences)

    # target size = size of >=0.90 band (or the largest band if that one is empty)
    target = len(rows_by_band[HIGH_BAND]) or max(len(v) for v in rows_by_band.values())

    rng = np.random.default_rng(SEED)
    new_idx: List[int] = []
    for band in BANDS_ORDER:
        rows = rows_by_band.get(band, [])
        if not rows:
            continue
        if band in ["<0.20", "0.20–0.39"]:
            deficit = target - len(rows)
            if deficit > 0:
                rows = rows + rng.choice(rows, deficit, replace=True).tolist()
        new_idx.extend(rows)

    rng.shuffle(new_idx)
    return np.asarray(new_idx, dtype=int)


def band_counts(indices: np.ndarray, sequences: List[str]) -> Dict[str, int]:
    """Return a *dict(band → count)* for the given row indices."""
    rows_by_band = _assign_bands(indices, sequences)
    return {band: len(rows_by_band.get(band, [])) for band in BANDS_ORDER}


# ───────────────────────── dataset utilities ──────────────────────── #
def load_eitlem() -> Tuple[List[str], List[int]]:
    """
    Load the EITLEM kcat dataset (same filtering as evaluation script).

    Returns
    -------
    sequences : list[str]
    groups    : list[int]   – protein ID per sequence (for GroupKFold)
    """
    data = json.loads(EITLEM_JSON.read_text())
    valid = [
        d for d in data
        if len(d["sequence"]) <= 1_499 and float(d["value"]) > 0
    ]
    sequences = [d["sequence"] for d in valid]

    # protein-ID groups
    seq_to_id = {seq: pid for pid, seq in pd.read_pickle(SEQ_LOOKUP).items()}
    groups = [seq_to_id[s] for s in sequences]
    return sequences, groups


def load_dlkcat() -> Tuple[List[str], List[int]]:
    """
    Load the DLKcat dataset (same filtering as training script).

    Returns
    -------
    sequences : list[str]
    groups    : list[int]
    """
    raw = json.loads(RAW_DLKCAT.read_text())
    valid = [
        d
        for d in raw
        if len(d["Sequence"]) <= 1_499
        and float(d["Value"]) > 0
        and "." not in d["Smiles"]
    ]
    sequences = [d["Sequence"] for d in valid]

    seq_to_id = {seq: pid for pid, seq in pd.read_pickle(SEQ_LOOKUP).items()}
    groups = [seq_to_id[s] for s in sequences]
    return sequences, groups


# ────────────────────────────── main logic ────────────────────────── #
def process_dataset(
    name: str,
    sequences: List[str],
    groups: List[int],
) -> List[Dict]:
    """
    Run 5-fold GroupKFold on *sequences*, returning a list of records:

        {
          "fold": int,
          "before": {band: int, ...},
          "after":  {band: int, ...}
        }
    """
    gkf = GroupKFold(5).split(sequences, groups=groups)
    records: List[Dict] = []

    for fold_no, (tr_idx, _te_idx) in enumerate(gkf, 1):
        tr_idx = np.asarray(tr_idx, dtype=int)

        # band counts BEFORE oversampling
        before = band_counts(tr_idx, sequences)

        # perform oversampling & recompute counts
        tr_bal = oversample_similarity_balanced_indices(tr_idx, sequences)
        after = band_counts(tr_bal, sequences)

        records.append(
            {
                "fold": fold_no,
                "before": before,
                "after": after,
            }
        )
        # --- console feedback (optional) ----------------------------
        print(
            f"[{name}] fold {fold_no}: "
            + ", ".join(
                f"{b}: {before[b]}→{after[b]}" for b in BANDS_ORDER
            )
        )
    return records

def print_stats(stats: Dict[str, List[Dict]]):
    for dataset, folds in stats.items():
        print(f"\nDataset: {dataset}")
        for entry in folds:
            fold = entry["fold"]
            print(f"  Fold {fold}")
            for stage in ["before", "after"]:
                print(f"    {stage.capitalize()}:")
                for band in BANDS_ORDER:
                    count = entry[stage].get(band, 0)
                    print(f"      {band:<10} {count}")



def flatten_to_rows(
    dataset_name: str,
    fold_records: List[Dict],
) -> List[Tuple]:
    """
    Convert the nested stats for one dataset into tabular rows:

        (dataset, fold, band, before, after)
    """
    rows: List[Tuple] = []
    for rec in fold_records:
        fold = rec["fold"]
        for band in BANDS_ORDER:
            rows.append(
                (
                    dataset_name,
                    fold,
                    band,
                    rec["before"].get(band, 0),
                    rec["after"].get(band, 0),
                )
            )
    return rows


def main() -> None:
    # === 1. load datasets ===========================================
    datasets = {
        "EITLEM": load_eitlem(),
        "DLKcat": load_dlkcat(),
    }

    stats_all: Dict[str, List[Dict]] = {}
    flat_rows: List[Tuple] = []

    # === 2. process each dataset ====================================
    for name, (seqs, groups) in datasets.items():
        print(f"\n══ Processing {name} ({len(seqs)} sequences) ══")
        fold_stats = process_dataset(name, seqs, groups)
        stats_all[name] = fold_stats
        flat_rows.extend(flatten_to_rows(name, fold_stats))

    # === 3. write outputs ===========================================
    # 3a. JSON
    with JSON_OUT.open("w") as fh:
        json.dump(stats_all, fh, indent=2)
    # 3b. pickle
    with PKL_OUT.open("wb") as fh:
        pickle.dump(stats_all, fh)

    # 3c. CSV
    df = pd.DataFrame(
        flat_rows, columns=["dataset", "fold", "band", "before", "after"]
    )
    df.to_csv(CSV_OUT, index=False)

    print("\n✓ Statistics written to:")
    print("  •", JSON_OUT)
    print("  •", PKL_OUT)
    print("  •", CSV_OUT)
    print_stats(stats_all)


if __name__ == "__main__":
    main()
