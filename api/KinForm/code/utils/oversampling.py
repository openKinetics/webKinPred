from __future__ import annotations
import subprocess
import tempfile
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
from config import MMSEQS_ENV, CONDA_EXE  # type: ignore
# oversampling thresholds ------------------------------------------- #
THR_BANDS = [0.90, 0.40, 0.20]             # high → low
# mapping threshold → human-readable band label
BAND_LABELS = {0.90: ">=0.90",
               0.40: "0.40–0.89",
               0.20: "0.20–0.39"}
BANDS_ORDER = [BAND_LABELS[t] for t in THR_BANDS] + ["<0.20"]   # keep order
HIGH_BAND   = BAND_LABELS[0.90]

SEED = 42
np.random.seed(SEED)
def cluster_sequences(seqs: List[str], min_id: float) -> Dict[int, int]:
    if not seqs:
        return {}
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        fa  = td / "seqs.fasta"
        pre = td / "clu"
        with fa.open("w") as fh:
            for i, s in enumerate(seqs):
                fh.write(f">{i}\n{s}\n")
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
                str(td),
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
        mapping: Dict[int, int] = {}
        with (pre.with_name(pre.name + "_cluster.tsv")).open() as fh:
            for line in fh:
                rep, mem = map(int, line.rstrip().split("\t"))
                mapping[mem] = rep
                mapping[rep] = rep
    return mapping


def _unique_seqs(seqs: List[str]) -> Tuple[List[str], Dict[int, int]]:
    uniq, m = [], {}
    for i, s in enumerate(seqs):
        if s not in uniq:
            uniq.append(s)
        m[i] = uniq.index(s)
    return uniq, m

def oversample_kcat_balanced_indices(
    tr_idx: np.ndarray,
    y_values: np.ndarray,
    bin_width: float = 1.0,
    target_multiplier: int = 2, # multiple of median bin size
) -> np.ndarray:
    """
    Oversample log₁₀(kcat) bins so that every non-empty 1-log unit bin
    has *exactly* 2 × median(bin-size) rows.

    • Bin edges: floor(min(y)) .. ceil(max(y)) in steps of `bin_width`
    • Only up-sampling (no down-sampling)
    • Uses the global SEED via `np.random.default_rng(SEED)` for
      reproducibility, just like the other oversampling helpers.
    """
    y = y_values[tr_idx]

    # ---------- fixed-width 1-log bins ----------
    bins = np.arange(np.floor(y.min()),
                     np.ceil(y.max()) + bin_width,
                     bin_width)
    bin_ids = np.digitize(y, bins, right=False)     # 1 … len(bins)-1

    # ---------- collect rows per bin ----------
    rows_by_bin: Dict[int, List[int]] = defaultdict(list)
    for row, b in zip(tr_idx, bin_ids):
        rows_by_bin[b].append(row)

    median_size = int(np.median([len(v) for v in rows_by_bin.values() if v]))
    target = target_multiplier* median_size

    rng = np.random.default_rng(SEED)
    new_idx: list[int] = []

    for b in range(1, len(bins)):                   # iterate over all bins
        idx_bin = rows_by_bin.get(b, [])
        if not idx_bin:
            continue
        new_idx.extend(idx_bin)                     # keep original rows
        deficit = target - len(idx_bin)
        if deficit > 0:                             # up-sample if needed
            extras = rng.choice(idx_bin, deficit, replace=True)
            new_idx.extend(extras.tolist())

    rng.shuffle(new_idx)
    return np.asarray(new_idx, int)

def oversample_similarity_balanced_indices(
    tr_idx: np.ndarray, sequences: List[str],
    target_multiplier: int = 0.15 # fraction of high-similarity band size
) -> np.ndarray:
    uniq_seqs, row2uniq = _unique_seqs([sequences[i] for i in tr_idx])
    # one clustering pass per threshold
    maps = {thr: cluster_sequences(uniq_seqs, thr) for thr in THR_BANDS}

    # ── band assignment for each unique sequence ────────────────────
    band_for: Dict[int, str] = {}
    for u in range(len(uniq_seqs)):
        band = "<0.20"
        for thr in THR_BANDS:
            cid = maps[thr][u]
            if sum(1 for v in maps[thr] if maps[thr][v] == cid) > 1:
                band = BAND_LABELS[thr]
                break
        band_for[u] = band

    # ── collect row indices per band ────────────────────────────────
    rows_by_band: Dict[str, List[int]] = defaultdict(list)
    for local_row, global_row in enumerate(tr_idx):
        rows_by_band[band_for[row2uniq[local_row]]].append(global_row)

    # target size = size of >=0.90 band (or max band if empty)
    target = int(target_multiplier * (len(rows_by_band[HIGH_BAND]) or max(len(v) for v in rows_by_band.values())))

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
    return np.asarray(new_idx, int)
