from __future__ import annotations
from typing import Iterable, Optional, Sequence, Tuple, Dict, List
import numpy as np
import subprocess
import tempfile
from pathlib import Path
from config import MMSEQS_ENV, CONDA_EXE  # type: ignore
import scipy.sparse as sp  # type: ignore
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
    uniq, m, pos = [], {}, {}
    for i, s in enumerate(seqs):
        j = pos.get(s)
        if j is None:
            j = len(uniq)
            uniq.append(s)
            pos[s] = j
        m[i] = j
    return uniq, m

def _band_labels_for_indices(tr_idx: np.ndarray, sequences: Sequence[str]) -> dict[int, str]:
    """
    Assign similarity bands to each global index in `tr_idx` using your existing
    mmseqs-based clustering at thresholds THR_BANDS (0.90, 0.40, 0.20).

    Returns a mapping {global_index -> band_label}.
    """
    uniq_seqs, row2uniq = _unique_seqs([sequences[i] for i in tr_idx])
    maps = {thr: cluster_sequences(uniq_seqs, thr) for thr in THR_BANDS}

    band_for_local: dict[int, str] = {}
    for u in range(len(uniq_seqs)):
        band = "<0.20"
        for thr in THR_BANDS:
            cid = maps[thr][u]
            if sum(1 for v in maps[thr] if maps[thr][v] == cid) > 1:
                band = BAND_LABELS[thr]
                break
        band_for_local[u] = band

    band_for_global: dict[int, str] = {}
    for local_row, global_row in enumerate(tr_idx):
        band_for_global[int(global_row)] = band_for_local[row2uniq[local_row]]
    return band_for_global


def _group_by_value(values: Sequence[str], pool: np.ndarray) -> dict[str, list[int]]:
    """Group global indices in `pool` by exact string equality of `values`."""
    out: dict[str, list[int]] = {}
    for gid in map(int, pool):
        out.setdefault(values[gid], []).append(gid)
    return out


def _build_candidates_for_perspective(
    tr_idx: np.ndarray,
    sequences: Optional[Sequence[str]],
    smiles: Optional[Sequence[str]],
    perspective: str,
) -> dict[int, np.ndarray]:
    """
    For each anchor in `tr_idx`, return an array of candidate partner indices
    according to the requested perspective.
    """
    perspective = perspective.lower()
    tr_idx = np.asarray(tr_idx, int)

    if perspective == "none":
        return {int(i): np.array([], int) for i in tr_idx}

    if perspective == "all_pair":
        pool = tr_idx.copy()
        return {int(i): pool for i in tr_idx}

    if sequences is None or smiles is None:
        raise ValueError("`sequences` and `smiles` are required for GBA perspectives other than 'none'/'all_pair'.")

    by_seq = _group_by_value(sequences, tr_idx)
    by_smi = _group_by_value(smiles, tr_idx)

    cand: dict[int, np.ndarray] = {}
    tr_set = set(map(int, tr_idx))

    for i in map(int, tr_idx):
        same_seq = set(by_seq.get(sequences[i], []))
        same_smi = set(by_smi.get(smiles[i], []))

        if perspective == "protein":
            arr = np.array(sorted(same_seq), int)
        elif perspective == "smiles":
            arr = np.array(sorted(same_smi), int)
        elif perspective == "protein_or_smiles":
            arr = np.array(sorted(same_seq | same_smi), int)
        elif perspective == "complement":
            arr = np.array([j for j in tr_set if j not in (same_seq | same_smi)], int)
        else:
            raise ValueError(f"Unknown perspective: {perspective}")

        cand[i] = arr
    return cand

def mixup_augment(
    X_tr: np.ndarray,
    y_tr: np.ndarray,
    tr_idx: np.ndarray,
    *,
    sequences: Optional[Sequence[str]] = None,
    smiles: Optional[Sequence[str]] = None,
    perspective: str = "protein_or_smiles",
    n_synthetic: Optional[int] = None,
    proportion: Optional[float] = None,
    alpha: float = 0.4,                 # λ ~ Beta(alpha, alpha)
    weight_mode: str = "gaussian",      # 'gaussian' | 'uniform'
    sigma: Optional[float] = None,      # if None, uses 0.5 * std(y_tr)
    avoid_self: bool = True,
    anchor_pool: str = "all",           # 'all' | 'low_identity'
    include_bands: Iterable[str] = ("<0.20",),  # used if anchor_pool='low_identity'
    random_state: int = SEED,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Simple, GBA-style MixUp for regression (log10(kcat)).

    Parameters
    ----------
    X_tr, y_tr : training features/targets for the current fold (rows align to `tr_idx`).
                 Works with dense numpy arrays and CSR/CSC sparse matrices.
    tr_idx     : global dataset indices (one per row in X_tr/y_tr).
    sequences, smiles : required for GBA perspectives except 'none'/'all_pair'.
                        Exact-string equality defines neighbourhoods.
    perspective : one of {'none','all_pair','protein','smiles','protein_or_smiles','complement'}.
    n_synthetic : absolute number of synthetic rows to add. If None, uses `proportion`
                  (defaults to +100% if both are None).
    proportion  : fraction of len(tr_idx) to add (e.g., 0.5 -> +50%).
    alpha       : Beta(alpha, alpha) parameter for λ.
    weight_mode : 'gaussian' uses exp(-Δy² / (2σ²)) to prefer label-similar pairs; 'uniform' ignores labels.
    sigma       : σ for 'gaussian' weighting; default is 0.5 * std(y_tr).
    avoid_self  : if True, do not pair an item with itself when possible.
    anchor_pool : 'all' to sample anchors uniformly; 'low_identity' to restrict anchors
                  to low-identity proteins (using the '<0.20' band unless `include_bands` is set).
    include_bands : which bands count as "low identity" (defaults to '<0.20').
    random_state : RNG seed.

    Returns
    -------
    (X_aug, y_aug) : augmented matrices with synthetic rows appended.

    Notes
    -----
    Synthetic sample is:
        x_mix = λ x_i + (1-λ) x_j
        y_mix = λ y_i + (1-λ) y_j
    where j is sampled from the neighbourhood defined by `perspective`,
    with optional affinity-weighted probabilities. This follows GBA‑Mixup. 
    """
    if perspective.lower() == "none":
        return X_tr[:0], y_tr[:0]

    # how many synthetic rows
    if n_synthetic is None:
        n_synth = int(len(tr_idx)) if proportion is None else int(round(proportion * len(tr_idx)))
    else:
        n_synth = int(n_synthetic)
    if n_synth <= 0:
        return X_tr, y_tr

    rng = np.random.default_rng(random_state)

    # map global id -> row position
    pos_for = {int(g): i for i, g in enumerate(tr_idx)}
    anchors = np.asarray(tr_idx, int)

    # optionally restrict anchors to low-identity proteins
    if anchor_pool == "low_identity":
        if sequences is None:
            raise ValueError("`sequences` required when anchor_pool='low_identity'.")
        band_map = _band_labels_for_indices(anchors, sequences)
        keep = [i for i in anchors if band_map.get(int(i)) in set(include_bands)]
        anchors = np.asarray(keep, int)
        if anchors.size == 0:
            # fall back to all if nothing qualifies
            anchors = np.asarray(tr_idx, int)

    # neighbourhood candidates
    cand_map = _build_candidates_for_perspective(tr_idx, sequences, smiles, perspective)

    def _sample_partner(xi: int) -> int:
        cands = cand_map.get(int(xi), np.array([], int))
        if cands.size == 0:
            cands = np.asarray(tr_idx, int)
        if avoid_self and cands.size > 1:
            cands = cands[cands != int(xi)]
        if cands.size == 0:
            return int(xi)  # degenerate
        if weight_mode == "uniform":
            return int(rng.choice(cands))
        # gaussian weighting on |Δy|
        yi = float(y_tr[pos_for[int(xi)]])
        y_c = np.array([y_tr[pos_for[int(j)]] for j in cands], float)
        dy = np.abs(y_c - yi)
        s = sigma if (sigma and sigma > 0.0) else max(1e-8, 0.5 * float(np.std(y_tr) or 1.0))
        w = np.exp(- (dy * dy) / (2.0 * s * s))
        w_sum = float(w.sum())
        p = (w / w_sum) if w_sum > 0 else None
        return int(rng.choice(cands, p=p))

    new_rows = []
    new_targets: list[float] = []

    for _ in range(n_synth):
        xi = int(rng.choice(anchors))
        yi = _sample_partner(xi)
        lam = float(rng.beta(alpha, alpha))
        px, py = pos_for[xi], pos_for[yi]

        if sp is not None and sp.issparse(X_tr):
            row = X_tr.getrow(px).multiply(lam) + X_tr.getrow(py).multiply(1.0 - lam)
            new_rows.append(row)
        else:
            # assume dense
            if not isinstance(X_tr, np.ndarray):
                X_np = np.asarray(X_tr)
            else:
                X_np = X_tr
            xmix = lam * X_np[px] + (1.0 - lam) * X_np[py]
            new_rows.append(xmix)

        ymix = lam * float(y_tr[px]) + (1.0 - lam) * float(y_tr[py])
        new_targets.append(ymix)

    if sp is not None and sp.issparse(X_tr):
        X_syn = sp.vstack(new_rows, format=X_tr.getformat()) if new_rows else X_tr[:0]
    else:
        X_np = np.asarray(X_tr)
        X_syn = np.vstack(new_rows) if new_rows else X_np[:0]

    y_syn = np.asarray(new_targets, dtype=y_tr.dtype)
    return X_syn, y_syn


