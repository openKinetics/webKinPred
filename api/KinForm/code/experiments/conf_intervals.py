"""
TODO:
Visualize:
    - How big are confidence intervals?
    - How correlated are confidence intervals with error
    - How accurate are confidence intervals?
    ... rest of metrics
"""
from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Dict, List
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold
from tqdm import tqdm
from scipy.stats import norm as scipy_norm

# local imports ------------------------------------------------------ #
import sys
sys.path.append(str(Path(__file__).resolve().parent.parent))  # add parent dir to path
from smiles_embeddings.smiles_transformer.build_vocab import WordVocab  
from config import SEQ_LOOKUP, BS_PRED_PATH, CONFIG_L, ROOT
from utils.smiles_features import smiles_to_vec
from utils.sequence_features import sequences_to_feature_blocks
from model_training import train_model
from utils.pca import make_design_matrices
SEED = 42
np.random.seed(SEED)
# paths -------------------------------------------------------------- #
EITLEM_DIR  = ROOT / "data/EITLEM_data"
JSON_FILE   = EITLEM_DIR / "KCAT/kcat_data.json"
TRAIN_PAIRS = EITLEM_DIR / "KCAT/KCATTrainPairInfo"
TEST_PAIRS  = EITLEM_DIR / "KCAT/KCATTestPairInfo"
RESULTS_PKL = ROOT / "results/unikp_comp_eitlem.pkl"
OUT_DIR = ROOT / "results"

def get_intervals(y_matrix, method='normal', alpha=0.1, centre="mean"):
    """
    Get confidence intervals for the predictions.
    args:
        - y_matrix: matrix of predicted values from all trees (n_trees, n_samples)
        - method: ["quantile","normal"]
            - "quantile": empirical lower/upper quantiles across trees.
            - "normal": centre ± z_{1-alpha/2} * SD_across_trees.
        - alpha:  miscoverage level (e.g., 0.1 -> 90% intervals).
    returns:
        - pred: predicted values (size: n_samples)
        - lower: lower bounds of the confidence intervals (size: n_samples)
        - upper: upper bounds of the confidence intervals (size: n_samples)
    """
    if centre == "mean":
        pred = y_matrix.mean(axis=0)
    elif centre == "median":
        pred = np.median(y_matrix, axis=0)
    else:
        raise ValueError(f"Unknown centre: {centre}")

    if method == 'quantile':
        lower = np.quantile(y_matrix, q=alpha/2, axis=0)
        upper = np.quantile(y_matrix, q=1 - alpha/2, axis=0)
    elif method == 'normal':
        z = scipy_norm.ppf(1 - alpha/2)
        s = y_matrix.std(axis=0)
        lower = pred - z * s
        upper = pred + z * s
    else:
        raise ValueError(f"Unknown method: {method}")
    return pred, lower, upper


def eval_intervals(y_real, y_matrix, centre, method, alpha):
    """
    Evaluate prediction intervals constructed from per-tree predictions.

    Parameters
    ----------
    y_real : array-like, shape (n_samples,)
        Ground-truth targets.
    y_matrix : array-like, shape (n_trees, n_samples)
        Per-tree predictions.
    centre : {"mean","median"}
        Centre used inside get_intervals (align with your reporting choice).
    method : {"quantile","normal"}
        Interval construction method used in get_intervals.
    alpha : float
        Miscoverage level (e.g., 0.10 -> nominal 90% interval).

    Returns
    -------
    Returns
    -------
    metrics : dict
        {
          "nominal_coverage": float,
              # Target coverage = 1 - alpha (e.g., 0.90 for alpha=0.10).
          "acc": float,
              # Observed coverage (PICP): fraction of y within [lower, upper].
          "coverage_gap": float,
              # acc - nominal_coverage; >0 over-covers, <0 under-covers.
          "under_rate": float,
              # Fraction with y < lower (misses on the low side).
          "over_rate": float,
              # Fraction with y > upper (misses on the high side).
          "avg_width": float,
              # Mean interval width (upper - lower): sharpness (smaller is better
              # at fixed coverage).
          "median_width": float,
              # Median interval width (robust sharpness).
          "width_p10": float,
              # 10th percentile of widths (lower-tail sharpness).
          "width_p90": float,
              # 90th percentile of widths (upper-tail sharpness/outliers).
          "mae": float,
              # Mean absolute error of the centre prediction vs. y (point accuracy).
          "winkler_mean": float,
              # Mean Winkler score (interval score; lower is better).
          "sd_abs_err_corr": float,
              # Correlation between across-trees SD and |y - pred|; indicates if
              # uncertainty tracks errors (NaN if degenerate).
          "coverage_by_sd_decile": list[float],
              # Coverage within each decile of across-trees SD (heteroscedastic
              # calibration diagnostic).
        }

    """
    # ---- Coerce and validate ----
    y_real = np.asarray(y_real).ravel()
    y_matrix = np.asarray(y_matrix)
    if y_matrix.ndim != 2:
        raise ValueError("y_matrix must be 2D: (n_trees, n_samples).")
    _, n_samples = y_matrix.shape
    if y_real.shape[0] != n_samples:
        raise ValueError("y_real length must match n_samples in y_matrix.")
    if not (0 < alpha < 1):
        raise ValueError("alpha must be in (0,1).")

    # ---- Get intervals (and centre predictions) ----
    pred, lower, upper = get_intervals(
        y_matrix, method=method, alpha=alpha, centre=centre
    )

    # ---- Core masks and basic stats ----
    in_interval = (y_real >= lower) & (y_real <= upper)
    acc = float(np.mean(in_interval))
    nominal = 1.0 - alpha
    coverage_gap = acc - nominal

    under_rate = float(np.mean(y_real < lower))
    over_rate  = float(np.mean(y_real > upper))

    width = upper - lower
    avg_width = float(np.mean(width))
    median_width = float(np.median(width))
    width_p10 = float(np.percentile(width, 10))
    width_p90 = float(np.percentile(width, 90))

    # Point accuracy of the centre prediction
    abs_err = np.abs(y_real - pred)
    mae = float(np.mean(abs_err))

    sd = y_matrix.std(axis=0)
    # Correlation between SD and absolute error (guard against zero variance)
    if np.allclose(sd.std(), 0.0) or np.allclose(abs_err.std(), 0.0):
        sd_abs_err_corr = np.nan
    else:
        sd_abs_err_corr = float(np.corrcoef(sd, abs_err)[0, 1])
    # S = (u-l) + (2/alpha)*(l - y)*1{y<l} + (2/alpha)*(y - u)*1{y>u}
    penal_under = np.maximum(lower - y_real, 0.0)
    penal_over  = np.maximum(y_real - upper, 0.0)
    winkler = width + (2.0 / alpha) * (penal_under + penal_over)
    winkler_mean = float(np.mean(winkler))

    deciles = np.percentile(sd, np.linspace(0, 100, 11))
    # Ensure strictly increasing bin edges (handle constant-SD edge case)
    deciles[0] = -np.inf
    deciles[-1] = np.inf
    bins = np.digitize(sd, deciles[1:-1], right=True)  # 0..9
    coverage_by_sd_decile = []
    for b in range(10):
        mask = bins == b
        if np.any(mask):
            coverage_by_sd_decile.append(float(np.mean(in_interval[mask])))
        else:
            coverage_by_sd_decile.append(np.nan)

    return {
        "nominal_coverage": nominal,
        "acc": acc,
        "coverage_gap": coverage_gap,
        "under_rate": under_rate,
        "over_rate": over_rate,
        "avg_width": avg_width,
        "median_width": median_width,
        "width_p10": width_p10,
        "width_p90": width_p90,
        "mae": mae,
        "winkler_mean": winkler_mean,
        "sd_abs_err_corr": sd_abs_err_corr,
        "coverage_by_sd_decile": coverage_by_sd_decile,
    }

# ───────────────────────── data loading ──────────────────────────── #
with JSON_FILE.open() as fp:
    raw = json.load(fp)
valid = [(i, r) for i, r in enumerate(raw) if len(r["sequence"]) <= 1499 and float(r["value"]) > 0]
orig_idx = [i for i, _ in valid]
seqs = [r["sequence"] for _, r in valid]
raw_smi  = [r["smiles"]   for _, r in valid]
y_np = np.array([math.log(float(r["value"]), 10) for _, r in valid], np.float32)
# groups for GroupKFold
lookup = pd.read_pickle(SEQ_LOOKUP)
seq_to_id = {v: k for k, v in lookup.items()}
groups = [seq_to_id[s] for s in seqs]
# binding-site predictions
bs_df = pd.read_csv(BS_PRED_PATH, sep="\t")
smiles_vec = smiles_to_vec(raw_smi, method="smiles_transformer")

cfg = CONFIG_L
blocks_all, names = sequences_to_feature_blocks(
    sequence_list=seqs,
    binding_site_df=bs_df,
    cat_sites_df=None,
    ec_num_df= None,
    seq_to_id=seq_to_id,
    use_ec_logits=False,
    use_esmc=cfg["use_esmc"],
    use_esm2=cfg["use_esm2"],
    use_t5=cfg["use_t5"],
    t5_last_layer=cfg["t5_last_layer"],
    prot_rep_mode=cfg["prot_rep_mode"],
    task="kcat",
)

fold_idx = list(GroupKFold(5).split(seqs, groups=groups))
pbar = tqdm(fold_idx)
folds_out: List[Dict] = []
for fold_no, (tr, te) in enumerate(pbar, 1):
    tr = np.asarray(tr, int)
    te = np.asarray(te, int)

    # baseline
    X_tr, X_te,_ = make_design_matrices(tr, te, blocks_all, names, cfg, smiles_vec)
    y_tr, y_te = y_np[tr], y_np[te]
    model, y_pred, metrics, y_pred_matrix = train_model(X_tr, y_tr, X_te, y_te, fold=fold_no,return_one_pred=False)
    # y_pred_matrix is the predicted value of each tree on each test sample (y_pred_matrix[i] is the ith tree's prediction for all test samples)
    matrix_metrics = []
    for centre in ['mean','median']:
        for method in ['quantile','normal']:
            for alpha in [0.05,0.1,0.2]:
                mm = eval_intervals(y_te, y_pred_matrix, centre=centre, method=method, alpha=alpha)
                matrix_metrics.append(dict(
                    centre=centre,
                    method=method,
                    alpha=alpha,
                    **mm
                ))
    folds_out.append(
        dict(
            fold=fold_no,
            r2=metrics["r2"],
            rmse=metrics["rmse"],
            y_true=y_te.tolist(),
            y_pred=y_pred.tolist(),
            train_indices=tr.tolist(),
            test_indices=te.tolist(),
            y_pred_matrix = y_pred_matrix,
            matrix_metrics = matrix_metrics,
        )
    )

# save ----------------------------------------------------------------
RESULTS_PKL.parent.mkdir(parents=True, exist_ok=True)
pd.to_pickle(folds_out, RESULTS_PKL)
print("\n✓ Results saved to", RESULTS_PKL)