#!/usr/bin/env python
"""
Comprehensive Cross-Validation Training Script for CatRange.

Faithfully reproduces the publication training pipeline (section "e"):
1. Load all data (WT, MD, Neg pools, PafA, OOD)
2. Create proportional holdout set before CV
3. Relabel negatives via hybrid policy (using full neg pool with WT values)
4. Split negatives 95/2.5/2.5 (publication split)
5. 5-fold stratified CV (composite class x source stratification)
6. Per fold: Add WT counterparts + PafA to positives -> Combine with neg -> SMOTE
   -> Standardize -> Weight (balanced + 0.1 neg reduction) -> Train XGBoost -> HNM
7. Evaluate on: test fold, holdout, OOD, PafA, seq-ID thresholds
8. Fold 5 designated as final public model (ESM-C)

Publication settings:
  - SMOTE: True (default)
  - Negative weight reduction: 0.1
  - Fold split: 80% train / 10% val / 10% test
  - Negative split: 95% train / 2.5% val / 2.5% test
  - WT counterparts added to positive partition
  - PafA added to positive partition (train + val, not test)
  - XGBoost: Optuna-tuned params (n_estimators=1920, max_depth=11, etc.)

Usage:
    python -m scripts.cv_train --config configs/kcat_esmc.yaml
    python -m scripts.cv_train --config configs/km_esmc.yaml
    python -m scripts.cv_train --config configs/kcat_esm2.yaml
    python -m scripts.cv_train --config configs/km_esm2.yaml
"""

import argparse
import json
import warnings
from collections import Counter
from pathlib import Path
from datetime import datetime

def _load_runtime_dependencies():
    """Import heavy runtime dependencies only after CLI args are parsed."""
    global yaml, np, pd, torch, xgb, joblib
    global StratifiedKFold, train_test_split, compute_class_weight, LabelEncoder
    global set_seed, check_device, print_device_info, tensor_to_numpy, safe_load
    global TensorDatasetSimple, extract_tensors, load_all_data
    global create_holdout_set_proportional, relabel_negatives_hybrid
    global apply_smote, get_train_stats_separate, standardize_array
    global evaluate_fold, calculate_e_accuracy, aggregate_cv_results

    if globals().get("xgb") is not None:
        return

    import yaml as _yaml
    import numpy as _np
    import pandas as _pd
    import torch as _torch
    import xgboost as _xgb
    import joblib as _joblib
    from sklearn.model_selection import StratifiedKFold as _StratifiedKFold, train_test_split as _train_test_split
    from sklearn.utils.class_weight import compute_class_weight as _compute_class_weight
    from sklearn.preprocessing import LabelEncoder as _LabelEncoder

    from src.utils import (
        set_seed as _set_seed,
        check_device as _check_device,
        print_device_info as _print_device_info,
        tensor_to_numpy as _tensor_to_numpy,
        safe_load as _safe_load,
    )
    from src.data_pipeline import (
        TensorDatasetSimple as _TensorDatasetSimple,
        extract_tensors as _extract_tensors,
        load_all_data as _load_all_data,
        create_holdout_set_proportional as _create_holdout_set_proportional,
        relabel_negatives_hybrid as _relabel_negatives_hybrid,
        apply_smote as _apply_smote,
        get_train_stats_separate as _get_train_stats_separate,
        standardize_array as _standardize_array,
    )
    from src.evaluation import (
        evaluate_fold as _evaluate_fold,
        calculate_e_accuracy as _calculate_e_accuracy,
        aggregate_cv_results as _aggregate_cv_results,
    )

    yaml = _yaml
    np = _np
    pd = _pd
    torch = _torch
    xgb = _xgb
    joblib = _joblib
    StratifiedKFold = _StratifiedKFold
    train_test_split = _train_test_split
    compute_class_weight = _compute_class_weight
    LabelEncoder = _LabelEncoder
    set_seed = _set_seed
    check_device = _check_device
    print_device_info = _print_device_info
    tensor_to_numpy = _tensor_to_numpy
    safe_load = _safe_load
    TensorDatasetSimple = _TensorDatasetSimple
    extract_tensors = _extract_tensors
    load_all_data = _load_all_data
    create_holdout_set_proportional = _create_holdout_set_proportional
    relabel_negatives_hybrid = _relabel_negatives_hybrid
    apply_smote = _apply_smote
    get_train_stats_separate = _get_train_stats_separate
    standardize_array = _standardize_array
    evaluate_fold = _evaluate_fold
    calculate_e_accuracy = _calculate_e_accuracy
    aggregate_cv_results = _aggregate_cv_results

warnings.filterwarnings("ignore")


# ============================================================================
# HARD NEGATIVE MINING (matches original train_model() function)
# ============================================================================
def apply_hard_negative_mining(model, X_train, y_train, X_val, y_val,
                                sample_weights, val_weights, params,
                                boost_factor=100.0, finetune_fraction=0.2):
    """Fine-tune model on misclassified (hard negative) samples.

    Matches original: uses weight_map from sample_weights (which already
    includes the 0.1 neg reduction). Boosted weights = weight_map[label] * boost_factor.
    """
    train_preds = model.predict(X_train)
    hard_mask = (train_preds != y_train)
    X_hard = X_train[hard_mask]
    y_hard = y_train[hard_mask]

    if len(X_hard) == 0:
        print("  [HNM] No hard negatives found, skipping.")
        return model

    print(f"  [HNM] Found {len(X_hard)} hard negatives; boost_factor={boost_factor}")

    # Build weight_map from sample_weights (matches original behavior:
    # picks first weight encountered per class from shuffled data)
    weight_map = {}
    for label, weight in zip(y_train, sample_weights):
        if label not in weight_map:
            weight_map[label] = weight

    boosted_weights = np.array([weight_map.get(l, 1.0) * boost_factor for l in y_hard])

    X_comb = np.vstack([X_train, X_hard])
    y_comb = np.hstack([y_train, y_hard])
    w_comb = np.hstack([sample_weights, boosted_weights])

    booster = model.get_booster()
    if hasattr(model, "best_iteration") and model.best_iteration is not None:
        orig_rounds = model.best_iteration + 1
    else:
        orig_rounds = len(booster.get_dump())

    ft_rounds = max(10, int(finetune_fraction * orig_rounds))
    print(f"  [HNM] Original rounds: {orig_rounds}, fine-tuning: {ft_rounds}")

    raw_params = {k: v for k, v in params.items()
                  if k not in ["n_estimators", "early_stopping_rounds"]}

    dtrain = xgb.DMatrix(X_comb, label=y_comb, weight=w_comb)
    dval = xgb.DMatrix(X_val, label=y_val, weight=val_weights)

    booster_ft = xgb.train(
        raw_params, dtrain,
        num_boost_round=ft_rounds,
        early_stopping_rounds=10,
        evals=[(dval, "validation")],
        xgb_model=booster,
        verbose_eval=False,
    )

    # Re-wrap booster into XGBClassifier (matches original)
    model._Booster = booster_ft
    model._le = LabelEncoder().fit(y_comb)
    print("  [HNM] Fine-tuning complete.")
    return model


# ============================================================================
# HELPERS
# ============================================================================
def _build_config(raw):
    """Build flat config dict from YAML.  YAML is the single source of truth."""
    model = raw.get("model", {})
    data = raw.get("data", {})
    emb = raw.get("embedding", {})
    param = raw.get("parameter", {})
    hnm = model.get("hard_negative_mining", {})
    smote = data.get("smote", {})
    holdout = data.get("holdout", {})

    # ---- XGBoost parameters ------------------------------------------------
    xgb_keys = ["n_estimators", "max_depth", "learning_rate", "subsample",
                "colsample_bytree", "objective", "eval_metric",
                "n_jobs", "verbosity"]
    xgb_params = {k: model[k] for k in xgb_keys if k in model}
    # YAML uses reg_alpha / reg_lambda; XGBoost wants alpha / lambda
    if "reg_alpha" in model:
        xgb_params["alpha"] = model["reg_alpha"]
    if "reg_lambda" in model:
        xgb_params["lambda"] = model["reg_lambda"]
    xgb_params["random_state"] = raw.get("seed", 42)
    xgb_params["num_class"] = param.get("n_classes", 8)

    return {
        "seed":                  raw.get("seed", 42),
        "param_type":            str(param.get("name", "kcat")).lower(),
        "embedding_type":        str(emb.get("model", "esmc")).lower(),
        "split_dim":             emb.get("feature_split", 1152),
        "n_folds":               data.get("n_folds", 5),
        "n_classes":             param.get("n_classes", 8),
        "holdout_n_positive":    holdout.get("n_positive", 100),
        "holdout_n_negative":    holdout.get("n_negative", 10),
        "use_pub_neg_wt_idx":    holdout.get("use_publication_neg_wt_indices", False),
        "neg_wt_indices":        holdout.get("neg_wt_indices", None),
        "smote_enabled":         smote.get("enabled", True),
        "smote_k_neighbors":     smote.get("k_neighbors", 5),
        "class_imbalance_factor":model.get("class_imbalance_factor", 0.1),
        "early_stopping_rounds": model.get("early_stopping_rounds", 10),
        "deterministic_histogram":model.get("deterministic_histogram", False),
        "hnm_enabled":           hnm.get("enabled", True),
        "hnm_boost_factor":      hnm.get("boost_factor", 100.0),
        "hnm_finetune_fraction": hnm.get("finetune_fraction", 0.2),
        "xgb_params":            xgb_params,
    }


# ============================================================================
# MAIN TRAINING
# ============================================================================
def main():
    parser = argparse.ArgumentParser(description="CatRange CV Training (Publication Pipeline)")
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config")
    parser.add_argument("--device", type=str, choices=["auto", "cuda", "cpu"], default="auto")
    parser.add_argument("--output-dir", type=str, default="outputs")
    parser.add_argument("--final-fold", type=int, default=5, help="Fold to designate as final model")
    parser.add_argument("--no-smote", action="store_true", default=False,
                        help="Disable SMOTE oversampling (publication default: enabled)")
    parser.add_argument("--no-synthetic-negatives", action="store_true", default=False,
                        help="Ablation: exclude synthetic catalytic-site alanine negatives and WT counterparts")
    args = parser.parse_args()

    try:
        _load_runtime_dependencies()
    except Exception as exc:
        parser.exit(
            1,
            "CatRange training dependencies failed to import. "
            "If this environment was upgraded to NumPy 2.x, repair it with "
            "`python3 -m pip install \"numpy<2\"` and reinstall the package entry points "
            "with `python3 -m pip install --no-deps -e .`.\n"
            f"Original import error: {exc}\n",
        )

    # ========================================================================
    # SETUP
    # ========================================================================
    print("=" * 70)
    print("CatRange Cross-Validation Training (Publication Pipeline)")
    print("=" * 70)

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)
    cfg = _build_config(config)

    # Core settings from YAML (single source of truth)
    param_type      = cfg["param_type"]
    embedding_type  = cfg["embedding_type"]
    split_dim       = cfg["split_dim"]
    n_folds         = cfg["n_folds"]
    seed            = cfg["seed"]
    xgb_params      = cfg["xgb_params"]

    use_smote = cfg["smote_enabled"] and not args.no_smote
    use_synthetic_negatives = not args.no_synthetic_negatives

    device = check_device() if args.device == "auto" else torch.device(args.device)
    print_device_info(device)
    set_seed(seed)
    print(f"Random seed: {seed}")

    print(f"Parameter: {param_type}")
    print(f"Embedding: {embedding_type} (split_dim={split_dim})")
    print(f"CV folds: {n_folds}")
    print(f"Final model fold: {args.final_fold}")
    print(f"SMOTE: {'enabled (publication default)' if use_smote else 'DISABLED'}")
    print(f"Synthetic negatives: {'enabled (publication default)' if use_synthetic_negatives else 'DISABLED'}")
    print(f"Neg weight reduction: {cfg['class_imbalance_factor']}")

    # XGBoost params from YAML (Optuna-tuned)
    xgb_params["tree_method"] = "gpu_hist" if device.type == "cuda" else "hist"
    if cfg["deterministic_histogram"]:
        xgb_params["deterministic_histogram"] = True
    print(f"XGBoost: n_est={xgb_params.get('n_estimators')}, "
          f"depth={xgb_params['max_depth']}, lr={xgb_params['learning_rate']:.6f}")

    # Output directories
    output_dir = Path(args.output_dir) / f"{param_type}_{embedding_type}"
    model_dir = output_dir / "models"
    results_dir = output_dir / "results"
    plots_dir = output_dir / "plots"
    for d in [model_dir, results_dir, plots_dir]:
        d.mkdir(parents=True, exist_ok=True)

    # ========================================================================
    # LOAD DATA
    # ========================================================================
    print("\n" + "=" * 70)
    print("DATA LOADING")
    print("=" * 70)

    data = load_all_data(embedding_type=embedding_type, param_type=param_type, device=device)
    X_wt, y_wt = data["wt"]
    X_md, y_md = data["md"]
    data_dir = data["data_dir"]
    label_idx = data["label_idx"]

    # ========================================================================
    # HOLDOUT CREATION (before CV)
    # ========================================================================
    print("\n" + "=" * 70)
    print("HOLDOUT SET CREATION")
    print("=" * 70)

    set_seed(seed)
    neg_test_ds = TensorDatasetSimple(*data["neg_test"]) if use_synthetic_negatives else None

    holdout, (X_wt, y_wt), (X_md, y_md), neg_test_updated, hold_indices = \
        create_holdout_set_proportional(
            X_wt, y_wt, X_md, y_md,
            neg_test_ds=neg_test_ds,
            n_hold=cfg["holdout_n_positive"],
            n_neg_hold=cfg["holdout_n_negative"] if use_synthetic_negatives else 0,
            device=device, seed=seed,
        )

    if use_synthetic_negatives:
        # ---- WT-counterpart negative holdout (for catalytic-inference eval) ----
        # In the publication, 10 hardcoded indices into negative_pool_wt_seq
        # are used to create a WT-counterpart holdout for drop-prediction checks.
        X_wt_counterparts_all, y_wt_counterparts_all = data["neg_wtseq_wtvals"]
        use_pub_neg_wt = cfg["use_pub_neg_wt_idx"]
        neg_wt_idx_list = cfg["neg_wt_indices"]

        if use_pub_neg_wt and neg_wt_idx_list is not None:
            neg_wt_idx = torch.tensor(neg_wt_idx_list, dtype=torch.long)
            print(f"[Holdout] Using publication neg-WT indices: {neg_wt_idx_list}")
        else:
            # Fallback: random indices (same count as neg holdout)
            torch.manual_seed(seed + 1)  # offset seed to avoid collision with pos holdout
            neg_wt_idx = torch.randperm(len(X_wt_counterparts_all))[:cfg["holdout_n_negative"]]
            print(f"[Holdout] Using random neg-WT indices: {neg_wt_idx.tolist()}")

        neg_holdout_wt_X = X_wt_counterparts_all[neg_wt_idx].to(device)
        neg_holdout_wt_y = y_wt_counterparts_all[neg_wt_idx].to(device)
        holdout["X_neg_wt"] = neg_holdout_wt_X
        holdout["y_neg_wt"] = neg_holdout_wt_y
        hold_indices["neg_wt"] = neg_wt_idx.cpu().numpy()
        print(f"[Holdout] Neg-WT counterpart holdout: {len(neg_wt_idx)} samples")
    else:
        holdout["X_neg_wt"] = None
        holdout["y_neg_wt"] = None
        hold_indices["neg_wt"] = np.array([], dtype=int)
        print("[Holdout] Synthetic negative holdout disabled for ablation")

    # Build combined positive pool (WT + MD) after holdout removal
    X_pos = torch.cat([X_wt, X_md], dim=0)
    y_pos = torch.cat([y_wt, y_md], dim=0)
    source_labels = torch.cat([torch.zeros(len(X_wt)), torch.ones(len(X_md))]).long()

    print(f"Positive pool (after holdout): {X_pos.shape}")
    print(f"  WT: {len(X_wt)}, MD: {len(X_md)}")

    # ========================================================================
    # NEGATIVE RELABELLING (full pool, aligned files)
    # ========================================================================
    print("\n" + "=" * 70)
    print("NEGATIVE RELABELLING")
    print("=" * 70)

    if use_synthetic_negatives:
        # All three negative pool files are aligned (same 5278 samples, same order):
        #   neg_full_drops:       (X_mutant_features, drops_binary)
        #   neg_mdseq_wtvals:     (X_mutant_features, wt_kcat_bins) - same X as drops
        #   neg_wtseq_wtvals:     (X_wt_features, wt_kcat_bins)     - WT counterparts
        X_neg_full, y_neg_drops = data["neg_full_drops"]       # mutant features + drops
        _, y_neg_wt = data["neg_full_wtvals"]                   # WT kcat/km bins
        X_wt_counterparts, y_wt_counterparts = data["neg_wtseq_wtvals"]  # WT features + bins

        # Relabel full pool using hybrid policy
        y_neg_relabelled = relabel_negatives_hybrid(
            y_neg_wt, y_neg_drops, kparam=param_type, seed=seed)

        print(f"Full neg pool: {len(X_neg_full)} samples relabelled")
        unique, counts = np.unique(y_neg_relabelled.numpy(), return_counts=True)
        print(f"New label distribution: {dict(zip(unique.tolist(), counts.tolist()))}")

        # Split negatives 95% / 2.5% / 2.5% (publication split)
        n_neg = len(X_neg_full)
        n_neg_tr = int(0.95 * n_neg)
        n_neg_va = int(0.975 * n_neg) - n_neg_tr

        rng_neg = np.random.default_rng(seed)
        perm_neg = rng_neg.permutation(n_neg)
        idx_neg_tr = perm_neg[:n_neg_tr]
        idx_neg_va = perm_neg[n_neg_tr:n_neg_tr + n_neg_va]
        idx_neg_te = perm_neg[n_neg_tr + n_neg_va:]

        # Relabelled negatives (mutant features + hybrid labels)
        X_neg_tr = X_neg_full[idx_neg_tr];  y_neg_tr = y_neg_relabelled[idx_neg_tr].to(device)
        X_neg_va = X_neg_full[idx_neg_va];  y_neg_va = y_neg_relabelled[idx_neg_va].to(device)
        X_neg_te = X_neg_full[idx_neg_te];  y_neg_te = y_neg_relabelled[idx_neg_te].to(device)

        # WT counterparts (WT features + WT kcat/km bins) - added to positives
        X_wt_ctr_tr = X_wt_counterparts[idx_neg_tr];  y_wt_ctr_tr = y_wt_counterparts[idx_neg_tr].to(device)
        X_wt_ctr_va = X_wt_counterparts[idx_neg_va];  y_wt_ctr_va = y_wt_counterparts[idx_neg_va].to(device)

        print(f"Neg splits: train={len(X_neg_tr)}, val={len(X_neg_va)}, test={len(X_neg_te)}")
        print(f"WT counterparts: train={len(X_wt_ctr_tr)}, val={len(X_wt_ctr_va)}")
    else:
        X_neg_tr = X_pos[:0]; y_neg_tr = y_pos[:0]
        X_neg_va = X_pos[:0]; y_neg_va = y_pos[:0]
        X_neg_te = X_pos[:0]; y_neg_te = y_pos[:0]
        X_wt_ctr_tr = X_pos[:0]; y_wt_ctr_tr = y_pos[:0]
        X_wt_ctr_va = X_pos[:0]; y_wt_ctr_va = y_pos[:0]
        print("Synthetic negative relabelling skipped for ablation")

    # ========================================================================
    # PafA DATA
    # ========================================================================
    has_pafa = "pafa_train" in data
    if has_pafa:
        X_pafa_tr, y_pafa_tr = data["pafa_train"]
        X_pafa_va, y_pafa_va = data["pafa_val"]
        X_pafa_te, y_pafa_te = data["pafa_test"]
        print(f"\nPafA: train={len(X_pafa_tr)}, val={len(X_pafa_va)}, test={len(X_pafa_te)}")

    # ========================================================================
    # CROSS-VALIDATION
    # ========================================================================
    print("\n" + "=" * 70)
    print(f"STARTING {n_folds}-FOLD CROSS-VALIDATION")
    print("=" * 70)

    # Composite stratification labels: "{class}_{source}" for balanced WT/MD per fold
    strat_labels = np.array([f"{int(y)}_{int(s)}" for y, s
                             in zip(y_pos.cpu().numpy(), source_labels.cpu().numpy())])

    # Filter strata with <n_folds samples (StratifiedKFold requirement)
    strat_counts = Counter(strat_labels)
    valid_mask = np.array([strat_counts[sl] >= n_folds for sl in strat_labels])
    if not valid_mask.all():
        n_filtered = (~valid_mask).sum()
        print(f"[CV] Filtered {n_filtered} samples with <{n_folds}-member strata")

    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    fold_results = []
    fold_holdout_results = []
    fold_pafa_results = []
    fold_ood_results = []

    X_pos_valid = X_pos[valid_mask]
    y_pos_valid = y_pos[valid_mask]
    strat_valid = strat_labels[valid_mask]
    valid_indices = np.where(valid_mask)[0]

    for fold_idx, (train_idx_local, temp_idx_local) in enumerate(
            skf.split(tensor_to_numpy(X_pos_valid), strat_valid), start=1):

        print(f"\n{'=' * 60}")
        print(f"FOLD {fold_idx}/{n_folds}")
        print(f"{'=' * 60}")

        # Map back to original pool indices
        train_idx = valid_indices[train_idx_local]
        temp_idx = valid_indices[temp_idx_local]

        # Publication: 80% train, 10% val, 10% test (split the 20% fold 50/50)
        y_temp_np = tensor_to_numpy(y_pos[temp_idx])
        try:
            val_sub, test_sub = train_test_split(
                np.arange(len(y_temp_np)), test_size=0.5,
                random_state=seed, stratify=y_temp_np)
        except ValueError:
            val_sub, test_sub = train_test_split(
                np.arange(len(y_temp_np)), test_size=0.5,
                random_state=seed)

        val_idx = temp_idx[val_sub]
        test_idx = temp_idx[test_sub]

        X_train_pos = X_pos[train_idx]
        y_train_pos = y_pos[train_idx]
        X_val_pos = X_pos[val_idx]
        y_val_pos = y_pos[val_idx]
        X_test_pos = X_pos[test_idx]
        y_test_pos = y_pos[test_idx]

        # ---- Add WT counterparts to positives (publication pipeline) ----
        if use_synthetic_negatives:
            X_train_pos = torch.cat([X_train_pos, X_wt_ctr_tr], dim=0)
            y_train_pos = torch.cat([y_train_pos, y_wt_ctr_tr], dim=0)
            X_val_pos = torch.cat([X_val_pos, X_wt_ctr_va], dim=0)
            y_val_pos = torch.cat([y_val_pos, y_wt_ctr_va], dim=0)

        # ---- Add PafA to positives (train + val, NOT test) ----
        if has_pafa:
            X_train_pos = torch.cat([X_train_pos, X_pafa_tr], dim=0)
            y_train_pos = torch.cat([y_train_pos, y_pafa_tr], dim=0)
            X_val_pos = torch.cat([X_val_pos, X_pafa_va], dim=0)
            y_val_pos = torch.cat([y_val_pos, y_pafa_va], dim=0)

        pos_train_len = len(X_train_pos)
        neg_train_len = len(X_neg_tr)

        # ---- Combine positives + relabelled negatives ----
        X_train_combined = torch.cat([X_train_pos, X_neg_tr], dim=0)
        y_train_combined = torch.cat([y_train_pos, y_neg_tr], dim=0)

        X_val_combined = torch.cat([X_val_pos, X_neg_va], dim=0)
        y_val_combined = torch.cat([y_val_pos, y_neg_va], dim=0)

        X_test_combined = torch.cat([X_test_pos, X_neg_te], dim=0)
        y_test_combined = torch.cat([y_test_pos, y_neg_te], dim=0)

        print(f"  Train: {X_train_combined.shape} (pos={pos_train_len}, neg={neg_train_len})")
        print(f"  Val:   {X_val_combined.shape}")
        print(f"  Test:  {X_test_combined.shape}")

        # ----- SMOTE (publication default: ON) -----
        if use_smote:
            set_seed(seed)
            X_train_sm, y_train_sm = apply_smote(
                X_train_combined, y_train_combined, device=device,
                seed=seed, k_neighbors=cfg["smote_k_neighbors"],
            )
        else:
            X_train_sm = X_train_combined
            y_train_sm = y_train_combined
            print(f"  [SMOTE] Disabled. Train size: {len(X_train_sm)}")

        # ----- STANDARDIZATION (computed from (SMOTE-)augmented training data) -----
        m1, s1, m2, s2 = get_train_stats_separate(X_train_sm, split_dim=split_dim)

        X_train_std = standardize_array(tensor_to_numpy(X_train_sm), m1, s1, m2, s2, split_dim)
        y_train_np = tensor_to_numpy(y_train_sm)

        X_val_std = standardize_array(tensor_to_numpy(X_val_combined), m1, s1, m2, s2, split_dim)
        y_val_np = tensor_to_numpy(y_val_combined)

        X_test_std = standardize_array(tensor_to_numpy(X_test_combined), m1, s1, m2, s2, split_dim)
        y_test_np = tensor_to_numpy(y_test_combined)

        label_encoder = LabelEncoder()
        y_train_fit = label_encoder.fit_transform(y_train_np)
        y_val_fit = label_encoder.transform(y_val_np)

        # ----- CLASS WEIGHTING (balanced + 0.1 reduction for negatives) -----
        unique_cls = np.unique(y_train_fit)
        cls_w = compute_class_weight("balanced", classes=unique_cls, y=y_train_fit)
        w_map = {c: w for c, w in zip(unique_cls, cls_w)}
        sample_weights = np.array([w_map.get(int(l), 1.0) for l in y_train_fit])
        val_w = np.array([w_map.get(int(l), 1.0) for l in y_val_fit])

        # Reduce ALL original-negative sample weights (publication: 0.1)
        # Layout: [orig_pos (pos_train_len) | orig_neg (neg_train_len) | SMOTE_synthetic]
        neg_factor = cfg["class_imbalance_factor"]
        sample_weights[pos_train_len:pos_train_len + neg_train_len] *= neg_factor
        # Same for validation negatives (layout: [val_pos | val_neg])
        pos_val_len = len(X_val_pos)
        val_w[pos_val_len:pos_val_len + len(X_neg_va)] *= neg_factor
        print(f"  [Weighting] Balanced + {neg_factor}x for {neg_train_len} train + {len(X_neg_va)} val negatives")

        # Shuffle training data (matches original prepare_weighted_training_batch)
        rng = np.random.default_rng(seed)
        shuf_idx = rng.permutation(len(y_train_np))
        X_train_std = X_train_std[shuf_idx]
        y_train_np = y_train_np[shuf_idx]
        y_train_fit = y_train_fit[shuf_idx]
        sample_weights = sample_weights[shuf_idx]

        # ----- TRAIN XGBOOST -----
        n_classes = len(label_encoder.classes_)
        params = {**xgb_params, "num_class": n_classes,
                  "early_stopping_rounds": cfg["early_stopping_rounds"]}

        model = xgb.XGBClassifier(**params)
        model.fit(
            X_train_std, y_train_fit,
            sample_weight=sample_weights,
            eval_set=[(X_train_std, y_train_fit), (X_val_std, y_val_fit)],
            verbose=False,
        )

        best_iter = getattr(model, "best_iteration", None)
        print(f"  XGBoost trained: best_iteration={best_iter}")

        # ----- HARD NEGATIVE MINING -----
        model = apply_hard_negative_mining(
            model, X_train_std, y_train_fit, X_val_std, y_val_fit,
            sample_weights, val_w, params,
            boost_factor=cfg["hnm_boost_factor"],
            finetune_fraction=cfg["hnm_finetune_fraction"],
        )

        def _predict_original(X_std):
            pred_fit = model.predict(X_std).astype(int)
            return label_encoder.inverse_transform(pred_fit)

        # ----- SAVE MODEL + STATS -----
        model_path = model_dir / f"{param_type}_{embedding_type}_fold{fold_idx}.pkl"
        joblib.dump(model, model_path)

        stats_path = model_dir / f"{param_type}_{embedding_type}_fold{fold_idx}_stats.pt"
        torch.save({"mean_1": m1, "std_1": s1, "mean_2": m2, "std_2": s2,
                     "split_dim": split_dim,
                     "label_classes": label_encoder.classes_}, stats_path)

        # ----- EVALUATE ON TEST FOLD -----
        y_pred = _predict_original(X_test_std)
        try:
            y_proba = model.predict_proba(X_test_std) if np.array_equal(
                label_encoder.classes_, np.arange(n_classes)) else None
        except Exception:
            y_proba = None
        metrics = evaluate_fold(y_test_np, y_pred, y_proba=y_proba,
                                fold_idx=fold_idx, verbose=True)
        metrics["fold_idx"] = fold_idx
        fold_results.append(metrics)

        # ----- EVALUATE ON HOLDOUT -----
        X_hold_std = standardize_array(tensor_to_numpy(holdout["X"]), m1, s1, m2, s2, split_dim)
        y_hold_np = tensor_to_numpy(holdout["y"])
        y_hold_pred = _predict_original(X_hold_std)
        hold_metrics = evaluate_fold(y_hold_np, y_hold_pred, fold_idx=fold_idx, verbose=False)
        hold_metrics["fold_idx"] = fold_idx
        # fold_holdout_results.append(hold_metrics)
        # print(f"  Holdout Acc={hold_metrics['accuracy']:.4f}, e-Acc={hold_metrics['e_accuracy']:.4f}")

        # ----- CATALYTIC-INFERENCE EVAL (neg holdout: mutant vs WT-counterpart) -----
        if holdout.get("X_neg") is not None and holdout.get("X_neg_wt") is not None:
            neg_hold_std = standardize_array(tensor_to_numpy(holdout["X_neg"]),
                                             m1, s1, m2, s2, split_dim)
            neg_wt_hold_std = standardize_array(tensor_to_numpy(holdout["X_neg_wt"]),
                                                m1, s1, m2, s2, split_dim)
            neg_hold_preds = _predict_original(neg_hold_std)
            wt_hold_preds = _predict_original(neg_wt_hold_std)
            bin_drop = wt_hold_preds - neg_hold_preds
            # Ground truth: y_neg from neg_test holdout is drop-type (1=expected drop)
            neg_y_drop = tensor_to_numpy(holdout["y_neg"])
            expected_drop_mask = (neg_y_drop == 1)
            actual_drop_mask = (bin_drop >= 2)
            correct_drops = int(np.sum(expected_drop_mask & actual_drop_mask))
            total_expected = int(np.sum(expected_drop_mask))
            # print(f"  [CatInf] Holdout: {correct_drops}/{total_expected} expected drops predicted (bin_diff>=2)")

        # ----- EVALUATE ON PafA TEST -----
        if has_pafa:
            X_pafa_std = standardize_array(tensor_to_numpy(X_pafa_te), m1, s1, m2, s2, split_dim)
            y_pafa_np = tensor_to_numpy(y_pafa_te)
            y_pafa_pred = _predict_original(X_pafa_std)
            pafa_acc = float((y_pafa_pred == y_pafa_np).mean())
            pafa_eacc = calculate_e_accuracy(y_pafa_np, y_pafa_pred)
            # fold_pafa_results.append({"fold_idx": fold_idx, "accuracy": pafa_acc,
            #                            "e_accuracy": pafa_eacc})
            # print(f"  PafA Test Acc={pafa_acc:.4f}, e-Acc={pafa_eacc:.4f}")

        # ----- EVALUATE ON OOD -----
        if "ood_wt" in data and "ood_md" in data:
            X_ood_wt_std = standardize_array(tensor_to_numpy(data["ood_wt"][0]), m1, s1, m2, s2, split_dim)
            X_ood_md_std = standardize_array(tensor_to_numpy(data["ood_md"][0]), m1, s1, m2, s2, split_dim)
            y_ood_wt_np = tensor_to_numpy(data["ood_wt"][1])
            y_ood_md_np = tensor_to_numpy(data["ood_md"][1])
            pred_ood_wt = _predict_original(X_ood_wt_std)
            pred_ood_md = _predict_original(X_ood_md_std)
            ood_wt_acc = float((pred_ood_wt == y_ood_wt_np).mean())
            ood_md_acc = float((pred_ood_md == y_ood_md_np).mean())
            ood_joint = float(((pred_ood_wt == y_ood_wt_np) & (pred_ood_md == y_ood_md_np)).mean())
            # fold_ood_results.append({"fold_idx": fold_idx, "wt_acc": ood_wt_acc,
            #                           "md_acc": ood_md_acc, "joint_acc": ood_joint})
            # print(f"  OOD wt_acc={ood_wt_acc:.4f}, md_acc={ood_md_acc:.4f}, joint={ood_joint:.4f}")

        # ----- SEQ-ID THRESHOLD EVALUATION -----
        seqid_results = []
        for thr in [40, 60, 80, 90, 99, 100]:
            seqid_file = data_dir / f"fold_{fold_idx}_{param_type}" / f"dataset_y1_y2_seq_idleq_{thr}.pt"
            if seqid_file.exists():
                d_thr = safe_load(seqid_file, device)
                X_thr = tensor_to_numpy(d_thr[0].to(device))
                y_thr = tensor_to_numpy(d_thr[label_idx].to(device))
                X_thr_std = standardize_array(X_thr, m1, s1, m2, s2, split_dim)
                pred_thr = _predict_original(X_thr_std)
                acc_thr = float((pred_thr == y_thr).mean())
                eacc_thr = calculate_e_accuracy(y_thr, pred_thr)
                seqid_results.append({"threshold": thr, "n": len(y_thr),
                                      "accuracy": acc_thr, "e_accuracy": eacc_thr})
                if fold_idx == args.final_fold:
                    print(f"    SeqID<={thr}: n={len(y_thr)}, Acc={acc_thr:.4f}, e-Acc={eacc_thr:.4f}")

        if seqid_results:
            pd.DataFrame(seqid_results).to_csv(
                results_dir / f"seqid_eval_fold{fold_idx}_{param_type}_{embedding_type}.csv",
                index=False)

    # ========================================================================
    # AGGREGATE RESULTS
    # ========================================================================
    print("\n" + "=" * 70)
    print("CROSS-VALIDATION SUMMARY")
    print("=" * 70)

    fold_df = pd.DataFrame(fold_results)
    fold_df.to_csv(results_dir / f"{param_type}_{embedding_type}_fold_metrics.csv", index=False)

    summary = aggregate_cv_results(
        fold_results,
        output_file=results_dir / f"{param_type}_{embedding_type}_fold_metrics_all.csv")
    summary.to_csv(results_dir / f"{param_type}_{embedding_type}_crossval_summary.csv", index=False)
    print(summary.to_string(index=False))

    # Holdout summary
    hold_df = pd.DataFrame(fold_holdout_results)
    # hold_df.to_csv(results_dir / f"{param_type}_{embedding_type}_holdout_metrics.csv", index=False)
    # print(f"\nHoldout mean Acc={hold_df['accuracy'].mean():.4f} +/- {hold_df['accuracy'].std():.4f}")
    # print(f"Holdout mean e-Acc={hold_df['e_accuracy'].mean():.4f} +/- {hold_df['e_accuracy'].std():.4f}")

    # PafA summary
    if fold_pafa_results:
        pafa_df = pd.DataFrame(fold_pafa_results)
        # pafa_df.to_csv(results_dir / f"{param_type}_{embedding_type}_pafa_metrics.csv", index=False)
        # print(f"\nPafA mean Acc={pafa_df['accuracy'].mean():.4f}")

    # OOD summary
    if fold_ood_results:
        ood_df = pd.DataFrame(fold_ood_results)
        # ood_df.to_csv(results_dir / f"{param_type}_{embedding_type}_ood_metrics.csv", index=False)
        # print(f"OOD mean joint Acc={ood_df['joint_acc'].mean():.4f}")

    # Mark final model
    final_model = model_dir / f"{param_type}_{embedding_type}_fold{args.final_fold}.pkl"
    final_dest = model_dir / f"{param_type}_{embedding_type}_FINAL.pkl"
    if final_model.exists():
        import shutil
        shutil.copy2(final_model, final_dest)
        final_stats = model_dir / f"{param_type}_{embedding_type}_fold{args.final_fold}_stats.pt"
        final_stats_dest = model_dir / f"{param_type}_{embedding_type}_FINAL_stats.pt"
        if final_stats.exists():
            shutil.copy2(final_stats, final_stats_dest)
        print(f"\n*** FINAL MODEL: Fold {args.final_fold} -> {final_dest}")

    # Save holdout indices for reproducibility
    np.savez(results_dir / f"holdout_indices_{param_type}_{embedding_type}.npz",
             **hold_indices)

    # Save training manifest
    manifest = {
        "timestamp": datetime.now().isoformat(),
        "param_type": param_type,
        "embedding_type": embedding_type,
        "n_folds": n_folds,
        "final_fold": args.final_fold,
        "seed": seed,
        "use_smote": use_smote,
        "use_synthetic_negatives": use_synthetic_negatives,
        "use_publication_neg_wt_indices": cfg["use_pub_neg_wt_idx"],
        "neg_wt_indices": cfg["neg_wt_indices"],
        "neg_weight_factor": cfg["class_imbalance_factor"],
        "xgb_params": {k: str(v) for k, v in xgb_params.items()},
        "split_dim": split_dim,
        "mean_accuracy": float(fold_df["accuracy"].mean()),
        "std_accuracy": float(fold_df["accuracy"].std()),
        "mean_e_accuracy": float(fold_df["e_accuracy"].mean()),
        "mean_mcc": float(fold_df["mcc"].mean()),
    }
    with open(results_dir / f"training_manifest_{param_type}_{embedding_type}.json", "w") as f:
        json.dump(manifest, f, indent=2)

    print(f"\n{'=' * 70}")
    print(f"TRAINING COMPLETE: {param_type} / {embedding_type}")
    print(f"Models: {model_dir}")
    print(f"Results: {results_dir}")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()
