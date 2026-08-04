#!/usr/bin/env python3
"""Train CatRange feature ablations without modifying the CatRange repository.

Feature modes:
- full: sequence ESM-C block + substrate/auxiliary block
- sequence: first `feature_split` columns only
- substrate: columns after `feature_split` only
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(description="Run CatRange sequence/substrate ablation CV.")
    parser.add_argument("--catrange-root", default="../CatRange", help="Path to CatRange root.")
    parser.add_argument("--config", default="../CatRange/configs/kcat_esmc.yaml", help="CatRange YAML config.")
    parser.add_argument("--output-dir", default="runs/kcat_esmc_ablation", help="Output directory.")
    parser.add_argument(
        "--feature-mode",
        action="append",
        choices=["full", "sequence", "substrate"],
        help="Feature mode to run. Repeatable. Default: all three.",
    )
    parser.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto")
    parser.add_argument("--max-folds", type=int, default=None, help="Optional quick-test fold limit.")
    parser.add_argument("--no-smote", action="store_true", help="Disable SMOTE.")
    parser.add_argument("--skip-hnm", action="store_true", help="Disable hard-negative mining.")
    return parser.parse_args()


def _feature_block(X, split_dim: int, mode: str):
    if mode == "full":
        return X
    if mode == "sequence":
        return X[:, :split_dim]
    if mode == "substrate":
        return X[:, split_dim:]
    raise ValueError(f"Unknown feature mode: {mode}")


def _standardize_selected(np, X_train, X_val, X_test, split_dim: int, mode: str):
    # CatRange standardizes full features with one scalar per block. For the
    # single-block ablations, the selected block gets the same scalar treatment.
    if mode == "full":
        mean_1 = float(X_train[:, :split_dim].mean())
        std_1 = max(float(X_train[:, :split_dim].std()), 1e-7)
        mean_2 = float(X_train[:, split_dim:].mean())
        std_2 = max(float(X_train[:, split_dim:].std()), 1e-7)

        def transform(X):
            return np.concatenate(
                [
                    (X[:, :split_dim] - mean_1) / std_1,
                    (X[:, split_dim:] - mean_2) / std_2,
                ],
                axis=1,
            )

        return transform(X_train), transform(X_val), transform(X_test), (mean_1, mean_2), (std_1, std_2)

    mean = float(X_train.mean())
    std = max(float(X_train.std()), 1e-7)
    return (X_train - mean) / std, (X_val - mean) / std, (X_test - mean) / std, mean, std


def _write_summary(pd, fold_df, output_dir: Path):
    metrics = [
        "accuracy",
        "e_accuracy",
        "precision",
        "recall",
        "f1",
        "mcc",
        "auc_pr",
        "n_samples",
    ]
    rows = []
    for metric in metrics:
        rows.append(
            {
                "metric": metric,
                "mean": fold_df[metric].mean(),
                "std": fold_df[metric].std(),
                "min": fold_df[metric].min(),
                "max": fold_df[metric].max(),
                "n_folds": fold_df["fold"].nunique(),
            }
        )
    summary_df = pd.DataFrame(rows)
    summary_df.to_csv(output_dir / "summary_metrics.csv", index=False)
    return summary_df


def run_one_mode(args, mode: str):
    catrange_root = Path(args.catrange_root).resolve()
    config_path = Path(args.config).resolve()
    output_dir = Path(args.output_dir).resolve() / mode
    output_dir.mkdir(parents=True, exist_ok=True)

    sys.path.insert(0, str(catrange_root))
    from scripts import cv_train as cvt

    cvt._load_runtime_dependencies()

    with config_path.open("r", encoding="utf-8") as handle:
        raw_config = cvt.yaml.safe_load(handle)
    cfg = cvt._build_config(raw_config)

    split_dim = int(cfg["split_dim"])
    seed = int(cfg["seed"])
    use_smote = bool(cfg["smote_enabled"]) and not args.no_smote
    use_hnm = bool(cfg["hnm_enabled"]) and not args.skip_hnm

    device = cvt.check_device() if args.device == "auto" else cvt.torch.device(args.device)
    cvt.set_seed(seed)

    xgb_params = dict(cfg["xgb_params"])
    xgb_params["tree_method"] = "gpu_hist" if device.type == "cuda" else "hist"
    if cfg["deterministic_histogram"]:
        xgb_params["deterministic_histogram"] = True

    # CatRange data loaders use relative data paths, so run from CatRange root.
    previous_cwd = Path.cwd()
    os.chdir(catrange_root)
    try:
        data = cvt.load_all_data(
            embedding_type=cfg["embedding_type"],
            param_type=cfg["param_type"],
            device=device,
        )
    finally:
        os.chdir(previous_cwd)

    X_wt, y_wt = data["wt"]
    X_md, y_md = data["md"]

    cvt.set_seed(seed)
    neg_test_ds = cvt.TensorDatasetSimple(*data["neg_test"])
    holdout, (X_wt, y_wt), (X_md, y_md), _, _ = cvt.create_holdout_set_proportional(
        X_wt,
        y_wt,
        X_md,
        y_md,
        neg_test_ds=neg_test_ds,
        n_hold=cfg["holdout_n_positive"],
        n_neg_hold=cfg["holdout_n_negative"],
        device=device,
        seed=seed,
    )

    X_pos = cvt.torch.cat([X_wt, X_md], dim=0)
    y_pos = cvt.torch.cat([y_wt, y_md], dim=0)
    source_labels = cvt.torch.cat([cvt.torch.zeros(len(X_wt)), cvt.torch.ones(len(X_md))]).long()

    X_neg_full, y_neg_drops = data["neg_full_drops"]
    _, y_neg_wt = data["neg_full_wtvals"]
    X_wt_counterparts, y_wt_counterparts = data["neg_wtseq_wtvals"]
    y_neg_relabelled = cvt.relabel_negatives_hybrid(
        y_neg_wt,
        y_neg_drops,
        kparam=cfg["param_type"],
        seed=seed,
    )

    n_neg = len(X_neg_full)
    n_neg_tr = int(0.95 * n_neg)
    n_neg_va = int(0.975 * n_neg) - n_neg_tr
    rng_neg = cvt.np.random.default_rng(seed)
    perm_neg = rng_neg.permutation(n_neg)
    idx_neg_tr = perm_neg[:n_neg_tr]
    idx_neg_va = perm_neg[n_neg_tr : n_neg_tr + n_neg_va]
    idx_neg_te = perm_neg[n_neg_tr + n_neg_va :]

    X_neg_tr = X_neg_full[idx_neg_tr]
    y_neg_tr = y_neg_relabelled[idx_neg_tr].to(device)
    X_neg_va = X_neg_full[idx_neg_va]
    y_neg_va = y_neg_relabelled[idx_neg_va].to(device)
    X_neg_te = X_neg_full[idx_neg_te]
    y_neg_te = y_neg_relabelled[idx_neg_te].to(device)

    X_wt_ctr_tr = X_wt_counterparts[idx_neg_tr]
    y_wt_ctr_tr = y_wt_counterparts[idx_neg_tr].to(device)
    X_wt_ctr_va = X_wt_counterparts[idx_neg_va]
    y_wt_ctr_va = y_wt_counterparts[idx_neg_va].to(device)

    has_pafa = "pafa_train" in data
    if has_pafa:
        X_pafa_tr, y_pafa_tr = data["pafa_train"]
        X_pafa_va, y_pafa_va = data["pafa_val"]

    strat_labels = cvt.np.array(
        [f"{int(y)}_{int(s)}" for y, s in zip(y_pos.cpu().numpy(), source_labels.cpu().numpy())]
    )
    strat_counts = Counter(strat_labels)
    valid_mask = cvt.np.array([strat_counts[sl] >= cfg["n_folds"] for sl in strat_labels])
    X_pos_valid = X_pos[valid_mask]
    strat_valid = strat_labels[valid_mask]
    valid_indices = cvt.np.where(valid_mask)[0]

    skf = cvt.StratifiedKFold(n_splits=cfg["n_folds"], shuffle=True, random_state=seed)
    fold_rows = []

    for fold, (train_idx_local, temp_idx_local) in enumerate(
        skf.split(cvt.tensor_to_numpy(X_pos_valid), strat_valid),
        start=1,
    ):
        if args.max_folds is not None and fold > args.max_folds:
            break

        train_idx = valid_indices[train_idx_local]
        temp_idx = valid_indices[temp_idx_local]
        y_temp_np = cvt.tensor_to_numpy(y_pos[temp_idx])
        try:
            val_sub, test_sub = cvt.train_test_split(
                cvt.np.arange(len(y_temp_np)),
                test_size=0.5,
                random_state=seed,
                stratify=y_temp_np,
            )
        except ValueError:
            val_sub, test_sub = cvt.train_test_split(
                cvt.np.arange(len(y_temp_np)),
                test_size=0.5,
                random_state=seed,
            )

        val_idx = temp_idx[val_sub]
        test_idx = temp_idx[test_sub]

        X_train_pos = X_pos[train_idx]
        y_train_pos = y_pos[train_idx]
        X_val_pos = X_pos[val_idx]
        y_val_pos = y_pos[val_idx]
        X_test_pos = X_pos[test_idx]
        y_test_pos = y_pos[test_idx]

        X_train_pos = cvt.torch.cat([X_train_pos, X_wt_ctr_tr], dim=0)
        y_train_pos = cvt.torch.cat([y_train_pos, y_wt_ctr_tr], dim=0)
        X_val_pos = cvt.torch.cat([X_val_pos, X_wt_ctr_va], dim=0)
        y_val_pos = cvt.torch.cat([y_val_pos, y_wt_ctr_va], dim=0)

        if has_pafa:
            X_train_pos = cvt.torch.cat([X_train_pos, X_pafa_tr], dim=0)
            y_train_pos = cvt.torch.cat([y_train_pos, y_pafa_tr], dim=0)
            X_val_pos = cvt.torch.cat([X_val_pos, X_pafa_va], dim=0)
            y_val_pos = cvt.torch.cat([y_val_pos, y_pafa_va], dim=0)

        pos_train_len = len(X_train_pos)
        neg_train_len = len(X_neg_tr)
        X_train = cvt.torch.cat([X_train_pos, X_neg_tr], dim=0)
        y_train = cvt.torch.cat([y_train_pos, y_neg_tr], dim=0)
        X_val = cvt.torch.cat([X_val_pos, X_neg_va], dim=0)
        y_val = cvt.torch.cat([y_val_pos, y_neg_va], dim=0)
        X_test = cvt.torch.cat([X_test_pos, X_neg_te], dim=0)
        y_test = cvt.torch.cat([y_test_pos, y_neg_te], dim=0)

        X_train = _feature_block(X_train, split_dim, mode)
        X_val = _feature_block(X_val, split_dim, mode)
        X_test = _feature_block(X_test, split_dim, mode)

        if use_smote:
            cvt.set_seed(seed)
            X_train, y_train = cvt.apply_smote(
                X_train,
                y_train,
                device=device,
                seed=seed,
                k_neighbors=cfg["smote_k_neighbors"],
            )

        X_train_np = cvt.tensor_to_numpy(X_train)
        X_val_np = cvt.tensor_to_numpy(X_val)
        X_test_np = cvt.tensor_to_numpy(X_test)
        X_train_std, X_val_std, X_test_std, _, _ = _standardize_selected(
            cvt.np,
            X_train_np,
            X_val_np,
            X_test_np,
            split_dim,
            mode,
        )
        y_train_np = cvt.tensor_to_numpy(y_train)
        y_val_np = cvt.tensor_to_numpy(y_val)
        y_test_np = cvt.tensor_to_numpy(y_test)

        unique_cls = cvt.np.unique(y_train_np)
        cls_w = cvt.compute_class_weight("balanced", classes=unique_cls, y=y_train_np)
        weight_map = {c: w for c, w in zip(unique_cls, cls_w)}
        sample_weights = cvt.np.array([weight_map.get(int(label), 1.0) for label in y_train_np])
        val_weights = cvt.np.array([weight_map.get(int(label), 1.0) for label in y_val_np])

        sample_weights[pos_train_len : pos_train_len + neg_train_len] *= cfg["class_imbalance_factor"]
        pos_val_len = len(X_val_pos)
        val_weights[pos_val_len : pos_val_len + len(X_neg_va)] *= cfg["class_imbalance_factor"]

        rng = cvt.np.random.default_rng(seed)
        shuffle_idx = rng.permutation(len(y_train_np))
        X_train_std = X_train_std[shuffle_idx]
        y_train_np = y_train_np[shuffle_idx]
        sample_weights = sample_weights[shuffle_idx]

        params = {
            **xgb_params,
            "num_class": int(max(unique_cls)) + 1,
            "early_stopping_rounds": cfg["early_stopping_rounds"],
        }
        model = cvt.xgb.XGBClassifier(**params)
        model.fit(
            X_train_std,
            y_train_np,
            sample_weight=sample_weights,
            eval_set=[(X_train_std, y_train_np), (X_val_std, y_val_np)],
            verbose=False,
        )

        if use_hnm:
            model = cvt.apply_hard_negative_mining(
                model,
                X_train_std,
                y_train_np,
                X_val_std,
                y_val_np,
                sample_weights,
                val_weights,
                xgb_params,
                boost_factor=cfg["hnm_boost_factor"],
                finetune_fraction=cfg["hnm_finetune_fraction"],
            )

        y_pred = model.predict(X_test_std)
        try:
            y_proba = model.predict_proba(X_test_std)
        except Exception:
            y_proba = None
        metrics = cvt.evaluate_fold(y_test_np, y_pred, y_proba=y_proba, fold_idx=fold, verbose=True)
        metrics.update({"fold": fold, "feature_mode": mode})
        fold_rows.append(metrics)

    fold_df = cvt.pd.DataFrame(fold_rows)
    fold_df.to_csv(output_dir / "fold_metrics.csv", index=False)
    summary_df = _write_summary(cvt.pd, fold_df, output_dir)

    metadata = {
        "feature_mode": mode,
        "catrange_root": str(catrange_root),
        "config": str(config_path),
        "split_dim": split_dim,
        "smote": use_smote,
        "hard_negative_mining": use_hnm,
        "device": str(device),
    }
    (output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {mode} results to {output_dir}")
    return fold_df, summary_df


def main():
    args = parse_args()
    modes = args.feature_mode or ["full", "sequence", "substrate"]
    all_folds = []
    all_summary = []
    for mode in modes:
        fold_df, summary_df = run_one_mode(args, mode)
        all_folds.append(fold_df)
        summary_df = summary_df.copy()
        summary_df["feature_mode"] = mode
        all_summary.append(summary_df)

    output_dir = Path(args.output_dir).resolve()
    if all_folds:
        import pandas as pd

        pd.concat(all_folds, ignore_index=True).to_csv(output_dir / "all_fold_metrics.csv", index=False)
        pd.concat(all_summary, ignore_index=True).to_csv(output_dir / "all_summary_metrics.csv", index=False)
    print(f"Ablation complete: {output_dir}")


if __name__ == "__main__":
    main()
