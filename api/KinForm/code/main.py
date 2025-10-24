#!/usr/bin/env python3
"""
KinForm / UniKP – single-pass training & prediction script.

Usage
-----
Run from the code/ directory:

# TRAIN on default dataset
python main.py --mode train --task kcat --model_config KinForm-L

# TRAIN with cross-validation
python main.py --mode train --task kcat --model_config KinForm-L --train_test_split 0.8

# TRAIN on custom data
python main.py --mode train --task kcat --model_config KinForm-L --data_path ./my_data.json

# PREDICT on default dataset
python main.py --mode predict --task kcat --model_config KinForm-L --save_results ./predictions.csv

# PREDICT on custom data
python main.py --mode predict --task kcat --model_config KinForm-L \
               --save_results ./predictions.csv --data_path ./my_data.json

Custom Data Format
------------------
JSON file with array of objects:
- For kcat task: {"sequence": "MVKL...", "smiles": "CCO", "value": 123.4}
- For KM task:   {"Sequence": "MVKL...", "smiles": "CCO", "log10_KM": -2.5}

Note: All paths are relative to the repository root and work on any machine.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import tempfile
import warnings
from pathlib import Path
from typing import Dict, Tuple, List
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold, KFold
from sklearn.metrics import r2_score, mean_squared_error

# Silence warnings
warnings.filterwarnings("ignore", category=UserWarning, message=".*enable_nested_tensor.*")

# ──────────────────────────── local imports ───────────────────────── #
from config import CONFIG_H, CONFIG_L, CONFIG_UniKP, COMPUTED_EMBEDDINGS_PATHS, BS_PRED_PATH
from smiles_embeddings.smiles_transformer.build_vocab import WordVocab 
from utils.smiles_features import smiles_to_vec
from utils.sequence_features import sequences_to_feature_blocks
from utils.pca import make_design_matrices
from model_training import train_model
from pseq2sites.get_sites import get_sites
from code.utils.compute_embs import embeddings_exist, _compute_all_emb

# Global paths - relative to repository root
# This script is in code/, so go up one level to get to repo root
ROOT = Path(__file__).resolve().parent.parent
DATA_KCAT = ROOT / "data/EITLEM_data/KCAT/kcat_data.json"
DATA_KM   = ROOT / "data/KM_data_raw.json"
SEQ_LOOKUP   = ROOT / "results/sequence_id_to_sequence.pkl"
# ------------------------------------------------------------------- #
CONFIG_MAP = {
    "KinForm-H": CONFIG_H,
    "KinForm-L": CONFIG_L,
    "UniKP":     CONFIG_UniKP,
}


# ═════════════════════════ data loading ════════════════════════════ #
def _get_or_create_seq_id(sequence, seq_id_to_sequence, sequence_to_seq_id):
    if sequence in sequence_to_seq_id:
        return sequence_to_seq_id[sequence], seq_id_to_sequence, sequence_to_seq_id, False
    else:
        print(f"↪ New sequence encountered – assigning new ID.")
        new_id = f"Sequence {len(sequence_to_seq_id)+1}"
        i = 2
        while new_id in seq_id_to_sequence:
            new_id = f"Sequence {len(sequence_to_seq_id)+i}"
            i += 1
        sequence_to_seq_id[sequence] = new_id
        seq_id_to_sequence[new_id] = sequence
        return new_id, seq_id_to_sequence, sequence_to_seq_id, True

def compute_embeddings(sequences: List[str]) -> Tuple[Dict[str, bool], Dict[str, str]]:
    """
    Input:
        sequences : List[str]
            List of protein sequences
    Output:
        computed : Dict[str, bool]
            Dictionary indicating whether embeddings were computed for each sequence
        reasons : Dict[str, str]
            Dictionary of reasons for computation (e.g., missing embeddings)
    """
    # Check which embeddings are missing
    seq_id_to_sequence = pd.read_pickle(SEQ_LOOKUP)
    sequence_to_seq_id = {v: k for k, v in seq_id_to_sequence.items()}
    seq_ids = []
    changed_list = []
    for seq in sequences:
        seq_id, seq_id_to_sequence, sequence_to_seq_id, changed = _get_or_create_seq_id(
            seq, seq_id_to_sequence, sequence_to_seq_id
        )
        seq_ids.append(seq_id)
        changed_list.append(changed)
    if any(changed_list):
        print("↪ New sequences were added to the sequence ID lookup. Updating cache...")
        pd.to_pickle(seq_id_to_sequence, SEQ_LOOKUP)

    computed_dict = {seq_id: None for seq_id in seq_ids}
    reasons = {seq_id: [] for seq_id in seq_ids}

    exists_dict = embeddings_exist(seq_ids)
    esm2_exists, esmc_exists, t5_exists = (
        exists_dict["esm2"],
        exists_dict["esmc"],
        exists_dict["t5"],
    )
    if (all(esm2_exists) and all(esmc_exists) and all(t5_exists)):
        print("✓ All embeddings already exist. No computation needed.")
        return [True] * len(sequences)
    print(f"Missing {sum(not x for x in esm2_exists)} ESM-2 embeddings.")
    print(f"Missing {sum(not x for x in esmc_exists)} ESM-C embeddings.")
    print(f"Missing {sum(not x for x in t5_exists)} Prot-T5 embeddings.")
    # load binding site DF
    bs_df = pd.read_csv(BS_PRED_PATH, sep="\t")
    bs_df_ids = bs_df['PDB'].tolist()
    missing_bs_seqs = set(seq_ids) - set(bs_df_ids)
    print(f"Missing binding-site predictions for {len(missing_bs_seqs)} sequences.")
    # compute ProtT5 embeddings (in batches):
    bs_computed, bs_reasons, _ = get_sites(missing_bs_seqs, seq_id_to_sequence, bs_df, save_path=BS_PRED_PATH, return_prot_t5=False)
    # update bool list
    for seq_id in bs_computed:
        computed_dict[seq_id] = bs_computed[seq_id]
        reasons[seq_id].append(bs_reasons[seq_id])
    (esmc_computed, esmc_reasons, esm2_computed,
     esm2_reasons, t5_computed, t5_reasons) = _compute_all_emb(sequences, seq_id_to_sequence)
    for seq_id in seq_ids:
        if not esm2_exists[seq_ids.index(seq_id)]:
            computed_dict[seq_id] = esm2_computed[seq_id]
            reasons[seq_id].append(esm2_reasons[seq_id])
        if not esmc_exists[seq_ids.index(seq_id)]:
            computed_dict[seq_id] = esmc_computed[seq_id]
            reasons[seq_id].append(esmc_reasons[seq_id])
        if not t5_exists[seq_ids.index(seq_id)]:
            computed_dict[seq_id] = t5_computed[seq_id]
            reasons[seq_id].append(t5_reasons[seq_id])
    return computed_dict, reasons

def load_kcat(data_path: Path | None = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return sequences, smiles and log10(kcat) as numpy arrays."""
    data_file = data_path if data_path else DATA_KCAT
    print(f"Loading kcat data from {data_file}...")
    
    with data_file.open() as fp:
        raw = json.load(fp)
    
    valid = [(r["sequence"], r["smiles"], float(r["value"]))
            for r in raw if len(r["sequence"]) <= 1499 and float(r["value"]) > 0]
    seqs, smis, y = zip(*valid)
    y = np.array([math.log(v, 10) for v in y], dtype=np.float32)
    emb_computed, reasons = compute_embeddings(list(seqs))
    if not all(emb_computed):
        failed_seqs = [seqs[i] for i, v in enumerate(emb_computed) if not v]
        print(f"Warning: Embedding computation failed for {len(failed_seqs)} sequences.")
        for seq_id, reason_list in reasons.items():
            if any(reason_list):
                print(f"  - Sequence ID {seq_id}: {'; '.join([r for r in reason_list if r])}")
    return np.asarray(seqs), np.asarray(smis), y

def load_km(data_path: Path | None = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return sequences, smiles and log10(KM) as numpy arrays."""
    data_file = data_path if data_path else DATA_KM
    print(f"Loading KM data from {data_file}...")
    
    with data_file.open() as fp:
        raw = json.load(fp)
    
    valid = [(r["Sequence"], r["smiles"], float(r["log10_KM"]))
            for r in raw if len(r["Sequence"]) <= 1499 and "." not in r["smiles"]]
    seqs, smis, y = zip(*valid)
    emb_computed, reasons = compute_embeddings(list(seqs))
    if not all(emb_computed):
        failed_seqs = [seqs[i] for i, v in enumerate(emb_computed) if not v]
        print(f"Warning: Embedding computation failed for {len(failed_seqs)} sequences.")
        for seq_id, reason_list in reasons.items():
            if any(reason_list):
                print(f"  - Sequence ID {seq_id}: {'; '.join([r for r in reason_list if r])}")
    return np.asarray(seqs), np.asarray(smis), np.asarray(y, dtype=np.float32)


def get_dataset(task: str, data_path: Path | None = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Load dataset for the specified task.
    
    Parameters
    ----------
    task : str
        Either 'kcat' or 'KM'
    data_path : Path | None
        Optional path to custom JSON data file. If None, uses default dataset.
    
    Returns
    -------
    Tuple[np.ndarray, np.ndarray, np.ndarray]
        (sequences, smiles, target_values)
    """
    if task.lower() == "kcat":
        return load_kcat(data_path)
    if task.lower() == "km":
        return load_km(data_path)
    raise ValueError(f"Unknown task: {task}")


# ═════════════ feature construction (+ optional PCA) ═══════════════ #
def build_design_matrix(
    seqs: np.ndarray,
    smis: np.ndarray,
    cfg: Dict,
    task: str = "kcat",
    transformers: dict | None = None,
) -> tuple[np.ndarray, dict | None]:
    """
    Convert sequences & SMILES to the final feature matrix.
    Uses make_design_matrices which handles PCA internally.
    """
    # Binding-site predictions
    bs_df = pd.read_csv(BS_PRED_PATH, sep="\t")

    # map sequence → id (for GroupKFold compatibility, but we only need ids here)
    seq_lookup = pd.read_pickle(SEQ_LOOKUP)
    seq_to_id = {v: k for k, v in seq_lookup.items()}

    blocks_all, block_names = sequences_to_feature_blocks(
        sequence_list=seqs,
        binding_site_df=bs_df,
        cat_sites_df=None,
        ec_num_df=None,
        seq_to_id=seq_to_id,
        use_ec_logits=False,
        use_esmc=cfg["use_esmc"],
        use_esm2=cfg["use_esm2"],
        use_t5=cfg["use_t5"],
        t5_last_layer=cfg.get("t5_last_layer", -1),
        prot_rep_mode=cfg["prot_rep_mode"],
        task=task,
    )

    smiles_vec = smiles_to_vec(smis, method="smiles_transformer")

    # Use all indices for training
    all_idx = np.arange(len(seqs))
    X, _, fitted = make_design_matrices(all_idx, all_idx, blocks_all, block_names, cfg, smiles_vec, transformers=transformers)

    return X, fitted


# ═════════════════════════ main routine ════════════════════════════ #
def train(task: str, cfg_name: str, model_dir: Path, train_test_split: float = 1.0, data_path: Path | None = None) -> None:
    model_dir.mkdir(parents=True, exist_ok=True)
    cfg = CONFIG_MAP[cfg_name]

    seqs, smis, y = get_dataset(task, data_path)
    print(f"✓ Loaded {len(seqs)} {task} samples with sequences and SMILES.")
    
    # If train_test_split < 1.0, perform cross-validation
    if train_test_split < 1.0:
        print(f"\n{'='*70}")
        print(f"Running cross-validation with {int(train_test_split*100)}% train split")
        print(f"{'='*70}\n")
        
        # Get sequence groups for GroupKFold
        seq_lookup = pd.read_pickle(SEQ_LOOKUP)
        seq_to_id = {v: k for k, v in seq_lookup.items()}
        groups = [seq_to_id[s] for s in seqs]
        # Binding-site predictions (load once)
        bs_df = pd.read_csv(BS_PRED_PATH, sep="\t")
        # Build feature blocks once
        blocks_all, block_names = sequences_to_feature_blocks(
            sequence_list=seqs,
            binding_site_df=bs_df,
            cat_sites_df=None,
            ec_num_df=None,
            seq_to_id=seq_to_id,
            use_ec_logits=False,
            use_esmc=cfg["use_esmc"],
            use_esm2=cfg["use_esm2"],
            use_t5=cfg["use_t5"],
            t5_last_layer=cfg.get("t5_last_layer", -1),
            prot_rep_mode=cfg["prot_rep_mode"],
            task=task,
        )
        
        smiles_vec = smiles_to_vec(smis, method="smiles_transformer")
        
        # Run both KFold and GroupKFold
        for split_mode in ["kfold", "groupkfold"]:
            print(f"\n{'-'*70}")
            print(f"Running {split_mode.upper()}")
            print(f"{'-'*70}")
            
            if split_mode == "kfold":
                cv = KFold(n_splits=5, shuffle=True, random_state=42).split(seqs)
            else:
                cv = GroupKFold(n_splits=5).split(seqs, groups=groups)
            
            fold_results: List[Dict] = []
            
            for fold_no, (tr_idx, te_idx) in enumerate(cv, 1):
                tr_idx = np.asarray(tr_idx, int)
                te_idx = np.asarray(te_idx, int)
                
                # Build design matrices
                # For CV folds, do not reuse fitted transformers across folds
                X_tr, X_te, _ = make_design_matrices(
                    tr_idx, te_idx, blocks_all, block_names, cfg, smiles_vec
                )
                y_tr, y_te = y[tr_idx], y[te_idx]
                
                # Train model
                et_params = cfg.get("et_params", None)
                model, y_pred, metrics = train_model(
                    X_tr, y_tr, X_te, y_te, fold=fold_no, et_params=et_params
                )
                
                # Save model
                fold_model_dir = model_dir / split_mode / f"fold{fold_no}"
                fold_model_dir.mkdir(parents=True, exist_ok=True)
                joblib.dump(model, fold_model_dir / "model.joblib")
                
                fold_results.append({
                    "fold": fold_no,
                    "r2": metrics["r2"],
                    "mse": metrics["mse"],
                    "rmse": metrics["rmse"],
                })
                
                print(f"  Fold {fold_no}: R²={metrics['r2']:.4f}, "
                      f"MSE={metrics['mse']:.4f}, RMSE={metrics['rmse']:.4f}")
            
            # Print summary statistics
            r2_scores = [r["r2"] for r in fold_results]
            mse_scores = [r["mse"] for r in fold_results]
            rmse_scores = [r["rmse"] for r in fold_results]
            
            print(f"\n{split_mode.upper()} Summary:")
            print(f"  R²   : {np.mean(r2_scores):.4f} ± {np.std(r2_scores):.4f}")
            print(f"  MSE  : {np.mean(mse_scores):.4f} ± {np.std(mse_scores):.4f}")
            print(f"  RMSE : {np.mean(rmse_scores):.4f} ± {np.std(rmse_scores):.4f}")
        
        print(f"\n{'='*70}")
        print(f"✓ Cross-validation complete. Models saved to {model_dir}")
        print(f"{'='*70}\n")
        
    else:
        # Original behavior: train on all data
        X, fitted = build_design_matrix(seqs, smis, cfg, task=task)
        print(f"✓ Built design matrix with shape {X.shape}.")

        if cfg_name == "KinForm-L" and task == "kcat":
            from utils.oversampling import (
                oversample_similarity_balanced_indices,
                oversample_kcat_balanced_indices,
            )
            print("↪ Performing similarity-based oversampling...")
            indices = np.arange(len(seqs))
            indices = oversample_similarity_balanced_indices(indices, seqs)
            print(f"  ↪ After similarity oversampling: {len(indices)} samples")
            indices = oversample_kcat_balanced_indices(indices, y)
            print(f"  ↪ After kcat oversampling: {len(indices)} samples")
            X = X[indices]
            y = y[indices]

        model, _, metrics = train_model(X, y, X, y, fold=0)
        print(f"✓ Training finished – R² on full data: {metrics['r2']:.3f}")

        joblib.dump(model, model_dir / "model.joblib")
        print(f"✓ Model saved to {model_dir}")

        # If PCA was used, save the fitted transformers (scalers + PCA) used to build X
        if cfg.get("use_pca", False) and fitted is not None:
            joblib.dump(fitted, model_dir / "transformers.joblib")
            print(f"✓ Transformers saved to {model_dir / 'transformers.joblib'}")


def predict(task: str, cfg_name: str, model_dir: Path, csv_out: Path, data_path: Path | None = None) -> None:
    model = joblib.load(model_dir / "model.joblib")

    seqs, smis, y_true_log = get_dataset(task, data_path)
    cfg = CONFIG_MAP[cfg_name]

    # Load transformers if present and PCA is used by the config
    transformers = None
    if cfg.get("use_pca", False) and (model_dir / "transformers.joblib").exists():
        transformers = joblib.load(model_dir / "transformers.joblib")
        print(f"✓ Loaded transformers from {model_dir / 'transformers.joblib'}")

    X, _ = build_design_matrix(seqs, smis, cfg, task=task, transformers=transformers)
    y_pred_log = model.predict(X)
    
    # Convert from log10 scale back to original scale
    if task.lower() == "kcat":
        y_true = 10 ** y_true_log
        y_pred = 10 ** y_pred_log
    else:  # KM task - also in log10
        y_true = 10 ** y_true_log
        y_pred = 10 ** y_pred_log

    out = pd.DataFrame({
        "sequence": seqs,
        "smiles":   smis,
        "y_true":   y_true,
        "y_pred":   y_pred,
    })
    csv_out.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(csv_out, index=False)
    print(f"✓ Predictions saved to {csv_out} (values in original scale, not log10)")


# ══════════════════════════ CLI parser ═════════════════════════════ #
if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Single-pass training / inference for KinForm & UniKP.")
    p.add_argument("--mode", required=True, choices=["train", "predict"],
                help="'train' – fit model on all data; 'predict' – run inference with a saved model")
    p.add_argument("--task", required=True, choices=["kcat", "KM"],
                help="What to train/predict on (kcat or KM)")
    p.add_argument("--model_config", required=True, choices=["KinForm-H", "KinForm-L", "UniKP"],
                help="Which model configuration to use")
    p.add_argument("--save_results", type=Path,
                help="CSV path for predictions (required in predict mode)")
    p.add_argument("--train_test_split", type=float, default=1.0,
                help="Proportion of data to use for training (default: 1.0 = all data). "
                     "If < 1.0, performs 5-fold KFold and GroupKFold cross-validation.")
    p.add_argument("--data_path", type=Path,
                help="Optional path to custom data JSON file. "
                     "For kcat: JSON with 'sequence', 'smiles', 'value' (raw kcat, not log). "
                     "For KM: JSON with 'Sequence', 'smiles', 'log10_KM'. "
                     "If not provided, uses default datasets.")

    args = p.parse_args()
    results_dir = ROOT / "results"
    os.makedirs(results_dir, exist_ok=True)
    model_dir = results_dir / f"./trained_models/{args.task}_{args.model_config}"
    
    if args.mode == "train":
        if args.train_test_split <= 0.0 or args.train_test_split > 1.0:
            p.error("--train_test_split must be in range (0.0, 1.0]")
        train(args.task, args.model_config, model_dir, args.train_test_split, args.data_path)
    else:  # predict
        if args.save_results is None:
            p.error("--save_results is required in predict mode")
        predict(args.task, args.model_config, model_dir, args.save_results, args.data_path)

"""
TODO:
- If protein embeddings OR binding-site preds missing, compute on the fly (with caching)
    - for each missing protein (or in batch):
        - compute embedding from each PLLM (if missing) --subprocess
        - compute binding-site predictions (if missing) --subprocess
        - Cache mean vector and weighted mean vector
"""