#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Training script for UniKP / KinForm ablations on the DLKcat kcat dataset.

Regimes per CV fold
───────────────────
  • baseline                          – original train indices
  • _OS-LS (band-balanced oversample) – equality across 5 identity bands
                                        (<0.20, 0.20–0.39, 0.40–0.59,
                                         0.60–0.89, ≥0.90)

Author: ChatGPT   |   2025-05-24
"""
from __future__ import annotations
import sys
from pathlib import Path
import json
import math
import random
import joblib
from typing import Dict, List
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold, KFold
from tqdm import tqdm
import ast
import scipy.sparse as sp
sys.path.append(str(Path(__file__).resolve().parent.parent))  # add parent dir to path
# ────────────────────────── local modules ─────────────────────────── #
from smiles_embeddings.smiles_transformer.build_vocab import WordVocab  
from config import (                                               # type: ignore
    RAW_DLKCAT,
    SEQ_LOOKUP,
    BS_PRED_PATH,
    CAT_PRED_DF,
    CONFIG_L,CONFIG_H, CONFIG_UniKP, ROOT
)
from utils.smiles_features import smiles_to_vec
from utils.sequence_features import sequences_to_feature_blocks
from model_training import train_model
from utils.pca import make_design_matrices
from utils.oversampling import (
    oversample_similarity_balanced_indices,
    oversample_kcat_balanced_indices,
)
# ─────────────────────── global constants ─────────────────────────── #
SEED = 42
random.seed(SEED)
np.random.seed(SEED)

CONFIGS = [CONFIG_L, CONFIG_H, CONFIG_UniKP]

# output ------------------------------------------------------------- #
OUT_DIR = ROOT / "results"
OUT_DIR.mkdir(parents=True, exist_ok=True)
OUT_PKL = OUT_DIR / "unikp_kineform_dlkcat_data.pkl"
CONFIG_PARAM_KEYS = ["name", "use_pca", "n_comps", "prot_rep_mode", 
                      "use_esmc", "use_esm2", "use_t5", "t5_last_layer",
                        "model_type"]
# ───────────────────────────────── main ───────────────────────────── #
def main():
    # 1. load raw --------------------------------------------------------
    raw = [
        d
        for d in json.loads(RAW_DLKCAT.read_text())
        if len(d["Sequence"]) <= 1499 and float(d["Value"]) > 0 and "." not in d["Smiles"]
    ]
    sequences = [d["Sequence"] for d in raw]
    raw_smiles = [d["Smiles"] for d in raw]
    y_full = np.array([math.log(float(d["Value"]), 10) for d in raw], dtype=np.float32)

    # protein-ID groups for GroupKFold
    seq_to_id = {v: k for k, v in pd.read_pickle(SEQ_LOOKUP).items()}
    groups = [seq_to_id[s] for s in sequences]

    # global ligand features
    # smiles_vec = smiles_to_vec(smiles, method="smiles_transformer")

    # binding-site predictions (pre-computed)
    bs_df = pd.read_csv(BS_PRED_PATH, sep="\t")

    smiles_vec = smiles_to_vec(raw_smiles, method="smiles_transformer")
    # cat_df = pd.read_csv(CAT_PRED_DF)
    # cat_df['all_AS_probs'] = cat_df['all_AS_probs'].apply(ast.literal_eval)
    cat_df = None  
    results_all: Dict[str, List[Dict]] = {}
    fold_splits: Dict[str, List[Dict]] = {}

    for split_mode in ("groupkfold", "kfold"):
        cv = (
            GroupKFold(5).split(sequences, groups=groups)
            if split_mode == "groupkfold"
            else KFold(5, shuffle=True, random_state=SEED).split(sequences)
        )
        cv = list(cv)
        fold_splits[split_mode] = [
            {"train_indices": tr.tolist(), "test_indices": te.tolist()} for tr, te in cv
        ]

        pbar = tqdm(CONFIGS, total=len(CONFIGS) * 5, desc=f"Configs ({split_mode})", ncols=100)

        for cfg in CONFIGS:
            cfg_name = cfg["name"]
            pbar.set_description(f"[{split_mode}] {cfg_name}")

            # expensive: build sequence blocks once per CFG
            blocks_all, block_names = sequences_to_feature_blocks(
                sequence_list=sequences,
                binding_site_df=bs_df,
                ec_num_df=None,
                cat_sites_df=cat_df,
                seq_to_id=seq_to_id,
                use_ec_logits=None,
                use_esmc=cfg["use_esmc"],
                use_esm2=cfg["use_esm2"],
                use_t5=cfg["use_t5"],
                prot_rep_mode=cfg["prot_rep_mode"],
                t5_last_layer=cfg["t5_last_layer"],
                task="kcat",
            )
            for fold_no, (tr_idx, te_idx) in enumerate(cv, 1):
                pbar.update(1)
                tr_idx = np.asarray(tr_idx, int)
                te_idx = np.asarray(te_idx, int)

                X_tr, X_te,_ = make_design_matrices(tr_idx, te_idx, blocks_all, block_names, cfg, smiles_vec)

                y_tr, y_te = y_full[tr_idx], y_full[te_idx]
                et_params = cfg.get("et_params", None)
                model, y_pred, m = train_model(
                    X_tr, y_tr, X_te, y_te, fold=fold_no, et_params=et_params
                )
                model_out_dir = OUT_DIR / "ETmodels" / "dlkcat_dataset" / cfg_name / f"fold{fold_no}"
                model_out_dir.mkdir(parents=True, exist_ok=True)
                joblib.dump(model, model_out_dir / "model.joblib")
                base_rec = dict(
                    config=cfg_name,
                    split=split_mode,
                    fold=fold_no,
                    r2=m["r2"],
                    rmse=m["rmse"],
                    y_true=y_te.tolist(),
                    y_pred=y_pred.tolist(),
                    n_comps=cfg["n_comps"],
                    train_idx=tr_idx.tolist(),
                    test_idx=te_idx.tolist(),
                )
                results_all.setdefault(cfg_name, []).append(base_rec)
                pbar.set_postfix(
                    {
                        "fold": fold_no,
                        "r2": f"{m['r2']:.3f}",
                    }
                )

                if cfg_name.startswith("KinForm-L"):
                    tr_bal = oversample_similarity_balanced_indices(tr_idx, sequences)
                    print(f"Fold {fold_no} ({split_mode}): oversampled to {len(tr_bal)} rows from {len(tr_idx)} original rows using similarity-based oversampling.")
                    tr_bal = oversample_kcat_balanced_indices(tr_bal, y_full)
                    print(f"Fold {fold_no} ({split_mode}): oversampled to {len(tr_bal)} rows from {len(tr_idx)} original rows using kcat-based oversampling.")
                    Xb_tr, Xb_te,_ = make_design_matrices(tr_bal, te_idx, blocks_all, block_names, cfg, smiles_vec)
                    yb_tr = y_full[tr_bal]
                    tag = cfg_name + "(OS)"
                    model, y_pred_bal, m_bal = train_model(Xb_tr, yb_tr, Xb_te, y_te, fold=fold_no)
                    model_out_dir = OUT_DIR / "ETmodels" / "dlkcat_dataset" / tag / f"fold{fold_no}"
                    model_out_dir.mkdir(parents=True, exist_ok=True)
                    joblib.dump(model, model_out_dir / "model.joblib")
                    results_all.setdefault(tag, []).append(
                        {
                            **base_rec,
                            "config": tag,
                            "r2": m_bal["r2"],
                            "rmse": m_bal["rmse"],
                            "y_pred": y_pred_bal.tolist(),
                            "train_idx": tr_bal.tolist(),
                        }
                    ) 
    pd.to_pickle(results_all, OUT_PKL)
    print("\n✓ Results written to:", OUT_PKL)

if __name__ == "__main__":
    main()
