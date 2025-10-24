#!/usr/bin/env python3
"""
End-to-end evaluation script for UniKP / KinForm models on the EITLEM kcat
dataset (fixed split + KFold + GroupKFold).

Only two regimes per configuration:
  • baseline
  • _OS-LS   (band-balanced oversample, see DLKcat script for details)
"""
from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Dict, List
import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import GroupKFold, KFold
from tqdm import tqdm
import joblib
import scipy.sparse as sp
# local imports ------------------------------------------------------ #
import sys
sys.path.append(str(Path(__file__).resolve().parent.parent))  # add parent dir to path
from smiles_embeddings.smiles_transformer.build_vocab import WordVocab  
from config import SEQ_LOOKUP, BS_PRED_PATH, CONFIG_L, CONFIG_H, CONFIG_UniKP,ROOT
from utils.smiles_features import smiles_to_vec
from utils.sequence_features import sequences_to_feature_blocks
from model_training import train_model
from utils.pca import make_design_matrices
from utils.oversampling import (
    oversample_similarity_balanced_indices,
    oversample_kcat_balanced_indices,
)
SEED = 42
np.random.seed(SEED)
# paths -------------------------------------------------------------- #
EITLEM_DIR  = ROOT / "data/EITLEM_data"
JSON_FILE   = EITLEM_DIR / "KCAT/kcat_data.json"
TRAIN_PAIRS = EITLEM_DIR / "KCAT/KCATTrainPairInfo"
TEST_PAIRS  = EITLEM_DIR / "KCAT/KCATTestPairInfo"
RESULTS_PKL = ROOT / "results/unikp_comp_eitlem.pkl"
OUT_DIR = ROOT / "results"

CONFIGS = [CONFIG_L, CONFIG_H, CONFIG_UniKP]

# ───────────────────────── data loading ──────────────────────────── #
with JSON_FILE.open() as fp:
    raw = json.load(fp)
valid = [(i, r) for i, r in enumerate(raw) if len(r["sequence"]) <= 1499 and float(r["value"]) > 0]
orig_idx = [i for i, _ in valid]
seqs = [r["sequence"] for _, r in valid]
raw_smi  = [r["smiles"]   for _, r in valid]
y_np = np.array([math.log(float(r["value"]), 10) for _, r in valid], np.float32)
raw_to_pos = {raw_i: pos for pos, raw_i in enumerate(orig_idx)}

# fixed split indices
train_pairs = torch.load(TRAIN_PAIRS)
test_pairs  = torch.load(TEST_PAIRS)
pre_tr = np.array([raw_to_pos[p[3][0]] for p in train_pairs], int)
pre_te = np.array([raw_to_pos[p[3][0]] for p in test_pairs], int)

# groups for GroupKFold
lookup = pd.read_pickle(SEQ_LOOKUP)
seq_to_id = {v: k for k, v in lookup.items()}
groups = [seq_to_id[s] for s in seqs]

# binding-site predictions
bs_df = pd.read_csv(BS_PRED_PATH, sep="\t")
smiles_vec = smiles_to_vec(raw_smi, method="smiles_transformer")

results_all: Dict[str, Dict[str, List[Dict]]] = {}

# ───────────────────────── evaluation loop ───────────────────────── #
for cfg in CONFIGS:
    name = cfg["name"]
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

    cfg_res: Dict[str, List[Dict]] = {}

    # fixed split ------------------------------------------------------
    X_tr, X_te, _ = make_design_matrices(pre_tr, pre_te, blocks_all, names, cfg, smiles_vec)
    y_tr, y_te = y_np[pre_tr], y_np[pre_te]
    et_params = cfg.get("et_params", None)
    _, y_pred, m = train_model(X_tr, y_tr, X_te, y_te,fold=42, et_params=et_params)
    cfg_res["fixed"] = [
        dict(
            split="fixed",
            config=name,
            r2=m["r2"],
            rmse=m["rmse"],
            y_true=y_te.tolist(),
            y_pred=y_pred.tolist(),
            train_indices=pre_tr.tolist(),
            test_indices=pre_te.tolist(),
            n_comps=cfg["n_comps"],
        )
    ]

    # cross-validation -------------------------------------------------
    for mode in ("groupkfold","kfold"):
        if mode == "kfold":
            fold_idx = KFold(5, shuffle=True, random_state=SEED).split(seqs)
        else:
            fold_idx = GroupKFold(5).split(seqs, groups=groups)

        fold_idx = list(fold_idx)
        pbar = tqdm(fold_idx, desc=f"{name}/{mode}", ncols=100, leave=False)

        folds_out: List[Dict] = []
        for fold_no, (tr, te) in enumerate(pbar, 1):
            tr = np.asarray(tr, int)
            te = np.asarray(te, int)

            # baseline
            X_tr, X_te,_ = make_design_matrices(tr, te, blocks_all, names, cfg, smiles_vec)
            y_tr, y_te = y_np[tr], y_np[te]
            model, yp, m1 = train_model(X_tr, y_tr, X_te, y_te, fold=fold_no, et_params=et_params)

            model_out_dir = OUT_DIR / "ETmodels" / "eitlem_dataset" / name / f"fold{fold_no}"
            model_out_dir.mkdir(parents=True, exist_ok=True)
            joblib.dump(model, model_out_dir / "model.joblib")
            # instead of print
            pbar.set_postfix({
                "config": name,
                "fold": fold_no,
                "r2": f"{m1['r2']:.4f}",
                "rmse": f"{m1['rmse']:.4f}",
            })
            folds_out.append(
                dict(
                    split=mode,
                    fold=fold_no,
                    config=name,
                    r2=m1["r2"],
                    rmse=m1["rmse"],
                    y_true=y_te.tolist(),
                    y_pred=yp.tolist(),
                    train_indices=tr.tolist(),
                    test_indices=te.tolist(),
                    n_comps=cfg["n_comps"],
                )
            )

            if name.startswith("KinForm-L"):
                tr_bal = oversample_similarity_balanced_indices(tr, seqs)
                print(f"Fold {fold_no} ({mode}): oversampled {len(tr_bal)} rows from {len(tr)} original rows. Using similarity-based oversampling.")
                tr_bal = oversample_kcat_balanced_indices(tr_bal, y_np)
                print(f"Fold {fold_no} ({mode}): oversampled {len(tr_bal)} rows from {len(tr)} original rows. Using kcat-based oversampling.")
                Xb_tr, Xb_te,_ = make_design_matrices(tr_bal, te, blocks_all, names, cfg, smiles_vec)
                yb_tr = y_np[tr_bal]
                model, yp2, m2 = train_model(Xb_tr, yb_tr, Xb_te, y_te, fold=fold_no)
                tag = name + "(OS)" 
                model_out_dir = OUT_DIR / "ETmodels" / "eitlem_dataset" / tag / f"fold{fold_no}"
                model_out_dir.mkdir(parents=True, exist_ok=True)
                joblib.dump(model, model_out_dir / "model.joblib")
                folds_out.append(
                    dict(
                        split=mode,
                        fold=fold_no,
                        config=tag,
                        r2=m2["r2"],
                        rmse=m2["rmse"],
                        y_true=y_te.tolist(),
                        y_pred=yp2.tolist(),
                        train_indices=tr_bal.tolist(),
                        test_indices=te.tolist(),
                        n_comps=cfg["n_comps"],
                    )
                )
        cfg_res[mode] = folds_out

    results_all[name] = cfg_res

# save ----------------------------------------------------------------
RESULTS_PKL.parent.mkdir(parents=True, exist_ok=True)
pd.to_pickle(results_all, RESULTS_PKL)
print("\n✓ Results saved to", RESULTS_PKL)