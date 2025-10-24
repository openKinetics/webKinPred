#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Grid-search over CONFIGS_SMILES for **both** kinetic constants:

    • kcat   (turnover number, log10-transformed here)
    • KM     (Michaelis constant, already given as log10(KM))

For each target we keep **exactly** the same feature-extraction,
scaling, PCA-handling and model-training pipeline that was already
implemented for kcat; we simply run the whole pipeline twice and
serialize the two result dictionaries to separate .pkl files.

Only the handful of lines that really needed touching were changed;
everything else is identical to the previous working codebase.
"""

import json, math
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).resolve().parent.parent))
from typing  import Dict
import numpy  as np
import pandas as pd

from tqdm import tqdm
from sklearn.model_selection import KFold, GroupKFold

# --- project-specific imports (unchanged) -------------------------
from smiles_embeddings.smiles_transformer.build_vocab import WordVocab  
from config import RAW_DLKCAT, SEQ_LOOKUP, BS_PRED_PATH, CONFIGS_SMILES_KCAT, CONFIGS_SMILES_KM, ROOT
from utils.sequence_features import sequences_to_feature_blocks
from utils.smiles_features import smiles_to_vec
from model_training import train_model
from utils.pca import make_design_matrices
from utils.folds import get_folds

TASKS: Dict[str, Dict] = {
    "kcat": dict(
        raw_path   = RAW_DLKCAT,
        smiles_key = "Smiles",
        seq_key  = "Sequence",
        value_key  = "Value",
        value_proc = lambda v: math.log(float(v), 10),      # log10(kcat)
        record_filter = lambda d: (float(d["Value"]) > 0 and "." not in d['Smiles']),  # filter out invalid SMILES
        task_kw    = "kcat",
    ),
    "km": dict(
        raw_path   = ROOT / "data/KM_data_raw.json",
        smiles_key = "smiles",
        seq_key    = "Sequence",
        value_key  = "log10_KM",
        value_proc = lambda v: float(v),  # already log10-transformed
        record_filter = lambda d: True ,
        task_kw    = "KM",
    ),
}

RESULTS_DIR = ROOT / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

def run_grid_search(task_name: str):
    assert task_name in ["kcat", "km"], f"Invalid task: {task_name}"
    cfg_task = TASKS[task_name]

    # ------------------------------------------------------------------
    # 1. Load + pre-filter dataset
    # ------------------------------------------------------------------
    with cfg_task["raw_path"].open("r") as fp:
        raw = json.load(fp)

    raw = [d for d in raw
           if len(d[cfg_task["seq_key"]]) <= 1499
           and cfg_task["record_filter"](d)]

    sequences = [d[cfg_task["seq_key"]] for d in raw]
    smiles    = [d[cfg_task["smiles_key"]] for d in raw]
    labels_np = np.array([cfg_task["value_proc"](d[cfg_task["value_key"]])
                          for d in raw], dtype=np.float32)

    # ------------------------------------------------------------------
    # 2. Build group labels for GroupKFold
    # ------------------------------------------------------------------
    seq_id_to_seq = pd.read_pickle(SEQ_LOOKUP)
    seq_to_id = {v: k for k, v in seq_id_to_seq.items()}
    groups = [seq_to_id[seq] for seq in sequences]

    # ------------------------------------------------------------------
    # 3. Binding-site predictions dataframe 
    # -------------------------------------------------------------------
    bs_df = pd.read_csv(BS_PRED_PATH, sep="\t")

    # ------------------------------------------------------------------
    # 4. Pre-compute SMILES vectorisation once – independent of folds
    # ------------------------------------------------------------------
    smiles_vec_cache = {}       # method-> ndarray

    # ------------------------------------------------------------------
    # 5. Iterate over split-modes, configs, folds
    # ------------------------------------------------------------------
    split_modes   = ["groupkfold", "kfold"]
    all_results   = {}

    for split_mode in split_modes:
        print(f"\n===== {task_name.upper()} – {split_mode.upper()} =====")
        fold_indices = get_folds(sequences, groups, method=split_mode, n_splits=5)
        CONFIGS_SMILES = CONFIGS_SMILES_KCAT if task_name == "kcat" else CONFIGS_SMILES_KM
        prog = tqdm(CONFIGS_SMILES, total=len(CONFIGS_SMILES),
                    ncols=120, desc=f"Configs ({split_mode})")

        for cfg in prog:
            prog.set_description(f"[{split_mode}] {cfg['name']}")

            # 5.1 sequence blocks (depends on prot-rep mode but NOT on folds)
            blocks_all, block_names = sequences_to_feature_blocks(
                sequence_list=sequences,
                binding_site_df=bs_df,
                ec_num_df=None,
                cat_sites_df=None,
                seq_to_id=seq_to_id,
                use_ec_logits=False,
                use_esmc=cfg["use_esmc"],
                use_esm2=cfg["use_esm2"],
                use_t5=cfg["use_t5"],
                t5_last_layer=cfg["t5_last_layer"],
                prot_rep_mode=cfg["prot_rep_mode"],
                task=cfg_task["task_kw"]
            )
            # 5.2 SMILES vectors for this config (cache by method)
            smi_method = cfg["smiles_method"]
            if smi_method not in smiles_vec_cache:
                smiles_vec_cache[smi_method] = smiles_to_vec(smiles, method=smi_method)
            smiles_vec = smiles_vec_cache[smi_method]

            # 5.3 cross-validation
            fold_res = []
            for fold_id, (tr_idx, te_idx) in enumerate(fold_indices, 1):
                X_tr, X_te, _ = make_design_matrices(tr=tr_idx,
                                                       te=te_idx,
                                                       blocks_all=blocks_all,
                                                       names=block_names,
                                                       cfg=cfg,
                                                       smiles_vec=smiles_vec)
                y_tr, y_te = labels_np[tr_idx], labels_np[te_idx]

                # --- model training
                prog.set_description(f"[{split_mode}] {cfg['name']} – fit")
                _, y_pred, metrics = train_model(X_tr, y_tr, X_te, y_te, fold=fold_id)
                prog.set_postfix(fold=fold_id, r2=metrics["r2"])

                fold_res.append(dict(
                    task   = task_name,
                    config = cfg["name"],
                    fold   = fold_id,
                    split  = split_mode,
                    n_comps = cfg["n_comps"] if cfg["use_pca"] else None,
                    r2     = metrics["r2"],
                    rmse   = metrics.get("rmse", np.nan),
                    y_true = y_te.tolist(),
                    y_pred = y_pred.tolist(),
                ))

            all_results.setdefault(cfg["name"], []).extend(fold_res)

    # ------------------------------------------------------------------
    # 6. Persist results
    # ------------------------------------------------------------------
    out_pkl  = RESULTS_DIR / f"smiles_rep_gs_{task_name}.pkl"
    out_csv  = RESULTS_DIR / f"smiles_rep_gs_{task_name}.csv"

    pd.to_pickle(all_results, out_pkl)

    flat = [r for config_res in all_results.values() for r in config_res]
    df   = pd.DataFrame(flat)
    df.to_csv(out_csv, index=False)

    print(f"✔  Results for {task_name} saved to {out_pkl}")
# ------------------------------------------------------------------

if __name__ == "__main__":
    for _task in ["kcat", "km"]:
        run_grid_search(_task)
