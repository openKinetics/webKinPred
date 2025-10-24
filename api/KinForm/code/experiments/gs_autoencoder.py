"""
Grid-search script: AE vs Extra-Trees on kcat (DLKcat & EITLEM)

This is a **carbon-copy** of gs_pca.py with the PCA bit swapped for
utils.autoencoder.scale_and_reduce_blocks_ae().

Run:
    $ python gs_autoencoder.py

Author: 2025-06-15
"""

import json, math
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd
from tqdm import tqdm
import sys 
sys.path.append(str(Path(__file__).resolve().parent.parent))
from config import (
    RAW_DLKCAT, SEQ_LOOKUP, BS_PRED_PATH, CAT_PRED_DF, ROOT
)  
import ast
from model_training import train_model
from smiles_embeddings.smiles_transformer.build_vocab import WordVocab
from utils.smiles_features import smiles_to_vec
from utils.sequence_features import sequences_to_feature_blocks
from utils.autoencoder import scale_and_reduce_blocks_ae
from utils.folds import get_folds


# --------------------------------------------------------------------------- #
#                         data-loading helper (unchanged)                     #
# --------------------------------------------------------------------------- #

def load_data(dataset: str):
    assert dataset in {"dlkcat", "eitlem"}
    if dataset == "dlkcat":
        with RAW_DLKCAT.open() as fp:
            raw = json.load(fp)
        raw = [d for d in raw
               if len(d["Sequence"]) <= 1499 and float(d["Value"]) > 0
               and "." not in d["Smiles"]]
        sequences = [d["Sequence"] for d in raw]
        smiles    = [d["Smiles"]   for d in raw]
        labels_np = np.array([math.log(float(d["Value"]), 10) for d in raw],
                             dtype=np.float32)
    else:
        with open(ROOT / "data/EITLEM_data/KCAT/kcat_data.json") as fp:
            raw = json.load(fp)

        valid = [e for e in raw
                 if len(e["sequence"]) <= 1499 and float(e["value"]) > 0]
        sequences = [e["sequence"] for e in valid]
        smiles    = [e["smiles"]   for e in valid]
        labels_np = np.array([math.log(float(e["value"]), 10) for e in valid],
                             dtype=np.float32)
    return sequences, smiles, labels_np


# --------------------------------------------------------------------------- #
#                                one-and-only cfg                             #
# --------------------------------------------------------------------------- #

HIDDEN_DIMS = [32, 64, 128, 256, 512, 1024, 2048, 4096, 8192]
USE_ESM2 = True
USE_ESMC = True
USE_T5 = True
T5_LAST_LAYER = True
PROT_REP_MODE = "global+binding+cat"
CONFIG_AE = [
    dict(
        name=f"ESMC+ESM2+T5_AE{dim}",
        use_esmc=USE_ESMC,
        use_esm2=USE_ESM2,
        use_t5=USE_T5,
        t5_last_layer=T5_LAST_LAYER,
        prot_rep_mode=PROT_REP_MODE,
        use_ae=True,  # ← autoencoder case
        scale=False,
        latent_dim=dim,
        n_epochs=75,
    )
    for dim in HIDDEN_DIMS
] + [
    dict(
        name="ESMC+ESM2+T5_noAE",
        use_esmc=USE_ESMC,
        use_esm2=USE_ESM2,
        use_t5=USE_T5,
        t5_last_layer=T5_LAST_LAYER,
        prot_rep_mode=PROT_REP_MODE,
        use_ae=False,           # ← baseline case
        latent_dim=None,
        n_epochs=None,
    )
]
# --------------------------------------------------------------------------- #
#                                 main run                                    #
# --------------------------------------------------------------------------- #
def run_experiment(dataset: str):
    sequences, smiles, labels_np = load_data(dataset)

    seq_id_to_seq = pd.read_pickle(SEQ_LOOKUP)
    seq_to_id = {seq: id_ for id_, seq in seq_id_to_seq.items()}
    groups = [seq_to_id[s] for s in sequences]

    # concatenated binding-site predictions
    bs_df = pd.read_csv(BS_PRED_PATH, sep="\t")
    cat_df = pd.read_csv(CAT_PRED_DF)
    cat_df['all_AS_probs'] = cat_df['all_AS_probs'].apply(ast.literal_eval)
    print("Vectorising SMILES …")
    smiles_vec = smiles_to_vec(smiles)
    split_modes = ["groupkfold", "kfold"]
    all_results = {}
    blocks_all, block_names = sequences_to_feature_blocks(
        sequence_list=sequences,
        binding_site_df=bs_df,
        ec_num_df=None,
        cat_sites_df=cat_df,
        seq_to_id=seq_to_id,
        use_ec_logits=False,
        use_esmc=USE_ESMC,use_esm2=USE_ESM2,
        use_t5=USE_T5,t5_last_layer=T5_LAST_LAYER,
        prot_rep_mode=PROT_REP_MODE,
        task="kcat"
    )
    for split_mode in split_modes:
        print(f"\n===== {split_mode.upper()} =====")
        folds = get_folds(sequences, groups, method=split_mode, n_splits=5)

        pbar = tqdm(CONFIG_AE, ncols=100,
                    desc=f"[{split_mode}] running configs", total=len(CONFIG_AE)*5)

        for cfg in CONFIG_AE:
            pbar.set_description(f"[{split_mode}] {cfg['name']}")
            results: List[dict] = []

            for fold, (tr_idx, te_idx) in enumerate(folds, 1):
                pbar.update(1)

                # split blocks per fold
                blocks_tr = [b[tr_idx] for b in blocks_all]
                blocks_te = [b[te_idx] for b in blocks_all]

                if cfg["use_ae"]:
                    seq_tr, seq_te = scale_and_reduce_blocks_ae(
                        blocks_train=blocks_tr,
                        blocks_test=blocks_te,
                        block_names=block_names,
                        latent_dim=cfg["latent_dim"],
                        n_epochs=cfg["n_epochs"],
                        scale=cfg["scale"],  
                        batch_size=16,
                    )
                else:
                    # No AE: concatenate binding + global blocks
                    from utils.pca import reorder_blocks
                    seq_tr = np.concatenate(reorder_blocks(block_names, blocks_tr), axis=1)
                    seq_te = np.concatenate(reorder_blocks(block_names, blocks_te), axis=1)
                # build design matrices
                X_train = np.concatenate([smiles_vec[tr_idx], seq_tr], axis=1)
                X_test  = np.concatenate([smiles_vec[te_idx], seq_te], axis=1)
                y_train, y_test = labels_np[tr_idx], labels_np[te_idx]
                pbar.set_description(f"[{split_mode}] {cfg['name']} (fold {fold}) – fit ET")
                _, _, metrics = train_model(
                    X_train, y_train, X_test, y_test, fold=fold
                )
                pbar.set_postfix(r2=metrics["r2"])

                results.append(dict(
                    config=cfg["name"],
                    fold=fold,
                    split=split_mode,
                    latent_dim=cfg["latent_dim"],
                    **metrics
                ))

            all_results.setdefault(cfg["name"], []).extend(results)

    # --- persist results --------------------------------------------------- #
    out_dir = ROOT / "results"
    out_dir.mkdir(parents=True, exist_ok=True)

    flat = [e for lst in all_results.values() for e in lst]
    pd.DataFrame(flat).to_csv(out_dir / f"ae_gs_{dataset}.csv", index=False)
    pd.to_pickle(all_results, out_dir / f"ae_gs_{dataset}.pkl")


if __name__ == "__main__":
    run_experiment("dlkcat")
    # run_experiment("eitlem")
