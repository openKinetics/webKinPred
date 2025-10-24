"""
Train:
- base model (same as unikp)
- + esmc
- + esm2
- + binding site weights
- + pca 
- + oversampling

Do 5-fold SE-CV
"""
from pathlib import Path
import sys
import json
import math
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold
from tqdm import tqdm
import torch

sys.path.append(str(Path(__file__).resolve().parent.parent))  # add parent dir to path
from config import CONFIGS_ABLATION, BS_PRED_PATH, SEQ_LOOKUP, ROOT
from smiles_embeddings.smiles_transformer.build_vocab import WordVocab  
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
RESULTS_PKL = ROOT / "results/ablation.pkl"
OUT_DIR = ROOT / "results"

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

results_all = {}
for cfg in CONFIGS_ABLATION:
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

    fold_idx = GroupKFold(5).split(seqs, groups=groups)

    fold_idx = list(fold_idx)
    pbar = tqdm(fold_idx, desc=f"{name}", ncols=100, leave=False)

    folds_out = []
    for fold_no, (tr, te) in enumerate(pbar, 1):
        tr = np.asarray(tr, int)
        te = np.asarray(te, int)

        # baseline
        X_tr, X_te,_ = make_design_matrices(tr, te, blocks_all, names, cfg, smiles_vec)
        y_tr, y_te = y_np[tr], y_np[te]
        model, yp, m1 = train_model(X_tr, y_tr, X_te, y_te, fold=fold_no)
        # instead of print
        pbar.set_postfix({
            "config": name,
            "fold": fold_no,
            "r2": f"{m1['r2']:.4f}",
            "rmse": f"{m1['rmse']:.4f}",
        })
        folds_out.append(
            dict(
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
    results_all[name] = folds_out

RESULTS_PKL.parent.mkdir(parents=True, exist_ok=True)
pd.to_pickle(results_all, RESULTS_PKL)
print("\n✓ Results saved to", RESULTS_PKL)