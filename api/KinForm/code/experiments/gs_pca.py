
# plot everything in one boxplot, n_comps by color and esmc+prot5/esmc+esm2+prot5 by shape
import json
import numpy as np
import pandas as pd
import math
from tqdm import tqdm
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))  # add parent dir to path
from config import RAW_DLKCAT, SEQ_LOOKUP, BS_PRED_PATH, CONFIGS_PCA, CAT_PRED_DF, ROOT
import ast
from model_training import train_model
from smiles_embeddings.smiles_transformer.build_vocab import WordVocab 
from utils.smiles_features import smiles_to_vec
from utils.sequence_features import sequences_to_feature_blocks 
from utils.pca import (
    make_design_matrices
)
from utils.folds import get_folds
    
def load_data(dataset = "dlkcat"):
    assert dataset in ["dlkcat", "eitlem"], f"Invalid dataset: {dataset}"
    if dataset == "dlkcat":
        with RAW_DLKCAT.open("r") as fp:
            raw = json.load(fp)

        raw = [d for d in raw if len(d["Sequence"]) <= 1499 and float(d["Value"]) > 0 and "." not in d["Smiles"]]
        sequences = [d["Sequence"] for d in raw]
        smiles    = [d["Smiles"]    for d in raw]
        labels_np = np.array([math.log(float(d["Value"]), 10) for d in raw], dtype=np.float32)
    else:
        with open(ROOT / "data/EITLEM_data/KCAT/kcat_data.json", 'r') as fp:
            raw = json.load(fp)

        def is_valid(e):
            return len(e["sequence"]) <= 1499 and float(e["value"]) > 0

        filtered = [e for e in raw if is_valid(e)]
        sequences = [e["sequence"] for e in filtered]
        smiles    = [e["smiles"] for e in filtered]
        labels_np = np.array([math.log(float(e["value"]), 10) for e in filtered], dtype=np.float32)
    return sequences, smiles, labels_np

def main(dataset):
    sequences, smiles, labels_np = load_data(dataset=dataset)

    seq_id_to_seq = pd.read_pickle(SEQ_LOOKUP)
    seq_to_id = {v: k for k, v in seq_id_to_seq.items()}
    groups = [seq_to_id[seq] for seq in sequences]

    binding_site_df = pd.read_csv(BS_PRED_PATH, sep="\t")
    # cat_df = pd.read_csv(CAT_PRED_DF)
    # cat_df['all_AS_probs'] = cat_df['all_AS_probs'].apply(ast.literal_eval)
    cat_df = None
    print("Extracting SMILES vectors ...")
    smiles_vec = smiles_to_vec(smiles)

    split_modes = ["groupkfold","kfold"]
    all_results = {}
    for split_mode in split_modes:
        print(f"\n===== Running {split_mode.upper()} =====")
        fold_indices = get_folds(sequences, groups, method=split_mode, n_splits=5)

        progress_bar = tqdm(CONFIGS_PCA, desc=f"Configs ({split_mode})", ncols=100, total=len(CONFIGS_PCA)*5)
        for cfg in CONFIGS_PCA:
            progress_bar.set_description(f"[{split_mode}] Config: {cfg['name']}")

            blocks_all, block_names = sequences_to_feature_blocks(
                sequence_list=sequences,
                binding_site_df=binding_site_df,
                ec_num_df=None,
                cat_sites_df=cat_df,
                seq_to_id=seq_to_id,
                use_ec_logits=False,
                use_esmc=cfg["use_esmc"],
                use_esm2=cfg["use_esm2"],
                use_t5=cfg["use_t5"],
                prot_rep_mode=cfg["prot_rep_mode"],
                t5_last_layer=cfg["t5_last_layer"],
                task='kcat'
            )

            # Split blocks per fold and apply per-block scaling + PCA
            results = []
            for fold, (train_idx, test_idx) in enumerate(fold_indices, 1):
                progress_bar.update(1)
                X_tr, X_te,_ = make_design_matrices(train_idx, test_idx, blocks_all, block_names, cfg, smiles_vec)
                y_train, y_test = labels_np[train_idx], labels_np[test_idx]
                progress_bar.set_description(f"[{split_mode}] Config: {cfg['name']} - Fitting model")
                _, _, metrics = train_model(X_tr, y_train, X_te, y_test, fold=fold)
                progress_bar.set_postfix(fold=fold, r2=metrics["r2"])

                results.append(dict(
                    config=cfg["name"],
                    fold=fold,
                    split=split_mode,
                    n_comps=cfg["n_comps"] if cfg["use_pca"] else None,
                    **metrics
                ))

            if cfg["name"] not in all_results:
                all_results[cfg["name"]] = []
            all_results[cfg["name"]].extend(results)

    # Save all results in flat form
    flat_results = [entry for results in all_results.values() for entry in results]

    df = pd.DataFrame(flat_results)
    out_dir = ROOT / "results"
    out_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_dir / f"pca_gs_{dataset}.csv", index=False)
    # Save full results dict as pickle
    pd.to_pickle(all_results, out_dir / f"pca_gs_{dataset}.pkl")

if __name__ == "__main__":
    main('dlkcat')  
    main('eitlem')  