import json
import numpy as np
import pandas as pd
from tqdm import tqdm
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))
from smiles_embeddings.smiles_transformer.build_vocab import WordVocab  
from config import SEQ_LOOKUP, BS_PRED_PATH, CONFIG_H, CONFIG_UniKP, ROOT
from utils.smiles_features import smiles_to_vec
from utils.sequence_features import sequences_to_feature_blocks
from utils.pca import make_design_matrices
from utils.folds import get_folds
from model_training import train_model
CONFIG_H['name'] = "KinForm"
CONFIGS = [CONFIG_H, CONFIG_UniKP]
    
def main():
    with open(ROOT / "data/KM_data_raw.json", 'r') as fp:
        raw = json.load(fp)

    raw = [d for d in raw if len(d["Sequence"]) <= 1499 and "." not in d["smiles"]]
    sequences = [d["Sequence"] for d in raw]
    raw_smiles   = [d["smiles"] for d in raw]
    labels_np = np.array([float(d['log10_KM']) for d in raw], dtype=np.float32)

    seq_id_to_seq = pd.read_pickle(SEQ_LOOKUP)
    seq_to_id = {v: k for k, v in seq_id_to_seq.items()}
    groups = [seq_to_id[seq] for seq in sequences]

    binding_site_df = pd.read_csv(BS_PRED_PATH, sep="\t")

    split_modes = ["groupkfold", "kfold"]
    all_results = {}
    fold_splits_dict = {}

    for split_mode in split_modes:
        print(f"\n===== Running {split_mode.upper()} =====")
        fold_indices = get_folds(sequences, groups, method=split_mode, n_splits=5)
        current_fold_split = [
            {"train_indices": train_idx.tolist(), "test_indices": test_idx.tolist()}
            for train_idx, test_idx in fold_indices
        ]
        if split_mode in fold_splits_dict:
            assert fold_splits_dict[split_mode] == current_fold_split, (
                f"Mismatch in fold indices for {split_mode}!"
            )
        else:
            fold_splits_dict[split_mode] = current_fold_split

        smiles_vec = smiles_to_vec(raw_smiles, method="smiles_transformer")
        progress_bar = tqdm(CONFIGS, desc=f"Configs ({split_mode})", ncols=120, total=len(CONFIGS))
        for cfg in progress_bar:
            
            progress_bar.set_description(f"[{split_mode}] Config: {cfg['name']}")

            blocks_all, block_names = sequences_to_feature_blocks(
                sequence_list=sequences,
                binding_site_df=binding_site_df,
                ec_num_df=None,
                cat_sites_df=None,
                seq_to_id=seq_to_id,
                use_ec_logits=False,
                use_esmc=cfg["use_esmc"],
                use_esm2=cfg["use_esm2"],
                use_t5=cfg["use_t5"],
                t5_last_layer=cfg["t5_last_layer"],
                prot_rep_mode=cfg["prot_rep_mode"],
                task="KM" 
            )
            # Split blocks per fold and apply per-block scaling + PCA
            results = []
            for fold, (train_idx, test_idx) in enumerate(fold_indices, 1):
                y_train, y_test = labels_np[train_idx], labels_np[test_idx]
                X_train, X_test, _ = make_design_matrices(tr=train_idx,
                                                       te=test_idx,
                                                       blocks_all=blocks_all,
                                                       names=block_names,
                                                       cfg=cfg,
                                                       smiles_vec=smiles_vec)
                progress_bar.set_description(f"[{split_mode}] Config: {cfg['name']} - Fitting model")
                _, y_pred, metrics = train_model(X_train, y_train, X_test, y_test, fold=fold)
                progress_bar.set_postfix(fold=fold, r2=metrics["r2"])
                results.append(dict(
                    config=cfg["name"],
                    fold=fold,
                    split=split_mode,
                    n_comps=cfg["n_comps"] if cfg["use_pca"] else None,
                    r2=metrics["r2"],
                    rmse=metrics["rmse"],
                    train_idx =train_idx.tolist(),
                    test_idx=test_idx.tolist(),
                    y_true=y_test.tolist(),
                    y_pred=y_pred.tolist(),
                ))

            if cfg["name"] not in all_results:
                all_results[cfg["name"]] = []
            all_results[cfg["name"]].extend(results)

    # Save full results dict as pickle
    pd.to_pickle(all_results, ROOT / "results/unikp_comp_km.pkl")

if __name__ == "__main__":
    main()