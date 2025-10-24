import json
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold, GroupKFold
from tqdm import tqdm
import math
import sys
from pathlib import Path
import ast
sys.path.append(str(Path(__file__).resolve().parent.parent))
from smiles_embeddings.smiles_transformer.build_vocab import WordVocab  
from config import RAW_DLKCAT, SEQ_LOOKUP, BS_PRED_PATH, ROOT, CONFIGS
from utils.smiles_features import smiles_to_vec
from utils.sequence_features import sequences_to_features
from utils.utils import normalize_logits
from model_training import train_model

def get_folds(sequences, groups, method="kfold", n_splits=5):
    if method == "groupkfold":
        splitter = GroupKFold(n_splits=n_splits)
        return list(splitter.split(sequences, groups=groups))
    elif method == "kfold":
        splitter = KFold(n_splits=n_splits, shuffle=True, random_state=42)
        return list(splitter.split(sequences))
    else:
        raise ValueError(f"Unsupported fold method: {method}")

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
    binding_site_df = pd.read_csv(BS_PRED_PATH, sep="\t")
    # cat_df = pd.read_csv(CAT_PRED_DF)
    # cat_df['all_AS_probs'] = cat_df['all_AS_probs'].apply(ast.literal_eval)
    cat_df = None
    print("Extracting SMILES vectors ...")
    smiles_vec = smiles_to_vec(smiles,method='smiles_transformer')
    groups = [seq_to_id[seq] for seq in sequences]
    split_modes = ["groupkfold", "kfold"]
    all_results = {}

    for split_mode in split_modes:
        print(f"\n===== Running {split_mode.upper()} =====")
        fold_indices = get_folds(sequences, groups, method=split_mode, n_splits=5)

        progress_bar = tqdm(CONFIGS, desc=f"Configs ({split_mode})", ncols=100, total=len(CONFIGS)*5)
        for cfg in CONFIGS:
            progress_bar.set_description(f"[{split_mode}] Config: {cfg['name']}")

            seq_vec = sequences_to_features(
                sequence_list=sequences, 
                binding_site_df=binding_site_df,
                ec_num_df=None,
                cat_sites_df=cat_df,
                use_ec_logits=False,
                seq_to_id=seq_to_id,
                use_esmc=cfg["use_esmc"],
                use_esm2=cfg["use_esm2"],
                use_t5=cfg["use_t5"],
                t5_last_layer=True,
                prot_rep_mode=cfg["prot_rep_mode"],
                task='kcat'
            )
            results = []
            for fold, (train_idx, test_idx) in enumerate(fold_indices, 1):
                progress_bar.update(1)
                seq_train, seq_test = seq_vec[train_idx], seq_vec[test_idx]
                smi_train, smi_test = smiles_vec[train_idx], smiles_vec[test_idx]
                y_train, y_test = labels_np[train_idx], labels_np[test_idx]

                # No PCA here — concat directly
                X_train = np.concatenate([smi_train, seq_train], axis=1)
                X_test = np.concatenate([smi_test, seq_test], axis=1)

                _, _, metrics = train_model(X_train, y_train, X_test, y_test, fold=fold)
                progress_bar.set_postfix(metrics)
                results.append(dict(
                    config=cfg["name"],
                    fold=fold,
                    split=split_mode,
                    **metrics
                ))
            if cfg["name"] not in all_results:
                all_results[cfg["name"]] = []
            all_results[cfg["name"]].extend(results)
    flat_results = [entry for results in all_results.values() for entry in results]
    df_metrics = pd.DataFrame(flat_results)
    out_dir = ROOT / "results"
    out_dir.mkdir(parents=True, exist_ok=True)
    df_metrics.to_csv(out_dir / f"prot_rep_gs.csv_{dataset}", index=False)
    pd.to_pickle(all_results, out_dir / f"prot_rep_gs_{dataset}.pkl")

if __name__ == "__main__":
    main(dataset="dlkcat")
    main(dataset="eitlem")
