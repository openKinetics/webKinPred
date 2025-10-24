import os
import json
import math
import numpy as np
import pandas as pd
import pickle
import matplotlib.pyplot as plt
import seaborn as sns

from pathlib import Path
from tqdm import tqdm
from sklearn.model_selection import GroupKFold

from smiles_embeddings.smiles_transformer.build_vocab import WordVocab  
from config import RAW_DLKCAT, SEQ_LOOKUP, ROOT
from utils.smiles_features import smiles_to_vec
from model_training import train_model

# Directories for layerwise mean embeddings
ESMC_LAYER_DIR = ROOT / "results/embeddings/esmc_all_layers"
T5_LAYER_DIR   = ROOT / "results/embeddings/prot_t5_all_layers"
ESM2_LAYER_DIR = ROOT / "results/embeddings/esm2_all_layers"

N_LAYERS_T5 = 24
N_LAYERS_ESM2 = 34
N_LAYERS_ESMC = 36

RESULTS_PATH = ROOT / "results/all_layerwise_kcat_km_results.pkl"
PLOT_PATH = ROOT / "results/plots/layerwise_kcat_km_models_comparison.png"


def load_layer_mean(seq_id: str, layer: int, model: str):
    if model == "esmc":
        path = ESMC_LAYER_DIR / f"{seq_id}.npy"
        arr = np.load(path, allow_pickle=True)
        return arr[layer][1]  # mean only
    elif model == "prot_t5":
        path = T5_LAYER_DIR / f"{seq_id}.npy"
        arr = np.load(path)
        return arr[layer]
    elif model == "esm2":
        path = ESM2_LAYER_DIR / f"{seq_id}.npy"
        arr = np.load(path)
        return arr[layer]
    else:
        raise ValueError(f"Unsupported model: {model}")

def extract_features(sequences: list, seq_to_id: dict, layer: int, model: str) -> np.ndarray:
    feats = []
    for seq in sequences:
        seq_id = seq_to_id[seq]
        mean_emb = load_layer_mean(seq_id, layer, model)
        feats.append(mean_emb)
    return np.vstack(feats)

def get_group_kfolds(sequences, groups, n_splits=5):
    splitter = GroupKFold(n_splits=n_splits)
    return list(splitter.split(sequences, groups=groups))

def load_dataset_kcat():
    with RAW_DLKCAT.open("r") as fp:
        raw = json.load(fp)
    raw = [d for d in raw if len(d["Sequence"]) <= 1499 and float(d["Value"]) > 0 and "." not in d["Smiles"]]
    sequences = [d["Sequence"] for d in raw]
    smiles    = [d["Smiles"] for d in raw]
    labels_np = np.array([math.log(float(d["Value"]), 10) for d in raw], dtype=np.float32)
    return sequences, smiles, labels_np

def load_dataset_km():
    with open(ROOT / 'data/KM_data_raw.json', 'r') as fp:
        raw = json.load(fp)
    raw = [d for d in raw if len(d["Sequence"]) <= 1499 and "." not in d["smiles"]]
    sequences = [d["Sequence"] for d in raw]
    smiles    = [d["smiles"] for d in raw]
    labels    = [d["log10_KM"] for d in raw]
    labels_np = np.array(labels, dtype=np.float32)
    return sequences, smiles, labels_np

def run_experiments_for_model(model: str, layer_range: range, sequences: list, smiles: list, labels_np: np.ndarray, seq_id_to_seq: dict, task: str) -> list:
    print(f"\nRunning {task.upper()} for {model.upper()} ...")

    seq_to_id = {v: k for k, v in seq_id_to_seq.items()}
    groups = [seq_to_id[seq] for seq in sequences]
    fold_indices = get_group_kfolds(sequences, groups)
    smiles_vec = smiles_to_vec(smiles, method="smiles_transformer")

    results = []

    for layer in tqdm(layer_range, desc=f"{model.upper()}-{task.upper()}", ncols=150):
        seq_vec = extract_features(sequences, seq_to_id, layer, model)
        for fold, (train_idx, test_idx) in enumerate(fold_indices, 1):
            seq_train, seq_test = seq_vec[train_idx], seq_vec[test_idx]
            smi_train, smi_test = smiles_vec[train_idx], smiles_vec[test_idx]
            y_train, y_test     = labels_np[train_idx], labels_np[test_idx]

            X_train = np.concatenate([smi_train, seq_train], axis=1)
            X_test  = np.concatenate([smi_test, seq_test], axis=1)

            _, _, metrics = train_model(X_train, y_train, X_test, y_test, fold=fold)
            results.append({
                "model": model,
                "task": task,
                "layer": layer,
                "fold": fold,
                **metrics
            })
    return results

def main():
    seq_id_to_seq = pd.read_pickle(SEQ_LOOKUP)

    # Load datasets
    kcat_seqs, kcat_smiles, kcat_labels = load_dataset_kcat()
    km_seqs, km_smiles, km_labels       = load_dataset_km()

    # Layer ranges
    esmc_layers = range(N_LAYERS_ESMC)
    t5_layers   = range(N_LAYERS_T5)
    esm2_layers = range(N_LAYERS_ESM2)

    all_results = []

    for model, layer_range in [
        ("esmc", esmc_layers),
        ("prot_t5", t5_layers),
        ("esm2", esm2_layers)
    ]:
        results_kcat = run_experiments_for_model(model, layer_range, kcat_seqs, kcat_smiles, kcat_labels, seq_id_to_seq, task="kcat")
        results_km   = run_experiments_for_model(model, layer_range, km_seqs, km_smiles, km_labels, seq_id_to_seq, task="km")
        all_results.extend(results_kcat + results_km)

    # Save results
    with open(RESULTS_PATH, "wb") as f:
        pickle.dump(all_results, f)

if __name__ == "__main__":
    main()
