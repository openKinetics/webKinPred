"""
Data Pipeline Module for CatRange

Handles:
- Dataset loading and preparation
- Global standardization (separate for dual embeddings)
- SMOTE oversampling for imbalanced data
- Negative hybrid relabelling (mutation-aware)
- Holdout set creation
"""

import torch
import numpy as np
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
from imblearn.over_sampling import SMOTE
from typing import Tuple

try:
    from src.utils import safe_load, check_device, tensor_to_numpy
except ImportError:
    from utils import safe_load, check_device, tensor_to_numpy

# ── Local constants (no config.py dependency) ────────────────────────────────
_GLOBAL_SEED = 42
_CLAMP_MIN = 1e-7
_DATA_ROOT = Path("data_robust_v1")
_DATASET_ESMC = "data_split_curated_no_OOD_nodups_esmc"
_DATASET_ESM2 = "data_split_curated_no_OOD_nodups"


# ============================================================================
# CUSTOM DATASET CLASS
# ============================================================================
class TensorDatasetSimple(Dataset):
    """Simple dataset class for handling (features, labels) pairs."""

    def __init__(self, data: torch.Tensor, labels: torch.Tensor):
        self.data = data
        self.labels = labels
        assert len(data) == len(labels), "Data and labels must have same length"

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.data[idx], self.labels[idx]

    def get_labels(self) -> torch.Tensor:
        return self.labels


# ============================================================================
# TENSOR EXTRACTION UTILITY
# ============================================================================
def extract_tensors(dataset: Dataset) -> Tuple[torch.Tensor, torch.Tensor]:
    """Extract full (X, y) tensors from any Dataset via DataLoader."""
    loader = DataLoader(dataset, batch_size=len(dataset), shuffle=False)
    X, y = next(iter(loader))
    return X, y


# ============================================================================
# STANDARDIZATION
# ============================================================================
def get_train_stats_separate(
    train_data: torch.Tensor, split_dim: int = 1152
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute global mean and std for each embedding part from training data."""
    X1 = train_data[:, :split_dim]
    X2 = train_data[:, split_dim:]
    return X1.mean(), X1.std(), X2.mean(), X2.std()


def standardize_array(X: np.ndarray, mean_1, std_1, mean_2, std_2,
                      split_dim: int = 1152) -> np.ndarray:
    """Standardize numpy array using precomputed stats (for XGBoost input)."""
    std_1 = max(float(std_1), _CLAMP_MIN)
    std_2 = max(float(std_2), _CLAMP_MIN)
    X1 = (X[:, :split_dim] - float(mean_1)) / std_1
    X2 = (X[:, split_dim:] - float(mean_2)) / std_2
    return np.concatenate([X1, X2], axis=1)


# ============================================================================
# NEGATIVE HYBRID RELABELLING (Publication logic)
# ============================================================================
def relabel_negatives_hybrid(
    wt_labels: torch.Tensor,
    drops: torch.Tensor,
    kparam: str = "kcat",
    seed: int = _GLOBAL_SEED,
) -> torch.Tensor:
    """
    Relabel negative samples based on WT kcat/km bin and mutation drop severity.

    For kcat: drop=1 -> lower bins [0, wt-2]; drop=0 -> nearby [wt-1, wt+1]
    For km:   drop=1 -> higher bins [wt+1, 5]; drop=0 -> nearby [0, wt]
    """
    rng = np.random.default_rng(seed)
    new_labels = []
    for wt_bin, drop in zip(wt_labels.cpu().numpy(), drops.cpu().numpy()):
        wt_bin = int(wt_bin)
        drop = int(drop)
        if kparam == "kcat":
            if drop == 1:
                upper = max(0, wt_bin - 2)
                valid = list(range(0, upper + 1))
            else:
                lower = max(0, wt_bin - 1)
                upper = min(5, wt_bin + 1)
                valid = list(range(lower, upper + 1))
        elif kparam == "km":
            if drop == 1:
                lower = min(5, wt_bin + 1)
                valid = list(range(lower, 6))
            else:
                upper = min(5, wt_bin)
                valid = list(range(0, upper + 1))
        else:
            raise ValueError("kparam must be 'kcat' or 'km'")
        if not valid:
            valid = [wt_bin]
        new_labels.append(rng.choice(valid))
    return torch.tensor(new_labels, dtype=torch.long)


# ============================================================================
# SMOTE OVERSAMPLING
# ============================================================================
def apply_smote(
    train_data: torch.Tensor,
    train_labels: torch.Tensor,
    device: torch.device,
    seed: int = _GLOBAL_SEED,
    k_neighbors: int = 5,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Apply SMOTE oversampling to address class imbalance."""
    X_np = train_data.cpu().numpy()
    y_np = train_labels.cpu().numpy()

    smote = SMOTE(random_state=seed, k_neighbors=k_neighbors)
    X_res, y_res = smote.fit_resample(X_np, y_np)

    X_res_t = torch.tensor(X_res, dtype=torch.float32).to(device)
    y_res_t = torch.tensor(y_res, dtype=torch.int64).to(device)

    print(f"[SMOTE] {len(X_np)} -> {len(X_res)} samples")
    return X_res_t, y_res_t


# ============================================================================
# HOLDOUT CREATION (Before CV)
# ============================================================================
def create_holdout_set_proportional(
    X_wt, y_wt, X_md, y_md, neg_test_ds=None,
    n_hold=100, n_neg_hold=10, device="cpu", seed=_GLOBAL_SEED
):
    """
    Create a proportional holdout set from WT and MD pools before CV.

    Returns:
        (holdout_dict, updated_wt, updated_md, updated_neg, holdout_indices)
    """
    torch.manual_seed(seed)
    total = len(X_wt) + len(X_md)
    n_wt = round(len(X_wt) / total * n_hold)
    n_md = n_hold - n_wt

    idx_wt = torch.randperm(len(X_wt))[:n_wt]
    X_wt_hold, y_wt_hold = X_wt[idx_wt], y_wt[idx_wt]
    mask_wt = torch.ones(len(X_wt), dtype=torch.bool); mask_wt[idx_wt] = False
    X_wt_rest, y_wt_rest = X_wt[mask_wt], y_wt[mask_wt]

    idx_md = torch.randperm(len(X_md))[:n_md]
    X_md_hold, y_md_hold = X_md[idx_md], y_md[idx_md]
    mask_md = torch.ones(len(X_md), dtype=torch.bool); mask_md[idx_md] = False
    X_md_rest, y_md_rest = X_md[mask_md], y_md[mask_md]

    if neg_test_ds is not None:
        X_neg_all, y_neg_all = extract_tensors(neg_test_ds)
        neg_indices = torch.randperm(len(X_neg_all))[:n_neg_hold]
        X_neg_hold, y_neg_hold = X_neg_all[neg_indices], y_neg_all[neg_indices]
        mask_neg = torch.ones(len(X_neg_all), dtype=torch.bool); mask_neg[neg_indices] = False
        updated_neg = TensorDatasetSimple(X_neg_all[mask_neg].to(device), y_neg_all[mask_neg].to(device))
    else:
        X_neg_hold = torch.empty((0, X_wt.shape[1]))
        y_neg_hold = torch.empty((0,), dtype=torch.long)
        updated_neg = None
        neg_indices = torch.tensor([], dtype=torch.long)

    holdout_dict = {
        "X": torch.cat([X_wt_hold, X_md_hold]).to(device),
        "y": torch.cat([y_wt_hold, y_md_hold]).to(device),
        "X_neg": X_neg_hold.to(device) if len(X_neg_hold) > 0 else None,
        "y_neg": y_neg_hold.to(device) if len(y_neg_hold) > 0 else None,
        "n_wt": n_wt, "n_md": n_md,
    }
    updated_wt = (X_wt_rest.to(device), y_wt_rest.to(device))
    updated_md = (X_md_rest.to(device), y_md_rest.to(device))
    indices = {"wt": idx_wt.cpu().numpy(), "md": idx_md.cpu().numpy(),
               "neg": neg_indices.cpu().numpy()}

    print(f"[Holdout] Created: {n_wt} WT + {n_md} MD = {n_hold} pos, {len(neg_indices)} neg")
    return holdout_dict, updated_wt, updated_md, updated_neg, indices


# ============================================================================
# COMPREHENSIVE DATA LOADING
# ============================================================================
def load_all_data(embedding_type="esmc", param_type="kcat", device=None):
    """
    Load all datasets needed for the full training pipeline.

    Returns dict with: wt, md, neg_train, neg_val, neg_test,
    neg_mdseq_wtvals, neg_wtseq_wtvals, pafa_train/val/test, ood_wt/md
    """
    if device is None:
        device = check_device()

    if embedding_type.lower() == "esmc":
        data_dir = Path(_DATA_ROOT / _DATASET_ESMC)
    else:
        data_dir = Path(_DATA_ROOT / _DATASET_ESM2)

    label_idx = 1 if param_type.lower() == "kcat" else 2

    def _load(filename, lidx=label_idx):
        d = safe_load(data_dir / filename, device)
        return d[0].to(device), d[lidx].to(device)

    datasets = {}

    print(f"[Data] Loading {embedding_type.upper()} data for {param_type}...")
    X_wt, y_wt = _load("dataset_y1_y2_WT.pt")
    X_md, y_md = _load("dataset_y1_y2_MD.pt")
    datasets["wt"] = (X_wt, y_wt)
    datasets["md"] = (X_md, y_md)
    print(f"  WT: {X_wt.shape}, MD: {X_md.shape}")

    # Negative splits (use y1 = drop binary labels always)
    for split in ["train", "val", "test"]:
        X, y = _load(f"neg_{split}_dataset_y1_y2_md_drops.pt", lidx=1)
        datasets[f"neg_{split}"] = (X, y)
    print(f"  Neg: train={datasets['neg_train'][0].shape[0]}, "
          f"val={datasets['neg_val'][0].shape[0]}, test={datasets['neg_test'][0].shape[0]}")

    # Negative with WT values (for relabelling)
    X_mdwt, y_mdwt = _load("dataset_y1_y2_Negative_mdseq_wtVals.pt")
    datasets["neg_mdseq_wtvals"] = (X_mdwt, y_mdwt)
    X_wtwt, y_wtwt = _load("dataset_y1_y2_Negative_wtseq_wtVals.pt")
    datasets["neg_wtseq_wtvals"] = (X_wtwt, y_wtwt)

    # Full negative pool with drop labels
    X_nf, y_nf = _load("dataset_y1_y2_Negative_md_drops.pt", lidx=1)
    datasets["neg_full_drops"] = (X_nf, y_nf)
    datasets["neg_full_wtvals"] = (X_mdwt, y_mdwt)

    # PafA data
    pafa_available = True
    for key, fname in [("pafa_train", "PafA_1_train_dataset_2.pt"),
                       ("pafa_val", "PafA_1_val_dataset_2.pt"),
                       ("pafa_test", "PafA_1_test_dataset_2.pt")]:
        fpath = data_dir / fname
        if fpath.exists():
            datasets[key] = _load(fname)
        else:
            pafa_available = False

    if not pafa_available:
        pafa_path = data_dir / "dataset_y1_y2_PafA.pt"
        if pafa_path.exists():
            from sklearn.model_selection import train_test_split
            d = safe_load(pafa_path, device)
            X_pafa, y_pafa = d[0].to(device), d[label_idx].to(device)
            n = len(X_pafa)
            idx = np.arange(n)
            idx_tr, idx_rest = train_test_split(idx, test_size=0.46, random_state=_GLOBAL_SEED,
                                                 stratify=y_pafa.cpu().numpy())
            idx_va, idx_te = train_test_split(idx_rest, test_size=0.33, random_state=_GLOBAL_SEED,
                                               stratify=y_pafa[idx_rest].cpu().numpy())
            datasets["pafa_train"] = (X_pafa[idx_tr], y_pafa[idx_tr])
            datasets["pafa_val"] = (X_pafa[idx_va], y_pafa[idx_va])
            datasets["pafa_test"] = (X_pafa[idx_te], y_pafa[idx_te])
            print(f"  PafA split: train={len(idx_tr)}, val={len(idx_va)}, test={len(idx_te)}")
    else:
        print(f"  PafA: train={datasets['pafa_train'][0].shape[0]}, "
              f"val={datasets['pafa_val'][0].shape[0]}, test={datasets['pafa_test'][0].shape[0]}")

    for name, fname in [("ood_wt", "dataset_y1_y2_OOD_wt.pt"),
                        ("ood_md", "dataset_y1_y2_OOD_md.pt")]:
        fpath = data_dir / fname
        if fpath.exists():
            datasets[name] = _load(fname)
    if "ood_wt" in datasets:
        print(f"  OOD: wt={datasets['ood_wt'][0].shape[0]}, md={datasets['ood_md'][0].shape[0]}")

    datasets["data_dir"] = data_dir
    datasets["label_idx"] = label_idx
    datasets["embedding_type"] = embedding_type
    return datasets
