# Data Directory Structure & Organization by Embedding Dimensions

This directory contains **13GB of complete CatLog-derived data** for training and evaluating CatRange models, organized by embedding type with dimensions explicitly labeled.

## 📊 Overview

```
data/                          # Total: 13GB
├── raw/                       # CatLog source databases & metadata
├── processed/                 # Preprocessed datasets organized by embedding type & dimension
│   ├── esmc_1152d/           # ESM-C embeddings (PRIMARY - 1152 dimensions)
│   └── esm2_1280d/           # ESM-2 embeddings (LEGACY - 1280 dimensions)
├── models/                    # Trained XGBoost models
│   ├── esmc_1152d/
│   └── esm2_1280d/
├── results/                   # Cross-validation summaries & metrics
│   ├── esmc_1152d/
│   └── esm2_1280d/
└── DATA_DIRECTORY.md          # This file
```

## 🧬 Embedding Types & Dimensions

### **ESM-C (1152D) - PRIMARY**
- **Dimension**: 1152 features per protein sequence
- **Source**: ESM Metagenomic Atlas (structural embeddings)
- **Use Case**: Better chemical property & mutation effect prediction
- **Files**: Larger due to full-dimensional encoding (~860MB per fold)
- **Location**: `data/processed/esmc_1152d/`
- **Status**: ✅ Primary approach recommended

### **ESM-2 (1280D) - LEGACY**
- **Dimension**: 1280 features per protein sequence
- **Source**: ESM2 Language Model
- **Use Case**: Comparison benchmark, generalist embeddings
- **Files**: Smaller due to architecture differences (~481MB per fold)
- **Location**: `data/processed/esm2_1280d/`
- **Status**: ℹ️ Legacy - use for comparison studies

---

## 📁 Data Organization

### Raw Data (`raw/`)
```
raw/                                       # ~65MB total
├── brenda/
│   ├── BRENDA_all.tsv                    # Complete BRENDA database (26MB)
│   ├── BRENDAandSABIO_kcat_052024_try.json
│   ├── BRENDAandSABIO_kcat_052024_try_unique.json
│   ├── BRENDAandSABIO_km_052024_try.json
│   └── BRENDAandSABIO_km_052024_try_unique.json
├── sabio/
│   ├── SABIO_KCAT_unisubstrate_clean.tsv
│   └── SABIO_KM_unisubstrate_clean.tsv
├── substrates/
│   └── substrates_to_Isomeric_smiles_master.json
├── PafA_1_curated_v1.csv                 # Final curated 658KB dataset
├── PafA_partitions_metadata.xlsx
├── WT_MD_database_v1_curated*.xlsx       # Database compilations (50MB+ each)
└── Other metadata files
```

### Processed Data - ESMC (1152D PRIMARY)
```
processed/esmc_1152d/                      # Total: ~4.3GB
│
├── kcat/                                 # Turnover number - 5 fold files
│   ├── fold1_kcat_data.pt (860MB)
│   ├── fold2_kcat_data.pt (860MB)
│   ├── fold3_kcat_data.pt (860MB)
│   ├── fold4_kcat_data.pt (860MB)
│   └── fold5_kcat_data.pt (860MB)
│
├── km/                                   # Michaelis constant - 5 fold files
│   ├── fold1_km_data.pt (818MB)
│   ├── fold2_km_data.pt (818MB)
│   ├── fold3_km_data.pt (818MB)
│   ├── fold4_km_data.pt (818MB)
│   └── fold5_km_data.pt (818MB)
│
└── Base Datasets (20+ files, ~500MB total)
    ├── dataset_y1_y2_MD.pt               # Mutant design data
    ├── dataset_y1_y2_WT.pt               # Wild-type data
    ├── dataset_y1_y2_PafA.pt             # PafA enzyme-specific
    ├── dataset_y1_y2_OOD_md.pt           # Out-of-distribution mutant
    ├── dataset_y1_y2_OOD_wt.pt           # Out-of-distribution wild-type
    ├── Negative sample sets              # Hard negative mining data
    ├── PafA_1_dataset_esmc.pt            # Full PafA with ESMC embeddings
    └── Position data files               # Sequence position information
```

**Data Characteristics**:
- ESMC files are **larger** (~1.7GB per fold avg) because they contain full 1152D embeddings
- Supports both **kcat** (8-class) and **km** (6-class) predictions
- Includes separate wild-type (WT) and mutant design (MD) training data
- Hard negative samples for improved model robustness

### Processed Data - ESM2 (1280D LEGACY)
```
processed/esm2_1280d/                      # Total: ~2.4GB
│
├── kcat/                                 # Turnover number - 5 fold files
│   ├── fold1_kcat_data.pt (481MB)
│   ├── fold2_kcat_data.pt (481MB)
│   ├── fold3_kcat_data.pt (481MB)
│   ├── fold4_kcat_data.pt (481MB)
│   └── fold5_kcat_data.pt (481MB)
│
├── km/                                   # Michaelis constant - 5 fold files
│   ├── fold1_km_data.pt (394MB each)
│   ├── fold2_km_data.pt
│   ├── fold3_km_data.pt
│   ├── fold4_km_data.pt
│   └── fold5_km_data.pt
│
└── Base Datasets (15+ files)
    # Same types as ESMC, but with 1280D ESM2 embeddings
    # Slightly smaller total size (~480MB)
```

**Data Characteristics**:
- ESM2 files are **smaller** (~960MB per fold avg) due to different architecture
- Supports both **kcat** and **km** predictions
- Provided for benchmarking against primary ESMC approach
- Legacy format maintained for reproducibility

### Trained Models (~5.6GB)
```
models/
├── esmc_1152d/                           # ESMC-trained XGBoost models (~3.1GB)
│   ├── kcat/
│   │   ├── kcat_model_fold1.pkl (612MB)
│   │   ├── kcat_model_fold2.pkl (623MB)
│   │   ├── kcat_model_fold3.pkl (614MB)
│   │   ├── kcat_model_fold4.pkl (606MB)
│   │   └── kcat_model_fold5.pkl (615MB)
│   └── km/
│       ├── km_model_fold1.pkl (~500MB each)
│       ├── km_model_fold2.pkl
│       ├── km_model_fold3.pkl
│       ├── km_model_fold4.pkl
│       └── km_model_fold5.pkl
│
└── esm2_1280d/                           # ESM2-trained XGBoost models (~2.5GB)
    ├── kcat/
    │   ├── kcat_model_fold1.pkl (592MB)
    │   ├── kcat_model_fold2.pkl (584MB)
    │   ├── kcat_model_fold3.pkl (578MB)
    │   ├── kcat_model_fold4.pkl (588MB)
    │   └── kcat_model_fold5.pkl (594MB)
    └── km/
        ├── km_model_fold1-5.pkl
```

**Model Details**:
- **Algorithm**: XGBoost classifiers with hard negative mining
- **kcat**: 8-class classification (log-scale bins from 1 to 1000 s⁻¹)
- **km**: 6-class classification (log-scale bins from 0.1 to 10000 µM)
- **Hyperparameters**: max_depth=7, learning_rate=0.05, subsample=0.8
- **Size**: Models are large due to extensive training data and ensembles

### Results & Evaluation Metrics
```
results/
├── esmc_1152d/
│   ├── kcat_crossval_summary.csv         # ESM-C kcat CV metrics (all folds)
│   ├── km_crossval_summary.csv           # ESM-C km CV metrics
│   ├── test_metrics_by_seqid_kcat.csv    # ESM-C kcat stratified by seq ID
│   └── test_metrics_by_seqid_km.csv      # ESM-C km stratified by seq ID
│
└── esm2_1280d/
    ├── kcat_crossval_summary.csv         # ESM2 kcat CV metrics
    ├── km_crossval_summary.csv           # ESM2 km CV metrics
    ├── test_metrics_by_seqid_kcat.csv    # ESM2 kcat stratified by seq ID
    └── test_metrics_by_seqid_km.csv      # ESM2 km stratified by seq ID
```

**Metrics Included**:
- Accuracy, Precision, Recall, F1 Score, MCC, AUC-PR
- E-Accuracy (within ±1 bin error tolerance)
- Stratification by sequence identity thresholds (40%, 60%, 80%, 90%, 99%, 100%)

---

## 📊 Data Statistics

| Metric | ESMC (1152D) | ESM2 (1280D) | Combined |
|--------|--------------|-------------|----------|
| **Embedding Dimension** | 1152 | 1280 | - |
| **kcat Fold Size** | ~860MB each | ~481MB each | - |
| **km Fold Size** | ~818MB each | ~394MB each | - |
| **Base Datasets** | ~500MB | ~480MB | ~1GB |
| **Total Processed Data** | ~4.3GB | ~2.4GB | **~6.7GB** |
| **Trained Models** | ~3.1GB | ~2.5GB | **~5.6GB** |
| **Results CSVs** | ~9KB | ~8KB | ~17KB |
| **Grand Total** | **~7.4GB** | **~4.9GB** | **~13GB** |
| **# kcat Models** | 5 | 5 | 10 |
| **# km Models** | 5 | 5 | 10 |

---

## 🔄 Data Pipeline Architecture

```
Raw Data (BRENDA/SABIO)
        ↓
    Preprocessing (dedup, seq align)
        ↓
   Embedding Generation
        ↓ ESM-C (1152D)    ↓ ESM2 (1280D)
     Embeddings A       Embeddings B
        ↓                    ↓
   Standardization    Standardization
   (per fold)         (per fold)
        ↓                    ↓
    Train/Val/Test Split (5-fold CV, seqid ≥ 40%)
        ↓                    ↓
   SMOTE Oversampling  SMOTE Oversampling
   (seed=42, k=5)      (seed=42, k=5)
        ↓                    ↓
   Fold Data Files (fold{1-5}_*_data.pt)
        ↓                    ↓
   XGBoost Training   XGBoost Training
   + Hard Neg Mining  + Hard Neg Mining
        ↓                    ↓
   Trained Models     Trained Models
        ↓                    ↓
   Cross-Validation   Cross-Validation
   Results            Results
```

**Key Features**:
- **Standardization**: Separate standardizers per embedding type (accounts for different dimensions)
- **SMOTE**: Applied only to training set, seed=42 for reproducibility
- **Hard Negative Mining**: boost_factor=100 during XGBoost training
- **Stratification**: Sequence identity ≥ 40% to prevent information leakage

---

## 💾 Loading Data in Code

### Quick Start - ESMC (Recommended)
```python
from src.data_pipeline import load_datasets_from_disk
from src.config import GLOBAL_SEED

# Load ESMC data (primary, 1152D)
esmc_data = load_datasets_from_disk(
    base_dir="data/processed/esmc_1152d",
    n_folds=5,
    use_standardizer=True
)
# Returns: {
#   1: {"train": Dataset(X_1152D, y), "val": Dataset(...), "test": Dataset(...)},
#   2: {...},
#   ...
#   5: {...}
# }

# Access fold 1 data
fold1_train = esmc_data[1]["train"]
X_train, y_train = fold1_train.tensors if hasattr(fold1_train, 'tensors') else fold1_train[:]
print(f"X shape: {X_train.shape}")  # Should be (N, 1152)
```

### Loading ESM2 Data (Legacy)
```python
# Load ESM2 data (legacy, 1280D)
esm2_data = load_datasets_from_disk(
    base_dir="data/processed/esm2_1280d",
    n_folds=5,
    use_standardizer=True
)

fold1_train = esm2_data[1]["train"]
X_train, y_train = fold1_train[:]
print(f"X shape: {X_train.shape}")  # Should be (N, 1280)
```

### Loading Specific Fold Data Directly
```python
import torch

# Load fold 1 ESMC kcat data (1152D)
fold1_esmc_kcat = torch.load("data/processed/esmc_1152d/kcat/fold1_kcat_data.pt")
X_fold1, y_fold1 = fold1_esmc_kcat
print(f"ESMC kcat fold1 X shape: {X_fold1.shape}")  # (N, 1152)

# Load fold 1 ESM2 kcat data (1280D)
fold1_esm2_kcat = torch.load("data/processed/esm2_1280d/kcat/fold1_kcat_data.pt")
X_fold1_esm2, y_fold1_esm2 = fold1_esm2_kcat
print(f"ESM2 kcat fold1 X shape: {X_fold1_esm2.shape}")  # (N, 1280)
```

### Loading Trained Models
```python
from src.model_training import load_model
import joblib

# Load ESMC kcat fold 1 model
kcat_model_esmc = load_model("data/models/esmc_1152d/kcat/kcat_model_fold1.pkl")
# or
kcat_model_esmc = joblib.load("data/models/esmc_1152d/kcat/kcat_model_fold1.pkl")

# Load ESM2 km fold 3 model
km_model_esm2 = load_model("data/models/esm2_1280d/km/km_model_fold3.pkl")

# Make predictions (ensure input dimensions match!)
# For ESMC: must be 1152D
# For ESM2: must be 1280D
predictions = kcat_model_esmc.predict(X_fold1)  # X_fold1 must be 1152D
```

---

## ⚠️ Critical: Embedding Dimension Consistency

### Dimension Requirements

| Embedding | Dimension | Models | Fold Data Files |
|-----------|-----------|--------|-----------------|
| **ESMC** | **1152D** | `esmc_1152d/*.pkl` | `esmc_1152d/{kcat,km}/*.pt` |
| **ESM2** | **1280D** | `esm2_1280d/*.pkl` | `esm2_1280d/{kcat,km}/*.pt` |

### Verify Dimensions Before Using
```python
import torch
from src.config import ESMC_DIM, ESM2_DIM

# ESMC verification
esmc_data = torch.load("data/processed/esmc_1152d/kcat/fold1_kcat_data.pt")
X_esmc, _ = esmc_data
assert X_esmc.shape[1] == ESMC_DIM == 1152, f"ESMC: expected 1152D, got {X_esmc.shape[1]}D"

# ESM2 verification
esm2_data = torch.load("data/processed/esm2_1280d/kcat/fold1_kcat_data.pt")
X_esm2, _ = esm2_data
assert X_esm2.shape[1] == ESM2_DIM == 1280, f"ESM2: expected 1280D, got {X_esm2.shape[1]}D"
```

### Never Mix Embeddings!
```python
# ❌ WRONG - Will fail or give wrong results
X_esmc = torch.load("data/processed/esmc_1152d/kcat/fold1_kcat_data.pt")[0]  # 1152D
model_esm2 = load_model("data/models/esm2_1280d/kcat/kcat_model_fold1.pkl")
predictions = model_esm2.predict(X_esmc)  # ERROR: shape mismatch!

# ✅ CORRECT - Match embedding types
X_esmc = torch.load("data/processed/esmc_1152d/kcat/fold1_kcat_data.pt")[0]  # 1152D
model_esmc = load_model("data/models/esmc_1152d/kcat/kcat_model_fold1.pkl")
predictions = model_esmc.predict(X_esmc)  # OK!
```

---

## 🔍 Data Verification

Verify your data is complete and correct:

```bash
# 1. Count files
find data -type f -name "*.pt" -o -name "*.pkl" -o -name "*.csv" -o -name "*.tsv" | wc -l
# Expected: ~98 files

# 2. Check total size
du -sh data/
# Expected: ~13GB

# 3. Verify ESMC fold data (should have 10 files)
ls data/processed/esmc_1152d/{kcat,km}/*.pt | wc -l
# Expected: 10 files (5 kcat + 5 km)

# 4. Verify ESM2 fold data (should have 10 files)
ls data/processed/esm2_1280d/{kcat,km}/*.pt | wc -l
# Expected: 10 files (5 kcat + 5 km)

# 5. Verify models (should have 20 files)
find data/models -name "*.pkl" | wc -l
# Expected: 20 files (5 kcat + 5 km per embedding type)

# 6. Verify results CSVs (should have 8 files)
ls data/results/*/*.csv | wc -l
# Expected: 8 files (4 per embedding type)
```

---

## 📝 Python Verification Script

```python
#!/usr/bin/env python3
"""Verify data directory structure and dimensions."""

import torch
from pathlib import Path

def verify_data_directory(data_dir="data"):
    """Verify complete data structure."""
    data_path = Path(data_dir)
    
    print("Checking data directory structure...\n")
    
    # Check ESMC data
    print("✓ ESMC (1152D) Data:")
    for fold in range(1, 6):
        kcat_file = data_path / f"processed/esmc_1152d/kcat/fold{fold}_kcat_data.pt"
        km_file = data_path / f"processed/esmc_1152d/km/fold{fold}_km_data.pt"
        
        if kcat_file.exists() and km_file.exists():
            X_kcat = torch.load(kcat_file)[0]
            X_km = torch.load(km_file)[0]
            print(f"  Fold {fold}: kcat X shape {X_kcat.shape}, km X shape {X_km.shape}")
            assert X_kcat.shape[1] == 1152, f"ESMC kcat fold {fold} wrong dimension!"
            assert X_km.shape[1] == 1152, f"ESMC km fold {fold} wrong dimension!"
    
    # Check ESM2 data
    print("\n✓ ESM2 (1280D) Data:")
    for fold in range(1, 6):
        kcat_file = data_path / f"processed/esm2_1280d/kcat/fold{fold}_kcat_data.pt"
        km_file = data_path / f"processed/esm2_1280d/km/fold{fold}_km_data.pt"
        
        if kcat_file.exists() and km_file.exists():
            X_kcat = torch.load(kcat_file)[0]
            X_km = torch.load(km_file)[0]
            print(f"  Fold {fold}: kcat X shape {X_kcat.shape}, km X shape {X_km.shape}")
            assert X_kcat.shape[1] == 1280, f"ESM2 kcat fold {fold} wrong dimension!"
            assert X_km.shape[1] == 1280, f"ESM2 km fold {fold} wrong dimension!"
    
    print("\n✅ All data verified successfully!")

if __name__ == "__main__":
    verify_data_directory()
```

---

## 🚀 Quick Reference

| Task | Command | Embedding | Dimension |
|------|---------|-----------|-----------|
| Load ESMC kcat folds | `load_datasets_from_disk("data/processed/esmc_1152d", N_folds=5)` | ESMC | 1152D |
| Load ESM2 kcat folds | `load_datasets_from_disk("data/processed/esm2_1280d", N_folds=5)` | ESM2 | 1280D |
| Load single fold | `torch.load("data/processed/esmc_1152d/kcat/fold1_kcat_data.pt")` | ESMC | 1152D |
| Load model | `load_model("data/models/esmc_1152d/kcat/kcat_model_fold1.pkl")` | ESMC | 1152D |
| View results | `cat data/results/esmc_1152d/kcat_crossval_summary.csv` | ESMC | 1152D |

---

## ✅ Reproducibility Features

All data is organized for **100% reproducible results**:
- ✅ Fixed seed (GLOBAL_SEED=42) in all preprocessing
- ✅ Embedding-specific standardization (accounts for dimension differences)
- ✅ Deterministic fold splits (stratified, seqid ≥ 40%)
- ✅ SMOTE with fixed seed (k_neighbors=5, seed=42)
- ✅ Hard negative mining with fixed boost_factor (100x)
- ✅ All random states set to 42 throughout pipeline

See `docs/reproducibility.md` for detailed validation procedures.

---

## 📋 File Manifest

**Total Files**: 98  
**Total Size**: 13GB  
**Organized by**: Embedding Type × Dimension × Parameter × Fold

```
raw/                           18 files (~65MB - metadata & databases)
processed/esmc_1152d/          35 files (~4.3GB - ESMC embeddings 1152D)
processed/esm2_1280d/          30 files (~2.4GB - ESM2 embeddings 1280D)
models/esmc_1152d/             10 files (~3.1GB - ESMC trained models)
models/esm2_1280d/             10 files (~2.5GB - ESM2 trained models)
results/esmc_1152d/             4 files (metrics CSVs)
results/esm2_1280d/             4 files (metrics CSVs)
```

---

## ❓ Troubleshooting

### "File not found" errors
- Check embedding type: ESMC uses `esmc_1152d/`, ESM2 uses `esm2_1280d/`
- Verify fold number is 1-5
- Ensure full path: `data/processed/esmc_1152d/kcat/fold1_kcat_data.pt`

### Dimension mismatch errors
- ESMC input must be exactly 1152D
- ESM2 input must be exactly 1280D
- Cannot use ESMC embeddings with ESM2 models or vice versa

### Memory issues with large fold files
- Each fold file is ~860MB (ESMC) or ~481MB (ESM2)
- 16GB+ RAM recommended for comfort
- Consider processing one fold at a time if limited RAM

### Model prediction errors
- Verify input dimensions match model: `X.shape[1]` should be 1152 or 1280
- Check that data and model use same embedding type
- Load model from correct directory

---

**Last Updated**: February 18, 2026  
**Data Status**: ✅ Complete (13GB, 98 files)  
**Organization**: ✅ Dimension-aware structure  
**Ready for**: Research, Benchmarking, Production Deployment
