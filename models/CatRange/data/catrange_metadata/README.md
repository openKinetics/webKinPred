# Data Directory Structure

This directory contains the CatLog datasets and embeddings used for CatRange training, validation, and testing.

## Directory Layout

```
data/
├── raw/                          # Original raw data files
│   ├── BRENDA_all.tsv           # BRENDA kinetic parameters
│   └── substrates.json          # Substrate SMILES mappings
│
├── processed/                    # Preprocessed data and embeddings
│   ├── embeddings/
│   │   ├── esm_c/               # ESM-C embeddings (1152D)
│   │   │   ├── fold_1/
│   │   │   ├── fold_2/
│   │   │   ├── fold_3/
│   │   │   ├── fold_4/
│   │   │   └── fold_5/
│   │   └── esm2/                # ESM-2 embeddings (1280D)
│   │       ├── fold_1/
│   │       ├── fold_2/
│   │       ├── fold_3/
│   │       ├── fold_4/
│   │       └── fold_5/
│   │
│   └── metadata/
│       ├── fold_splits.pkl      # Train/val/test fold assignments
│       ├── class_bins_kcat.pkl  # kcat binning specification
│       └── class_bins_km.pkl    # Km binning specification
│
├── splits/                       # Split datasets ready for training
│   ├── kcat_esmc/               # kcat with ESM-C
│   │   ├── fold_1/
│   │   │   ├── train_data.pt    # Training tensors
│   │   │   ├── val_data.pt      # Validation tensors
│   │   │   └── test_data.pt     # Test tensors
│   │   ...
│   │
│   ├── km_esmc/                 # km with ESM-C
│   │   └── (same structure)
│   │
│   ├── kcat_esm2/               # kcat with ESM-2
│   │   └── (same structure)
│   │
│   └── km_esm2/                 # km with ESM-2
│       └── (same structure)
│
└── README.md                     # This file
```

## File Descriptions

### raw/
- `BRENDA_all.tsv`: Source enzyme kinetic parameter export incorporated into CatLog
  - Columns: Enzyme, Substrate, kcat, Km, organism, conditions, etc.
  - Format: Tab-separated values
  
- `substrates.json`: Mapping of substrate names to SMILES strings
  - Format: JSON dictionary

### processed/embeddings/
Contains pre-computed protein sequence embeddings:
- `esm_c/`: ESM-C embeddings (1152 dimensions)
  - Superior structural information
  - Primary approach (recommended)
  
- `esm2/`: ESM-2 embeddings (1280 dimensions)
  - Language model embeddings
  - Alternative/legacy approach

### splits/
Cross-validation folds ready for training:
- Each fold contains 80% training / 10% validation / 10% test
- Data is pre-split and pre-standardized
- File format: PyTorch binary (.pt)

## Data Preparation

### Step 1: Download Raw Data
```bash
# Download BRENDA dump
# 1. Visit https://brenda-enzyme.org/
# 2. Request full database export
# 3. Save as data/raw/BRENDA_all.tsv

# Download or create substrate mapping
# Save as data/raw/substrates.json
```

### Step 2: Generate Embeddings
```bash
# Generate ESM-C embeddings
python scripts/embeddings/generate_esmc.py \
    --data data/raw/BRENDA_all.tsv \
    --output data/processed/embeddings/esm_c/

# Generate ESM-2 embeddings  
python scripts/embeddings/generate_esm2.py \
    --data data/raw/BRENDA_all.tsv \
    --output data/processed/embeddings/esm2/
```

### Step 3: Create Train/Val/Test Splits
```bash
# Generate fold splits (you can use data_robust_v1 structure as reference)
python scripts/data/create_splits.py \
    --embeddings data/processed/embeddings/esm_c/ \
    --output data/splits/kcat_esmc/ \
    --n_folds 5 \
    --parameter kcat
```

## Using Existing Data

If you already have the `data_robust_v1` directory from the original project:

```bash
# Create symlink or copy
ln -s /path/to/data_robust_v1 data/

# Or copy specific folders
cp -r /path/to/data_robust_v1/data_split_curated_no_OOD_nodups_esmc data/splits/kcat_esmc/
cp -r /path/to/data_robust_v1/data_split_curated_no_OOD_nodups data/splits/kcat_esm2/
```

## Data Format

### Embedding Files (.pt)
PyTorch tensor files containing:
- **ESM-C**: Concatenated [ESM-C (1152D) + auxiliary (varies)]
- **ESM-2**: Concatenated [ESM-2 (1280D) + auxiliary (varies)]

### Label Files
- **kcat**: 8 classes (bins based on log scale)
- **Km**: 6 classes (bins based on log scale)

### Train/Val/Test Split Ratio
- Training: 60% (for 5-fold CV: 4 folds × 60% each split = 80% of data)
- Validation: 20% (1 fold × 20% per split)
- Test: 20% (1 fold × 20% per split)

## Reproducibility

To ensure reproducibility:
1. Use the **exact same fold assignments**
2. Use **pre-computed standardization statistics** from training set
3. Apply SMOTE only to training set
4. Use **fixed seed** (GLOBAL_SEED=42) for data shuffling

See [QUICKSTART.md](../QUICKSTART.md) and [docs/reproducibility.md](../docs/reproducibility.md) for details.

## Troubleshooting

### Missing Data Files
If you get "File not found" errors:
1. Verify data directory structure matches above
2. Check that fold directories exist (fold_1, fold_2, etc.)
3. Confirm .pt files are present

### Dimension Mismatch
If you get shape errors:
- **ESM-C**: Expect first 1152 dimensions
- **ESM-2**: Expect first 1280 dimensions
- Check you're using the right embedding type

### Data Not Loading
Check requirements:
1. PyTorch installed (`pip install torch`)
2. PyTorch version compatible with saved tensors
3. Sufficient disk space (embeddings ~5-10GB per fold)

## References

- BRENDA: https://www.brenda-enzyme.org/
- ESM-2: https://github.com/facebookresearch/esm
- ESM-C: https://github.com/meta-ai/protein-language-models

---

**Data Directory Version**: 1.0  
**Last Updated**: February 17, 2026
