# CatRange: Robust Prediction of Enzyme Variant Kinetic Ranges

[![Python 3.9+](https://img.shields.io/badge/Python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.0+](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)
[![XGBoost 2.0+](https://img.shields.io/badge/XGBoost-2.0+-green.svg)](https://xgboost.readthedocs.io/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Status: Reproducible](https://img.shields.io/badge/Status-Reproducible-brightgreen.svg)](#reproducibility)

**CatRange** is a reproducible machine-learning pipeline for predicting enzyme kinetic parameters — k<sub>cat</sub> (catalytic constant) and K<sub>M</sub> (Michaelis constant) — from curated enzyme–substrate data. It combines protein-language-model embeddings with XGBoost classification to predict **log-scale kinetic ranges** rather than brittle point estimates.

The curated enzyme-kinetics database supporting CatRange is now referred to as **CatLog**. CatLog is intended to be periodically refreshed by an agentic curation pipeline so benchmark folds, metadata, and future releases can trace back to a living data resource.

---

## Quick Inference (Colab)

For interactive inference from **raw enzyme sequence + Isomeric SMILES** — no local install required:

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ssbio/CatRange/blob/main/CatRange_Inference_Interface.ipynb)

The CatRange inference notebook handles embedding generation, model loading, and prediction end-to-end. Simply connect to a Colab runtime and follow the instructions.


---

## Table of Contents

1. [Why Range/Class Prediction?](#why-rangeclass-prediction)
2. [Key Results](#key-results)
3. [Quick Inference (Colab)](#quick-inference-colab)
4. [Installation](#installation)
5. [Training](#training)
6. [Local Inference](#local-inference)
7. [Input Constraints](#input-constraints)
8. [Approach & Parameter Selection](#approach--parameter-selection)
9. [Repository Structure](#repository-structure)
10. [Reproducibility](#reproducibility)
11. [Citation](#citation)
12. [License](#license)
13. [FAQ](#faq)

---

## Why Range/Class Prediction?

Enzyme kinetics are intrinsically heterogeneous: k<sub>cat</sub> and K<sub>M</sub> can vary by orders of magnitude across family members and assay contexts. Experimental labels are condition-dependent (assay setup, cofactors, substrate analogs, mutation context), so exact scalar targets are often noisier than their mechanistic range.

CatRange v1.2.0 therefore predicts **log-bin classes** as robust kinetic ranges:

| Parameter | Classes | Rationale |
|-----------|---------|-----------|
| k<sub>cat</sub> | 8 | Captures turnover number across enzymatic regimes |
| K<sub>M</sub> | 6 | Captures substrate-affinity ranges |

This preserves biochemical interpretability, improves robustness under distribution shift, and better matches practical decision-making for enzyme screening and prioritization.

---


## Installation

### Prerequisites

- Python 3.9–3.11
- pip or conda
- (Optional) CUDA-capable GPU for faster training

### 1. Clone & enter the repository

```bash
git clone https://github.com/TKAI-LAB-Mali/CatRange.git
cd CatRange
```

### 2. Create the environment

**CPU only:**
```bash
conda env create --name CatRange_env_cpu -f CatRange_env_cpu.yml
conda activate CatRange_env_cpu
pip install -r requirements_cpu.txt
```

**GPU (CUDA 11.8+):**
```bash
conda env create --name CatRange_env_gpu -f CatRange_env_gpu.yml
conda activate CatRange_env_gpu
pip install -r requirements_gpu.txt
```

### 3. Install package entry points

```bash
python3 -m pip install --no-deps -e .
```

This registers CatRange CLI commands (defined in `pyproject.toml`):
- `catrange-train` — cross-validation training
- `catrange-predict` — inference from precomputed embeddings

The legacy `realkcat-train` and `realkcat-predict` aliases are also retained so existing environments and scripts keep working during the rename.

Using `--no-deps` avoids re-resolving the environment after the conda + requirements setup above. This is important because upgrading to NumPy 2.x can break compiled packages such as `pyarrow`, `scikit-learn`, and `xgboost`.

### 4. Verify

```bash
python -c "from src.config import GLOBAL_SEED; print(f'Setup OK  seed={GLOBAL_SEED}')"
catrange-train --help
catrange-predict --help
```

If you already see NumPy 2.x / `pyarrow` / `_ARRAY_API` errors, repair the environment with:

```bash
python3 -m pip install "numpy<2"
python3 -m pip install --no-deps -e .
```
---

## Training

Each experiment is defined by a YAML config in `configs/`. All hyperparameters, data paths, and pipeline settings live in the config file — no code changes needed.

| Config | Parameter | Embedding | Role |
|--------|-----------|-----------|------|
| `kcat_esmc.yaml` | k<sub>cat</sub> | ESM-C (1152D) | **Primary** |
| `km_esmc.yaml` | K<sub>M</sub> | ESM-C (1152D) | **Primary** |
| `kcat_esm2.yaml` | k<sub>cat</sub> | ESM-2 (1280D) | Alternative |
| `km_esm2.yaml` | K<sub>M</sub> | ESM-2 (1280D) | Alternative |

### Run training

```bash
# Recommended (after pip install -e .)
catrange-train --config configs/kcat_esmc.yaml --device cuda

# Or from source tree without install
PYTHONPATH=. python scripts/cv_train.py --config configs/kcat_esmc.yaml --device cuda
```

Outputs (models, metrics, holdout indices) are saved to `outputs/<experiment>/`.

### XGBoost hyperparameters (Optuna-tuned)

The YAML configs contain Optuna-tuned hyperparameters that should not be changed without re-tuning:

| Hyperparameter | Value |
|----------------|-------|
| `n_estimators` | 1920 |
| `max_depth` | 11 |
| `learning_rate` | 0.00211 |
| `subsample` | 1.0 |
| `colsample_bytree` | 0.460 |
| `reg_alpha` | 1.118 |
| `reg_lambda` | 14.803 |

Hard-negative mining with boost factor 100× and 20% fine-tune fraction is applied after the initial training pass.

---

## Local Inference

Local `catrange-predict` operates on **precomputed concatenated embeddings** (enzyme + substrate, `.pt` tensor files). It does **not** generate embeddings from raw sequence/SMILES — use the [Colab notebook](#quick-inference-colab) for that.

```bash
catrange-predict \
  --model outputs/kcat_esmc/models/kcat_esmc_fold5.pkl \
  --features path/to/concatenated_embeddings.pt \
  --output predictions.csv
```

---

## Input Constraints

| Constraint | Value |
|------------|-------|
| Minimum sequence length | 9 |
| Maximum sequence length | 1022 |
| Maximum Isomeric SMILES length | 512 |

---

## Approach & Parameter Selection

### Embedding

| Embedding | Dimensionality | Strengths |
|-----------|----------------|-----------|
| **ESM-C** (recommended) | 1152D | Structural encoding; +1–2% accuracy over ESM-2 |
| ESM-2 (alternative) | 1280D | Language-model baseline; good for benchmarking |

### Versioning

| Version | Mode | Notes |
|---------|------|-------|
| **v1.2.0** (default) | Mechanistic mutation-aware | Graded mutation effects; recommended |
| v1.1.0 / v1.0.0 | Binary alanine | Over-simplified; benchmarking only |

---

## Repository Structure

```
CatRange/
├── README.md
├── pyproject.toml                  # Package metadata & CLI entry points
├── setup.py
├── requirements.txt
├── requirements_cpu.txt
├── requirements_gpu.txt
├── CatRange_env_cpu.yml            # Conda environment (CPU)
├── CatRange_env_gpu.yml            # Conda environment (GPU)
├── LICENSE
│
├── configs/                        # YAML experiment configs (single source of truth)
│   ├── kcat_esmc.yaml              #   kcat + ESM-C  (PRIMARY)
│   ├── km_esmc.yaml                #   KM   + ESM-C  (PRIMARY)
│   ├── kcat_esm2.yaml              #   kcat + ESM-2  (alternative)
│   └── km_esm2.yaml                #   KM   + ESM-2  (alternative)
│
├── src/                            # Core Python package
│   ├── __init__.py
│   ├── config.py                   #   Reference constants (GLOBAL_SEED=42)
│   ├── utils.py                    #   Device, seed, I/O helpers
│   ├── data_pipeline.py            #   Dataset loading, SMOTE, holdout
│   ├── model_training.py           #   Model loading & prediction
│   └── evaluation.py               #   Metrics, e-accuracy, CV aggregation
│
├── scripts/                        # Entry-point scripts
│   ├── cv_train.py                 #   5-fold cross-validation training
│   └── predict.py                  #   Inference on new embeddings
│
├── data/                           # Data directory
│   ├── raw/
│   ├── processed/
│   └── results/
│
├── data_robust_v1/                 # Curated training data (no OOD, no dups)
│   ├── data_split_curated_no_OOD_nodups_esmc/
│   └── data_split_curated_no_OOD_nodups/
│
├── outputs/                        # Training outputs (models, metrics, plots)
└── logs/                           # Training logs
```

---

## Reproducibility

All random operations use a fixed global seed (`GLOBAL_SEED = 42`) with deterministic PyTorch settings (`cudnn.deterministic = True`). Running the same config twice produces identical outputs.

```bash
# Run twice
catrange-train --config configs/kcat_esmc.yaml
catrange-train --config configs/kcat_esmc.yaml

# Compare — no diff means fully reproducible
diff outputs/kcat_esmc/results/crossval_summary.csv \
     outputs/kcat_esmc/results/crossval_summary.csv
```
---
## FAQ

**Q: Why ESM-C over ESM-2?**
ESM-C provides 1–2% better accuracy due to superior structural encoding.

**Q: Can I train on CPU?**
Yes. Install with `requirements_cpu.txt` and pass `--device cpu`.

**Q: How do I get exact result reproduction?**
Use the same config YAML, the same data directory, and the same environment. The fixed seed and deterministic settings ensure bitwise-identical outputs.

**Q: Can I tune the hyperparameters?**
Yes — edit the relevant YAML config. The shipped values are Optuna-tuned and recommended as-is.

**Q: Does `catrange-predict` accept raw sequences?**
No. It expects precomputed concatenated embeddings (`.pt`). Use the [Colab notebook](#quick-inference-colab) for end-to-end inference from raw enzyme sequence + Isomeric SMILES.

---
## 📚 Citation

If you use **CatRange** in your work, please cite the following:

> 🧬 Anna Sajeevan K, Osinuga A, B A, Ferdous S, Shahreen N, Noor MS, Koneru S, Santos-Correa LM, Salehi R, Chowdhury NB, Aryee R,Calderon-Lopez B, Mali A, Saha R, Chowdhury R.  
> **Robust Prediction of Enzyme Variant Kinetic Ranges with CatRange**  
> *bioRxiv* [Preprint], 2025 Feb 15. doi: [10.1101/2025.02.10.637555](https://doi.org/10.1101/2025.02.10.637555)  
> PMID: 39990461 · PMCID: PMC11844551

## License
This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

**Contact & Support:** [GitHub Issues](https://github.com/TKAI-LAB-Mali/CatRange/issues) · [GitHub Discussions](https://github.com/TKAI-LAB-Mali/CatRange/discussions)

*Last updated: February 27, 2026 · Version 1.2.0*

<details>
<summary>📄 BibTeX</summary>

```bibtex
@article{sajeevan2025robust,
  author = {Sajeevan, Anna K and Osinuga, Abraham and B, A and Ferdous, Sakib and Shahreen, Nabia and Noor, Mohammed Sakib and Koneru, Shashank and Santos-Correa, Laura Mariana and Salehi, Rahil and Chowdhury, Niaz Bahar and Aryee, Randy and Calderon-Lopez, Brisa and Mali, Ankur and Saha, Rajib and Chowdhury, Ratul},
  title = {Robust Prediction of Enzyme Variant Kinetic Ranges with CatRange},
  journal = {bioRxiv},
  year = {2025},
  month = {Feb},
  day = {15},
  note = {Preprint},
  doi = {10.1101/2025.02.10.637555},
  pmid = {39990461},
  pmcid = {PMC11844551}
}
