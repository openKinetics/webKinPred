## Quick Inference (Colab): [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ssbio/CatRange/blob/main/CatRange_Inference_Interface.ipynb)
For interactive inference from **raw enzyme sequence + Isomeric SMILES** — no local install required: The CatRange inference notebook handles embedding generation, model loading, and prediction end-to-end. Simply connect to a Colab runtime and follow the instructions.
---

# CatRange (RealKcat): Robust Prediction of Enzyme Variant Kinetic Ranges

**CatRange** is a machine-learning pipeline for predicting enzyme kinetic parameters — k<sub>cat</sub> (catalytic constant) and K<sub>M</sub> (Michaelis constant) — from curated enzyme–substrate data. It combines protein-language-model embeddings with XGBoost classification to predict **log-scale kinetic ranges** rather than brittle point estimates.

The curated enzyme-kinetics database supporting CatRange is now referred to as **CatLog**. CatLog is intended to be periodically refreshed by an agentic curation pipeline so benchmark folds, metadata, and future releases can trace back to a living data resource.

---


[![Python 3.9+](https://img.shields.io/badge/Python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.0+](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)
[![XGBoost 2.0+](https://img.shields.io/badge/XGBoost-2.0+-green.svg)](https://xgboost.readthedocs.io/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Status: Reproducible](https://img.shields.io/badge/Status-Reproducible-brightgreen.svg)](#reproducibility)


# CatRange

Publication repository for **CatRange Enables Robust Prediction of Enzyme
Variant Kinetic Regimes**.

This repository is organized around the manuscript scope: CatLog-27k curation,
the CatRange XGBoost kinetic-regime classifier, mutation-aware catalytic-site
alanine evaluation, retrained comparator benchmarks, feature ablations, and
user inference from protein sequence plus substrate SMILES.

## What Is Included

```text
catrange_model/                  Full CatRange training and prediction code
data/                            CatLog/CatRange raw tables and metadata
results/catrange/                CatRange CV, OOD, holdout, and no-synthetic outputs
results/external_benchmarks/     Retrained CatPred/DLKcat/UniKP/EITLEM outputs
benchmarks/retrained_comparators/ Scripts for retraining/evaluating comparators
ablation/                        Full-vs-sequence-vs-substrate ablation scripts/results
figures/                         Benchmark figure source/output files
manuscript/                      Manuscript and SI PDFs plus submitted figures
inference/                       CatRange-only user inference utilities
notebooks/                       Three curated manuscript/inference notebooks
envs/                            Reproducible conda environment files
```

## Conda Environments

Create the recommended GPU environments:

```bash
bash scripts/env/create_conda_envs.sh all
```

The environments are intentionally split:

```text
catrange-notebooks-gpu     notebooks, CatRange training/evaluation, figures
catrange-esmc-gpu          ESM-C protein embeddings
catrange-chemberta-gpu     ChemBERTa SMILES embeddings
catrange-cpu-figures       CPU-only fallback for figure/table work
```

This split avoids the ESM-C and ChemBERTa/Transformers conflicts seen in a
single combined environment. See `envs/README.md` for kernel registration and
verification commands.

## Manuscript-Matched Scope

Included as first-class materials:

- CatRange kcat and KM regime classification with ESM-C/ChemBERTa features.
- ESM-2 comparison configs.
- SMOTE, hard-example reweighting, fold-level training, holdout, OOD, sequence-identity, and PafA evaluation code.
- CatPred, DLKcat, UniKP, and EITLEM-Kinetics retraining/evaluation scripts and benchmark outputs.
- Ablations from SI Fig. S4: full paired representation vs sequence-only vs substrate-only, plus no-synthetic-variant CatRange outputs.


## Train CatRange

```bash
cd catrange_model
python3 -m pip install --no-deps -e .

PYTHONPATH=. python scripts/cv_train.py --config configs/kcat_esmc.yaml --device cuda
PYTHONPATH=. python scripts/cv_train.py --config configs/km_esmc.yaml --device cuda
```

The primary manuscript configs are:

```text
catrange_model/configs/kcat_esmc.yaml
catrange_model/configs/km_esmc.yaml
```

Alternative ESM-2 comparison configs are also included:

```text
catrange_model/configs/kcat_esm2.yaml
catrange_model/configs/km_esm2.yaml
```

## Retrained Benchmarks

The manuscript states that CatPred, DLKcat, UniKP, and EITLEM-Kinetics were
retrained using matched CatRange fivefold partitions. The scripts for that
workflow are in:

```text
benchmarks/retrained_comparators/
```

The benchmark outputs currently included are in:

```text
results/external_benchmarks/
results/benchmark_summary_metrics.csv
results/benchmark_fold_metrics.csv
```

Large comparator caches/checkpoints are excluded from git.

## Ablations

The manuscript ablations are in:

```text
ablation/
```

This includes training/plotting scripts and completed kcat/KM outputs for:

- full CatRange paired enzyme-substrate representation;
- sequence-only;
- substrate-only.

The no-synthetic-variant CatRange outputs are in:

```text
results/catrange/no_synthetic_kcat_esmc/
```


## User Inference

Open:

```text
notebooks/00_CatRange_Data_Figures_Benchmarks_Inference.ipynb
```

The notebook supports:

- demo mode;
- interactive single sequence/SMILES input;
- batch CSV mode with `sequence` and `smiles` columns.

It generates ESM-C protein embeddings and ChemBERTa substrate embeddings before
running CatRange. The trained XGBoost binaries are too large for ordinary
GitHub tracking, so place downloaded models in:

```text
inference/models/
```

Expected model names:

```text
inference/models/kcat_esmc_FINAL.pkl
inference/models/km_esmc_FINAL.pkl
```

Small standardization-stat files are included when available.

## Data

The `data/raw/` folder contains the CatLog/CatRange raw and curated input
tables copied from the manuscript CatRange workspace, including BRENDA/SABIO
extracts, curated WT/MD tables, PafA data, OOD anchor data, and substrate
SMILES mappings.

## Citation

Please cite the CatRange manuscript when using this code or data. 
