# CatRange Conda Environments

CatRange uses separate conda environments for the notebook/model workflow and
for the two embedding backends that have historically conflicted when installed
together.

## Recommended Environments

| Environment | File | Purpose |
| --- | --- | --- |
| `catrange-notebooks-gpu` | `envs/catrange-notebooks-gpu.yml` | Main GPU environment for the three curated notebooks, CatRange model training/evaluation, manuscript figures, XGBoost, ESM-2/fair-esm, RDKit, and benchmark utilities. |
| `catrange-esmc-gpu` | `envs/catrange-esmc-gpu.yml` | ESM-C protein embeddings only. Kept separate from ChemBERTa/Transformers to avoid dependency conflicts. |
| `catrange-chemberta-gpu` | `envs/catrange-chemberta-gpu.yml` | ChemBERTa substrate embeddings only. |
| `catrange-cpu-figures` | `envs/catrange-cpu-figures.yml` | CPU-only fallback for figure generation and lightweight table analysis. |

The GPU YAMLs use `pytorch-cuda=11.8`. This is compatible with modern NVIDIA
drivers, including the tested local driver reporting CUDA 13.0 support, because
NVIDIA drivers are backward-compatible with CUDA runtimes older than the driver.

## Create Environments

For more reproducible solves, set strict channel priority before creating the
environments:

```bash
conda config --set channel_priority strict
```

Create all recommended GPU environments:

```bash
bash scripts/env/create_conda_envs.sh all
```

Or create them one by one:

```bash
bash scripts/env/create_conda_envs.sh notebooks-gpu
bash scripts/env/create_conda_envs.sh esmc-gpu
bash scripts/env/create_conda_envs.sh chemberta-gpu
```

CPU-only figure environment:

```bash
bash scripts/env/create_conda_envs.sh figures-cpu
```

## Register Jupyter Kernels

After creating an environment, register it as a notebook kernel:

```bash
conda run -n catrange-notebooks-gpu python -m ipykernel install --user --name catrange-notebooks-gpu --display-name "CatRange notebooks GPU"
conda run -n catrange-esmc-gpu python -m ipykernel install --user --name catrange-esmc-gpu --display-name "CatRange ESM-C GPU"
conda run -n catrange-chemberta-gpu python -m ipykernel install --user --name catrange-chemberta-gpu --display-name "CatRange ChemBERTa GPU"
```

## Verify

```bash
bash scripts/env/verify_conda_envs.sh
```

## Why Separate ESM-C and ChemBERTa?

The inference code needs ESM-C for protein embeddings and ChemBERTa via
Hugging Face Transformers for SMILES embeddings. ESM-C comes from the
EvolutionaryScale/Biohub ESM package, while ChemBERTa relies on a specific
Transformers/tokenizers stack. Installing both into one environment has caused
resolver and runtime conflicts before, so the reproducible setup intentionally
splits them.

The intended workflow is:

1. Use `catrange-esmc-gpu` to generate protein sequence embeddings.
2. Use `catrange-chemberta-gpu` to generate substrate SMILES embeddings.
3. Use `catrange-notebooks-gpu` to train/evaluate CatRange, run XGBoost models,
   and generate manuscript figures from saved embeddings/results.
