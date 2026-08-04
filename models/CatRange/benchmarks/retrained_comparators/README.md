# CatRange External-Model Benchmark Workspace

This folder is a standalone workspace for benchmarking **CatPred**, **DLKcat**,
**UniKP**, and **EITLEM-Kinetics** against the same binning and classification
metrics used by **CatRange**.

It does **not** modify anything inside the legacy local `CatRange/` source tree.

## What This Workspace Does

- prepares a benchmark dataset into tool-specific input files
- preserves a stable `pair_id` across tools
- bins continuous predictions into the same CatRange classes
- computes the same classification metrics used in the CatRange evaluation code
  - `accuracy`
  - `e_accuracy` (within `+-1` bin)
  - `precision`
  - `recall`
  - `f1`
  - `mcc`
  - `auc_pr` (only when class probabilities are available; otherwise `NaN`)

## CatRange Bin Edges Used Here

- `kcat`: `[0, 1e-8, 1e-2, 1e-1, 1e0, 1e1, 1e2, 1e3, 1e8]`
- `km`: `[1e-14, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1e4]`

These are copied into this workspace so benchmarking stays independent of
the legacy local `CatRange/` source tree.

## Install

Use any Python environment with:

```bash
python3 -m pip install -r requirements.txt
```

If you want to drive the full workflow from Jupyter, open:

- `CatRange_CatPred_DLKcat_Benchmark.ipynb`

That notebook wraps the suite builder, CatRange/CatPred/DLKcat/UniKP/EITLEM
prediction runs, evaluation, and single-panel figure export into one place.
Edit the configuration cell near the top, then run the notebook from top to
bottom.

The benchmark orchestrator expects model-specific environments:

- `PY_REALKCAT` for CatRange inference, evaluation, and plotting
- `PY_CATPRED` for CatPred inference
- `PY_DLKCAT` for DLKcat inference
- `PY_UNIKP` for UniKP feature extraction and training
- `PY_EITLEM` for EITLEM feature extraction and training
- `PY_ESM` for CatPred ESM2 record generation

The default notebook config now targets these local prefixes under `$WORK/envs`:

- `/work/ssbio/aosinuga2/envs/CatRange_env_gpu`
- `/work/ssbio/aosinuga2/envs/catpred_model`
- `/work/ssbio/aosinuga2/envs/dlkcat_model`
- `/work/ssbio/aosinuga2/envs/unikp_model`
- `/work/ssbio/aosinuga2/envs/eitlem_model`

`PY_ESM` defaults to the same Python executable as `PY_CATPRED`.

On this machine those local prefixes can be lightweight symlinks to known-good
shared envs rather than full duplicated installs.

## Recommended Workflow

### Use CatRange’s Exact Fold Partitions

If you want the benchmark inputs to come directly from the exact CatRange fold
workbooks you used in training, use:

```bash
python3 import_realkcat_partitions.py \
  --realkcat-root /work/ssbio/aosinuga2/Python_work/CatRange \
  --parameter kcat \
  --fold 5 \
  --output-dir runs/realkcat_fold5_kcat
```

For the primary CatRange publication-style setup, this is the closest match:

- `configs/kcat_esmc.yaml` for parameter and embedding
- `--fold 5`, because `realkcat-train` defaults to `--final-fold 5`
- `--split test` during evaluation if you want a clean held-out comparison

That means the most direct benchmark path for CatPred vs DLKcat is:

```bash
python3 import_realkcat_partitions.py \
  --realkcat-root /work/ssbio/aosinuga2/Python_work/CatRange \
  --parameter kcat \
  --fold 5 \
  --output-dir runs/realkcat_kcat_esmc_fold5
```

This reads the same partition workbook CatRange uses, enriches it with metadata
from the CatRange master sheet, and writes:

- `realkcat_partition_master.csv`
- `truth_standardized.csv`
- `catpred_input.csv`
- `catpred_input_train.csv`
- `catpred_input_val.csv`
- `catpred_input_test.csv`
- `dlkcat_input.tsv`
- `dlkcat_input_train.tsv`
- `dlkcat_input_val.tsv`
- `dlkcat_input_test.tsv`

The `truth_standardized.csv` file includes a `split` column, so evaluation can
be restricted to the exact partition you want:

```bash
python3 evaluate_benchmark.py \
  --truth-csv runs/realkcat_kcat_esmc_fold5/truth_standardized.csv \
  --pred-csv runs/realkcat_kcat_esmc_fold5/catpred_standardized.csv \
  --pred-csv runs/realkcat_kcat_esmc_fold5/dlkcat_standardized.csv \
  --split test \
  --output-dir runs/realkcat_kcat_esmc_fold5/results_test_only
```

### 1. Prepare a benchmark master CSV

Your input CSV should contain, at minimum:

- a ground-truth continuous value
- `sequence`
- `smiles`
- optionally `substrate_name`
- ideally an explicit `pair_id`

### 2. Prepare standardized truth + tool inputs

```bash
python3 prepare_benchmark_inputs.py \
  --input-csv path/to/master.csv \
  --output-dir runs/my_kcat_benchmark \
  --parameter kcat \
  --truth-column true_kcat \
  --sequence-column sequence \
  --smiles-column smiles \
  --substrate-name-column substrate_name \
  --pair-id-column pair_id
```

This writes:

- `truth_standardized.csv`
- `catpred_input.csv`
- `dlkcat_input.tsv` for `kcat` benchmarks

### 3. Run the model predictions

The notebook wrapper can launch all supported models directly. If you prefer to
run them yourself, the benchmark runners are:

- `run_catpred_suite.py`
- `run_catpred_retrain_suite.py`
- `run_dlkcat_suite.py`
- `run_dlkcat_retrain_suite.py`
- `run_unikp_suite.py`
- `run_eitlem_suite.py`

The generated `pair_id` is preserved:

- CatPred usually keeps extra columns in its output
- `run_catpred_retrain_suite.py` trains one CatPred checkpoint root per CatRange fold,
  then predicts every suite set tied to that fold
- DLKcat ignores the extra `pair_id` column during prediction, but we recover it
  later by reading the original input TSV alongside the output TSV
- `run_dlkcat_retrain_suite.py` uses the official DLKcat graph/3-mer model with a
  fold-aware training wrapper so it can fit on the CatRange `train/val` splits
- UniKP trains one fold-specific regressor per CatRange fold, then predicts every
  suite set tied to that fold
- EITLEM trains one fold-specific core predictor per CatRange fold, using
  `train_partition` for fitting and `val_partition` for checkpoint selection

### 4. Normalize outputs into a shared schema

CatPred:

```bash
python3 normalize_catpred.py \
  --input-csv path/to/catpred_output.csv \
  --output-csv runs/my_kcat_benchmark/catpred_standardized.csv \
  --parameter kcat
```

DLKcat:

```bash
python3 normalize_dlkcat.py \
  --input-tsv runs/my_kcat_benchmark/dlkcat_input.tsv \
  --output-tsv path/to/dlkcat_output.tsv \
  --output-csv runs/my_kcat_benchmark/dlkcat_standardized.csv
```

### 5. Evaluate the benchmark

```bash
python3 evaluate_benchmark.py \
  --truth-csv runs/my_kcat_benchmark/truth_standardized.csv \
  --pred-csv runs/my_kcat_benchmark/catpred_standardized.csv \
  --pred-csv runs/my_kcat_benchmark/dlkcat_standardized.csv \
  --output-dir runs/my_kcat_benchmark/results
```

Outputs include:

- `summary_metrics.csv`
- `summary_metrics.txt`
- per-model matched tables
- per-model confusion matrices

### 6. Generate a manuscript-ready comparison figure

After evaluation, create a figure that compares the binned metrics and
row-normalized confusion matrices side by side:

```bash
python3 plot_benchmark_figure.py \
  --results-dir runs/realkcat_kcat_esmc_fold5/results_test_only \
  --title "Fold 5 kcat Benchmark on CatRange Test Partition"
```

This writes:

- `benchmark_manuscript_figure.png`
- `benchmark_manuscript_figure.pdf`

in the selected results directory.

## Notes

- `DLKcat` is only supported here for `kcat`.
- `CatPred` can be benchmarked for `kcat` and `km`.
- `UniKP` benchmarking here follows the published representation recipe
  (ProtT5 protein embedding + SMILES Transformer molecular embedding) and trains
  an `ExtraTreesRegressor` on each CatRange fold.
- `EITLEM-Kinetics` benchmarking here uses the published `KCAT` / `KM`
  predictor architecture with cached ESM1v residue embeddings and MACCS keys.
  It does **not** currently reconstruct the paper's full `KCAT/KM/KKM`
  iterative transfer-learning loop on CatRange fold exports.
- `CatPred` outputs `km` in `mM`, but CatRange bins `km` in `M`.
  `normalize_catpred.py` converts CatPred `km` values into CatRange-compatible
  units before binning.
- If your dataset contains duplicate `sequence + smiles + substrate_name`
  combinations, use an explicit `pair_id` column. That is the safest alignment key.

## Example

The `examples/` folder includes a tiny `kcat` toy dataset plus example CatPred and
DLKcat outputs so you can test the workflow locally.
