# CatRange Feature Ablation Study

This folder runs a CatRange-only ablation study without editing the main
`CatRange` repository.

The study compares these feature modes for both k<sub>cat</sub> and K<sub>M</sub>:

- `Full CatRange`: ESM-C sequence embedding plus substrate/auxiliary features
- `Sequence only`: first `1152` ESM-C columns only
- `Substrate only`: columns after the ESM-C block only

The split point is read from `CatRange/configs/kcat_esmc.yaml`:

```yaml
embedding:
  feature_split: 1152
```

## Run k<sub>cat</sub> From Start To Finish

From `/work/ssbio/aosinuga2/Python_work`:

```bash
/work/ssbio/aosinuga2/envs/CatRange_env_gpu/bin/python \
  CatRange_Ablation_Study/train_catrange_ablation.py \
  --catrange-root CatRange \
  --config CatRange/configs/kcat_esmc.yaml \
  --output-dir CatRange_Ablation_Study/runs/kcat_esmc_ablation \
  --device cuda

/work/ssbio/aosinuga2/envs/CatRange_env_gpu/bin/python \
  CatRange_Ablation_Study/plot_ablation_results.py \
  --results-dir CatRange_Ablation_Study/runs/kcat_esmc_ablation \
  --dpi 600
```

## Run K<sub>M</sub> From Start To Finish

From `/work/ssbio/aosinuga2/Python_work`:

```bash
/work/ssbio/aosinuga2/envs/CatRange_env_gpu/bin/python \
  CatRange_Ablation_Study/train_catrange_ablation.py \
  --catrange-root CatRange \
  --config CatRange/configs/km_esmc.yaml \
  --output-dir CatRange_Ablation_Study/runs/km_esmc_ablation \
  --device cuda \
  --feature-mode sequence \
  --feature-mode substrate

/work/ssbio/aosinuga2/envs/CatRange_env_gpu/bin/python \
  CatRange_Ablation_Study/plot_ablation_results.py \
  --results-dir CatRange_Ablation_Study/runs/km_esmc_ablation \
  --parameter km \
  --dpi 600
```

For a quick smoke test, run only the first fold:

```bash
/work/ssbio/aosinuga2/envs/CatRange_env_gpu/bin/python \
  CatRange_Ablation_Study/train_catrange_ablation.py \
  --catrange-root CatRange \
  --config CatRange/configs/kcat_esmc.yaml \
  --output-dir CatRange_Ablation_Study/runs/smoke_kcat_esmc_ablation \
  --device cpu \
  --max-folds 1 \
  --skip-hnm
```

## Outputs

The full run writes:

- `runs/kcat_esmc_ablation/full/fold_metrics.csv`
- `runs/kcat_esmc_ablation/sequence/fold_metrics.csv`
- `runs/kcat_esmc_ablation/substrate/fold_metrics.csv`
- `runs/kcat_esmc_ablation/all_fold_metrics.csv`
- `runs/kcat_esmc_ablation/all_summary_metrics.csv`
- `runs/kcat_esmc_ablation/ablation_summary_mean_pm_std.csv`
- `runs/kcat_esmc_ablation/ablation_manifest.csv`
- `runs/kcat_esmc_ablation/figures/catrange_kcat_feature_ablation_metrics.png`
- `runs/km_esmc_ablation/figures/catrange_km_feature_ablation_metrics.png`

## Interpretation

Use the `Full CatRange` row as the reference. The `Sequence only` and
`Substrate only` rows quantify how much of the kcat binning performance comes
from enzyme sequence representation versus substrate/auxiliary information.

The ablation follows the CatRange CV procedure: same seed, same WT/MD folds,
same negative relabelling logic, same SMOTE setting, same class weighting, and
same hard-negative mining unless `--skip-hnm` is used. In the completed
manuscript run, the `Full CatRange` baseline was imported from the existing
completed CatRange kcat ESM-C CV output, while `Sequence only` and
`Substrate only` were retrained in this ablation folder.
