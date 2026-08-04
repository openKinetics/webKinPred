#!/usr/bin/env bash
set -euo pipefail

ROOT="/work/ssbio/aosinuga2/Python_work"
PY="${PY_CATRANGE:-/work/ssbio/aosinuga2/envs/CatRange_env_gpu/bin/python}"
OUT="${1:-$ROOT/CatRange_Ablation_Study/runs/kcat_esmc_ablation}"

"$PY" "$ROOT/CatRange_Ablation_Study/train_catrange_ablation.py" \
  --catrange-root "$ROOT/CatRange" \
  --config "$ROOT/CatRange/configs/kcat_esmc.yaml" \
  --output-dir "$OUT" \
  --device auto

"$PY" "$ROOT/CatRange_Ablation_Study/plot_ablation_results.py" \
  --results-dir "$OUT" \
  --dpi 600

echo "CatRange ablation complete: $OUT"
