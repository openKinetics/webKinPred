#!/usr/bin/env bash
set -euo pipefail

echo "[CatRange] Verifying catrange-notebooks-gpu"
conda run -n catrange-notebooks-gpu python - <<'PY'
import torch, numpy, pandas, sklearn, xgboost, matplotlib, openpyxl, joblib
import esm
print("torch", torch.__version__, "cuda", torch.cuda.is_available())
print("numpy", numpy.__version__, "pandas", pandas.__version__)
print("sklearn", sklearn.__version__, "xgboost", xgboost.__version__)
PY

echo "[CatRange] Verifying catrange-esmc-gpu"
conda run -n catrange-esmc-gpu python - <<'PY'
import torch
from esm.models.esmc import ESMC
from esm.sdk.api import ESMProtein, LogitsConfig
print("torch", torch.__version__, "cuda", torch.cuda.is_available())
print("ESM-C imports OK")
PY

echo "[CatRange] Verifying catrange-chemberta-gpu"
conda run -n catrange-chemberta-gpu python - <<'PY'
import torch, transformers
from transformers import AutoModel, AutoTokenizer
print("torch", torch.__version__, "cuda", torch.cuda.is_available())
print("transformers", transformers.__version__)
print("ChemBERTa imports OK")
PY
