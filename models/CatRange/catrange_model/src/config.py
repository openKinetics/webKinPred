"""
CatRange Configuration Defaults

This module exists as a **reference** for the Optuna-tuned hyperparameters.
All runtime configuration is driven by YAML files in configs/.
cv_train.py reads everything from YAML; this module is NOT imported at runtime.
"""

# ── Reproducibility ─────────────────────────────────────────────────────────
GLOBAL_SEED = 42

# ── Data ────────────────────────────────────────────────────────────────────
KCAT_BIN_EDGES = [0, 1e-8, 1e-2, 1e-1, 1e0, 1e1, 1e2, 1e3, 1e8]   # 8 classes
KM_BIN_EDGES   = [1e-14, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1e4]       # 6 classes

ESMC_FEATURE_SPLIT = 1152
ESM2_FEATURE_SPLIT = 1280

# ── XGBoost (Optuna Trial 9, objective=0.6119) ─────────────────────────────
XGBOOST_PARAMS = {
    "n_estimators":     1920,
    "max_depth":        11,
    "learning_rate":    0.0021099437081941153,
    "subsample":        1.0,
    "colsample_bytree": 0.4604393961508079,
    "alpha":            1.1182596551965625,    # L1
    "lambda":           14.80330404900776,     # L2
    "objective":        "multi:softmax",
    "eval_metric":      ["mlogloss"],
    "n_jobs":           1,
    "verbosity":        0,
    "random_state":     GLOBAL_SEED,
}
