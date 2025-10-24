from __future__ import annotations
import argparse, json, math, pickle, joblib, signal, sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import optuna
from xgboost import XGBRegressor            # ← CHANGED (new import)
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.model_selection import GroupShuffleSplit


# ───────────────────── reproducibility ───────────────────── #
SEED = 42
rng  = np.random.default_rng(SEED)

# ─────────────────────── paths / constants ───────────────── #
# Determine repository root relative to this file
# hpo.py is in code/, so go up one level to get to repo root
ROOT        = Path(__file__).resolve().parent.parent
EITLEM_DIR  = ROOT / "data/EITLEM_data"
JSON_FILE   = EITLEM_DIR / "KCAT/kcat_data.json"
OUT_DIR     = ROOT / "results/hpo"
OUT_DIR.mkdir(parents=True, exist_ok=True)

SQLITE_DB   = OUT_DIR / "kcat_hpo.db"       # persistent Optuna backend
CSV_LOG     = OUT_DIR / "trials_log.csv"
N_CHECKPOINT = 10                           # checkpoint cadence
N_TRIALS     = 500                          # total trials

# ──────────────── import project utilities ───────────────── #
from smiles_embeddings.smiles_transformer.build_vocab import WordVocab  # noqa: F401
from utils.smiles_features    import smiles_to_vec
from utils.sequence_features  import sequences_to_feature_blocks
from utils.pca                import make_design_matrices
from config                   import SEQ_LOOKUP, BS_PRED_PATH

# ────────────────────── data utilities ───────────────────── #
def load_raw() -> Tuple[List[str], List[str], np.ndarray]:
    """Load JSON, filter invalid rows, subsample (30 for demo)."""
    with JSON_FILE.open() as fh:
        raw = json.load(fh)
    valid = [(i, r) for i, r in enumerate(raw)
             if len(r["sequence"]) <= 1499 and float(r["value"]) > 0]
    seqs  = [r["sequence"] for _, r in valid]
    smis  = [r["smiles"]   for _, r in valid]
    y     = np.array([math.log(float(r["value"]), 10) for _, r in valid],
                     dtype=np.float32)
    return seqs, smis, y

def make_groups(seqs: List[str]) -> List[int]:
    lookup     = pd.read_pickle(SEQ_LOOKUP)
    seq_to_id  = {v: k for k, v in lookup.items()}
    return [seq_to_id[s] for s in seqs]

def evaluate(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    rmse = math.sqrt(mean_squared_error(y_true, y_pred))
    return {"r2": r2_score(y_true, y_pred), "rmse": rmse}

# ────────────────── input preparation (runs once) ────────── #
print("→ Pre-computing feature blocks …")
seqs, smiles, y_all = load_raw()
groups              = make_groups(seqs)

bs_df       = pd.read_csv(BS_PRED_PATH, sep="\t")
smiles_vec  = smiles_to_vec(smiles, method="smiles_transformer")

blocks_all, names = sequences_to_feature_blocks(
    seqs, bs_df, None,
    {s: g for s, g in zip(seqs, groups)}, None,
    use_esmc=True, use_esm2=True, use_t5=True, t5_last_layer=True,
    prot_rep_mode="binding+global", task="kcat"
)
print("✓ Feature blocks ready.")

gss = GroupShuffleSplit(n_splits=1, test_size=0.10, random_state=SEED)
train_idx, test_idx = next(gss.split(seqs, groups=groups))
train_idx, test_idx = np.asarray(train_idx), np.asarray(test_idx)
y_train, y_test     = y_all[train_idx], y_all[test_idx]


# ───────────────────────── Optuna objective ──────────────────────────
def objective(trial: optuna.Trial) -> float:
    """One Optuna trial → R² on the fixed 10 % test split."""

    # ── 1. Feature-space decisions ────────────────────────────────── #
    use_pca = trial.suggest_categorical("use_pca", [True, False])
    n_comps = None 
    if use_pca:
        n_comps = trial.suggest_categorical(
            "n_comps", [300, 500, 700]
        )
    prot_rep = "both" if use_pca else trial.suggest_categorical(
        "prot_rep_mode", ["binding", "binding+global"]
    )

    # ── 2. XGBoost hyper-parameters ────────────────────────────────────── #
    booster_choice = trial.suggest_categorical("booster", ["gbtree", "dart"])

    xgb_params = {
        # core booster behaviour
        "booster": booster_choice,
        "objective": "reg:squarederror",
        "eval_metric": None,                # XGBoost default = rmse
        "base_score": 0.5,

        # tree growing
        "n_estimators": trial.suggest_int("n_estimators", 100, 2000, log=True),
        "learning_rate": trial.suggest_float("learning_rate", 1e-3, 0.3, log=True),
        "max_depth": trial.suggest_int("max_depth", 3, 12),
        "min_child_weight": trial.suggest_float("min_child_weight", 1e-3, 30, log=True),
        "gamma": trial.suggest_float("gamma", 0.0, 5.0),
        "max_delta_step": trial.suggest_int("max_delta_step", 0, 10),

        # column & row subsampling
        "subsample": trial.suggest_float("subsample", 0.3, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.3, 1.0),
        "colsample_bylevel": trial.suggest_float("colsample_bylevel", 0.5, 1.0),
        "colsample_bynode": trial.suggest_float("colsample_bynode", 0.5, 1.0),

        # regularisation
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 1.0, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 100.0, log=True),

        # histogram / GPU specifics
        "tree_method": trial.suggest_categorical("tree_method", ["hist", "gpu_hist"]),
        "max_bin": trial.suggest_int("max_bin", 128, 1024, log=True),

        # misc
        "gpu_id": 0,
        "random_state": SEED,
        "n_jobs": 0,
    }

    # ── booster-specific additions ──────────────────────────────────────── #
    if booster_choice == "dart":
        xgb_params.update({
            "rate_drop": trial.suggest_float("rate_drop", 0.0, 0.3),
            "skip_drop": trial.suggest_float("skip_drop", 0.0, 0.7),
            "sample_type": trial.suggest_categorical("sample_type",
                                                    ["uniform", "weighted"]),
            "normalize_type": trial.suggest_categorical("normalize_type",
                                                        ["tree", "forest"]),
            "one_drop": trial.suggest_categorical("one_drop", [0, 1]),
        })
        xgb_params["tree_method"] = "gpu_hist" 
    else:  # gbtree
        if xgb_params["tree_method"] == "gpu_hist":
            xgb_params["sampling_method"] = trial.suggest_categorical(
                "sampling_method", ["uniform", "gradient_based"]
            )
        else:  # hist → must be uniform
            xgb_params["sampling_method"] = "uniform"

    if xgb_params["tree_method"] == "hist":
        xgb_params["max_bin"] = min(xgb_params["max_bin"], 512)
        xgb_params["n_estimators"] = min(xgb_params["n_estimators"], 600)

    # ── 3. Build design matrices ──────────────────────────────────── #
    cfg = {"use_pca": use_pca, "n_comps": n_comps, "prot_rep_mode": prot_rep}
    X_tr, X_te,_ = make_design_matrices(
        train_idx, test_idx, blocks_all, names, cfg, smiles_vec
    )

    # ── 4. Train & evaluate ───────────────────────────────────────── #
    model = XGBRegressor(**xgb_params)
    model.fit(X_tr, y_train)
    y_pred = model.predict(X_te)

    r2 = r2_score(y_test, y_pred)
    rmse = math.sqrt(mean_squared_error(y_test, y_pred))
    trial.set_user_attr("rmse", rmse)

    return r2


# ─────────────────────── durability helpers ───────────────── #
def checkpoint(study: optuna.Study) -> None:
    """Flush the study to CSV."""
    study.trials_dataframe().to_csv(CSV_LOG, index=False)
    print("↳ checkpoint written:", CSV_LOG)

def on_trial_end(study: optuna.Study,
                 trial: optuna.trial.FrozenTrial) -> None:
    if trial.number % N_CHECKPOINT == 0:
        checkpoint(study)

def install_signal_handlers(study: optuna.Study) -> None:
    def _handler(signum, frame):
        print(f"\nSignal {signum} received — saving & exiting.")
        checkpoint(study)
        sys.exit(0)
    for sig in (signal.SIGINT, signal.SIGTERM):
        signal.signal(sig, _handler)

# ─────────────────────────── main ─────────────────────────── #
def main() -> None:
    study = optuna.create_study(
        study_name="kcat_hpo",
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=SEED),
        storage=f"sqlite:///{SQLITE_DB}",
        load_if_exists=True,
    )
    if len(study.trials) > 0:
        best_trial = max(
            study.trials,
            key=lambda t: t.value if t.value is not None else float("-inf")
        )
        print(f"→ Loaded existing study with {len(study.trials)} completed trials.")
        print(f"  Best R² so far: {best_trial.value:.4f} at trial #{best_trial.number}")
        remaining_trials = max(N_TRIALS - len(study.trials), 0)
        print(f"  {remaining_trials} trials remaining.")
    
    else:
        print("→ Starting a new study.")
        remaining_trials = N_TRIALS
        print(f"  {remaining_trials} trials to run.")

    install_signal_handlers(study)
    
    try:
        study.optimize(
            objective,
            n_trials=remaining_trials,
            callbacks=[on_trial_end],
            show_progress_bar=True,
        )
    except MemoryError as e:
        print("MemoryError! checkpointing before abort.")
        checkpoint(study)
        raise
    except Exception as e:
        print(f"{type(e).__name__} encountered — checkpointing.")
        checkpoint(study)
        raise
    finally:
        checkpoint(study)

    print(f"\n🏆  Best R² = {study.best_value:.4f}")
    print(json.dumps(study.best_params, indent=2))
    print("✓ All artifacts saved under", OUT_DIR)

# ────────────────────────── cli hook ───────────────────────── #
if __name__ == "__main__":
    main()
