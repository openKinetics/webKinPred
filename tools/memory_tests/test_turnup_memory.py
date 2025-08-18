#!/usr/bin/env python3
"""
Measure max RAM needed to run TurNup for a single (max-length) sequence:
- Starts monitoring before heavy imports.
- Captures stage peaks: imports, model load, and one full prediction path.
- Reports both sampled RSS peaks and kernel ru_maxrss (kernel-reported peak).
"""

import os
import sys
import time
import gc
import threading
import psutil
import resource
import numpy as np

# ---------------- Memory helpers ----------------

def rss_mb() -> float:
    """Current process RSS in MB."""
    return psutil.Process(os.getpid()).memory_info().rss / (1024 * 1024)

def ru_maxrss_mb() -> float:
    """Kernel-reported peak RSS (MB). Linux returns KB; macOS returns bytes."""
    val = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    if sys.platform.startswith("linux"):
        return val / 1024.0
    else:
        return val / (1024.0 * 1024.0)

class MemoryMonitor:
    """High-frequency RSS sampler to catch short spikes."""
    def __init__(self, interval_s: float = 0.005):
        self.interval_s = interval_s
        self._peak_abs_mb = 0.0
        self._running = False
        self._t = None

    @property
    def peak_abs_mb(self) -> float:
        return self._peak_abs_mb

    def start(self):
        self._peak_abs_mb = rss_mb()
        self._running = True
        self._t = threading.Thread(target=self._loop, daemon=True)
        self._t.start()

    def stop(self):
        self._running = False
        if self._t is not None:
            self._t.join()

    def _loop(self):
        while self._running:
            cur = rss_mb()
            if cur > self._peak_abs_mb:
                self._peak_abs_mb = cur
            time.sleep(self.interval_s)

def run_stage(label, fn, mon, report):
    """Run a stage, logging RSS/ru_maxrss before/after and peak increases."""
    before_peak = mon.peak_abs_mb
    before_ru   = ru_maxrss_mb()
    before_rss  = rss_mb()
    t0 = time.time()

    out = fn()

    dt = time.time() - t0
    after_peak = mon.peak_abs_mb
    after_ru   = ru_maxrss_mb()
    after_rss  = rss_mb()

    report[label] = {
        "rss_after_mb": after_rss,
        "rss_delta_mb": after_rss - before_rss,
        "stage_peak_inc_mb": max(0.0, after_peak - before_peak),
        "ru_maxrss_inc_mb": max(0.0, after_ru - before_ru),
        "seconds": dt,
    }
    return out

# ---------------- Helpers from kcat_prediction_batch.py ----------------

def calculate_xgb_input_matrix(df):
    fingerprints = np.reshape(np.array(list(df["difference_fp"])), (-1, 2048))
    ESM1b = np.reshape(np.array(list(df["enzyme rep"])), (-1, 1280))
    X = np.concatenate([fingerprints, ESM1b], axis=1)
    return X

def merging_reaction_and_enzyme_df(df_reaction, df_enzyme, df_kcat):
    df_kcat["difference_fp"], df_kcat["enzyme rep"] = "", ""
    df_kcat["complete"] = True

    for ind in df_kcat.index:
        diff_fp = list(
            df_reaction["difference_fp"]
            .loc[df_reaction["substrates"] == df_kcat["substrates"][ind]]
            .loc[df_reaction["products"] == df_kcat["products"][ind]]
        )[0]
        esm1b_rep = list(
            df_enzyme["enzyme rep"]
            .loc[df_enzyme["amino acid sequence"] == df_kcat["enzyme"][ind]]
        )[0]

        if isinstance(diff_fp, str) and isinstance(esm1b_rep, str):
            df_kcat["complete"][ind] = False
        else:
            df_kcat["difference_fp"][ind] = diff_fp
            df_kcat["enzyme rep"][ind] = esm1b_rep
    return df_kcat

# ---------------- Main workflow ----------------

def test_turnup_memory():
    print("=== Peak RAM measurement: TurNup (single sequence) ===")

    # --- Config (edit as needed) ---
    code_dir = '/home/saleh/webKinPred/api/TurNup/code'
    data_dir = '/home/saleh/webKinPred/api/TurNup/data'
    seq_len = 1022  # max-length for ESM-1b token limit (approx.)
    test_substrate = 'InChI=1S/H2O/h1H2;InChI=1S/H2O/h1H2'
    test_product   = 'InChI=1S/H2O/h1H2;InChI=1S/H2O/h1H2'

    # Optionally cap threads for reproducible memory (set to None to disable)
    threads = 1
    if threads is not None:
        os.environ["OMP_NUM_THREADS"] = str(threads)
        os.environ["MKL_NUM_THREADS"] = str(threads)
        os.environ["OPENBLAS_NUM_THREADS"] = str(threads)
        os.environ["NUMEXPR_NUM_THREADS"] = str(threads)

    # Ensure TurNup code is importable before heavy imports
    if code_dir not in sys.path:
        sys.path.insert(0, code_dir)

    print(f"Baseline RSS before monitoring: {rss_mb():.2f} MB")

    mon = MemoryMonitor(interval_s=0.005)
    mon.start()
    stage_report = {}

    # Stage 1: imports (torch, xgboost, pandas, and TurNup utilities)
    def do_imports():
        global torch, xgb, pd, pickle, join, warnings
        global reaction_preprocessing, calcualte_esm1b_ts_vectors
        import torch
        import xgboost as xgb
        import pandas as pd
        import pickle
        import warnings
        from os.path import join
        from metabolite_preprocessing import reaction_preprocessing
        # Note: TurNup function name is spelled 'calcualte_esm1b_ts_vectors' in the repo
        from enzyme_representations import calcualte_esm1b_ts_vectors
        warnings.filterwarnings("ignore")
    run_stage("imports", do_imports, mon, stage_report)

    # Stage 2: load XGBoost model
    def load_xgb():
        global bst
        model_path = join(data_dir, "saved_models", "xgboost", "xgboost_train_and_test.pkl")
        with open(model_path, "rb") as f:
            bst = pickle.load(f)
    run_stage("xgb_load", load_xgb, mon, stage_report)

    # Stage 3: one full prediction path (reaction + enzyme rep + predict)
    def do_one_prediction():
        import numpy as _np

        # Create a max-length sequence of valid amino acids
        max_sequence_list = np.random.choice(list("ACDEFGHIKLMNPQRSTVWY"), size=(1022,))
        enzyme_upper = ''.join(max_sequence_list)

        # Reaction processing
        df_reaction = reaction_preprocessing(
            substrate_list=[test_substrate],
            product_list=[test_product]
        )

        # Enzyme representation (ESM1b-TS via TurNup util)
        df_enzyme = calcualte_esm1b_ts_vectors(enzyme_list=[enzyme_upper])

        # Build kcat dataframe and merge features
        import pandas as _pd
        df_kcat = _pd.DataFrame({
            "substrates": [test_substrate],
            "products": [test_product],
            "enzyme": [enzyme_upper],
            "index": [0],
        })

        df_kcat = merging_reaction_and_enzyme_df(df_reaction, df_enzyme, df_kcat)
        df_kcat_valid = df_kcat.loc[df_kcat["complete"]].reset_index(drop=True)

        if len(df_kcat_valid) == 0:
            print("Sample invalid for TurNup feature pipeline.")
            return

        # Prepare matrix and predict
        X = calculate_xgb_input_matrix(df=df_kcat_valid)
        dX = xgb.DMatrix(X)
        kcat = 10 ** bst.predict(dX)[0]
        print(f"Predicted kcat (one sample): {kcat:.4f} s^(-1)")

        # Encourage freeing transient buffers
        del df_reaction, df_enzyme, df_kcat, df_kcat_valid, X, dX
        gc.collect()

    run_stage("embed_infer_one_sequence", do_one_prediction, mon, stage_report)

    # Finish monitoring and summarise
    mon.stop()
    final_rss = rss_mb()
    peak_sampled = mon.peak_abs_mb
    peak_ru = ru_maxrss_mb()
    total_max_ram_needed = max(peak_sampled, peak_ru)

    print("\n=== Stage summary (MB) ===")
    for k, v in stage_report.items():
        print(f"- {k}: "
              f"rss_after={v['rss_after_mb']:.2f}, "
              f"rss_delta={v['rss_delta_mb']:.2f}, "
              f"stage_peak_inc={v['stage_peak_inc_mb']:.2f}, "
              f"ru_maxrss_inc={v['ru_maxrss_inc_mb']:.2f}, "
              f"time={v['seconds']:.3f}s")

    print("\n=== Overall ===")
    print(f"Final RSS: {final_rss:.2f} MB")
    print(f"Peak RSS (sampled): {peak_sampled:.2f} MB")
    print(f"Peak RSS (ru_maxrss): {peak_ru:.2f} MB")
    print(f"\nTOTAL MAX RAM NEEDED (single process, one sequence): {total_max_ram_needed:.2f} MB")

if __name__ == "__main__":
    test_turnup_memory()
