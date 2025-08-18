#!/usr/bin/env python3
"""
Measure max RAM required to load ESM-1v + EITLEM and process one sequence.

What it does:
- Starts memory monitoring *before* heavy imports.
- Records stage-by-stage peaks (imports, ESM load, EITLEM load, embedding+inference).
- Reports both sampled RSS peaks and kernel-reported ru_maxrss.
- Avoids unnecessary NumPy copies; uses torch.inference_mode().
- Lets you set thread caps for deterministic memory.

Usage (with your paths):
  python test_eitlem_peak_ram.py \
    --esm /home/saleh/webKinPred/api/EITLEM/Weights/esm1v/esm1v_t33_650M_UR90S_1.pt \
    --eitlem /home/saleh/webKinPred/api/EITLEM/Weights/KCAT/iter8_trainR2_0.9408_devR2_0.7459_RMSE_0.7751_MAE_0.4787 \
    --code /home/saleh/webKinPred/api/EITLEM/Code \
    --seq_len 1022 --smiles CCO --threads 1

Tip: run once with threads=1 (reproducible), then without setting threads to see the real-world worst case.
"""

import os
import sys
import time
import gc
import threading
import argparse
import psutil
import resource
from typing import Callable, Dict, Any

# ---------- Memory helpers ----------

def rss_mb() -> float:
    p = psutil.Process(os.getpid())
    return p.memory_info().rss / (1024 * 1024)

def ru_maxrss_mb() -> float:
    # Linux reports KB; macOS reports bytes
    val = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    if sys.platform.startswith("linux"):
        return val / 1024.0
    else:
        return val / (1024.0 * 1024.0)

class MemoryMonitor:
    """High-frequency RSS sampler to catch transient spikes."""
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

# ---------- Stage runner ----------

def run_stage(label: str, fn: Callable[[], Any], mon: MemoryMonitor, report: Dict[str, Dict[str, float]]):
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

# ---------- Main workflow ----------

def main():
    ap = argparse.ArgumentParser(description="Peak RAM measurement for ESM-1v + EITLEM (single sequence).")
    ap.add_argument("--esm",   default="/home/saleh/webKinPred/api/EITLEM/Weights/esm1v/esm1v_t33_650M_UR90S_1.pt", help="Path to ESM-1v weights (.pt)")
    ap.add_argument("--eitlem", default="/home/saleh/webKinPred/api/EITLEM/Weights/KCAT/iter8_trainR2_0.9408_devR2_0.7459_RMSE_0.7751_MAE_0.4787", help="Path to EITLEM kcat model state_dict")
    ap.add_argument("--code",  default="/home/saleh/webKinPred/api/EITLEM/Code", help="Path to EITLEM/Code to add to sys.path")
    ap.add_argument("--seq_len", type=int, default=1022, help="Sequence length to test (max for ESM-1v ~1022 aa)")
    ap.add_argument("--smiles", default="CCOCCOCCOCCOCCOCCOCCOCCOCCOCCOCCOCCOCCOCCOCCOCCOCCO", help="Substrate SMILES (MACCS fingerprint)")
    ap.add_argument("--threads", type=int, default=None, help="Set torch/OMP thread count (e.g., 1 for reproducible memory)")
    ap.add_argument("--monitor_interval_ms", type=float, default=5.0, help="Memory sampling interval in milliseconds")
    args = ap.parse_args()

    # Optional thread caps for reproducibility
    if args.threads is not None:
        os.environ["OMP_NUM_THREADS"] = str(args.threads)
        os.environ["MKL_NUM_THREADS"] = str(args.threads)
        os.environ["OPENBLAS_NUM_THREADS"] = str(args.threads)
        os.environ["NUMEXPR_NUM_THREADS"] = str(args.threads)

    # Ensure EITLEM code on path before imports
    if args.code and args.code not in sys.path:
        sys.path.insert(0, args.code)

    print("=== Peak RAM measurement: ESM-1v + EITLEM (single sequence) ===")
    print(f"Baseline RSS before monitoring: {rss_mb():.2f} MB")

    mon = MemoryMonitor(interval_s=max(0.001, args.monitor_interval_ms / 1000.0))
    mon.start()
    stage_report: Dict[str, Dict[str, float]] = {}

    # Stage 1: heavy imports (torch, esm, rdkit, pyg, EITLEM)
    def do_imports():
        global torch, esm, Chem, MACCSkeys, Data, Batch, EitlemKcatPredictor
        import torch
        import esm
        from rdkit import Chem
        from rdkit.Chem import MACCSkeys
        from torch_geometric.data import Data, Batch
        from KCM import EitlemKcatPredictor
        _ = (torch.__version__, esm.__version__ if hasattr(esm, "__version__") else "unknown")
    run_stage("imports", do_imports, mon, stage_report)

    # Stage 2: load ESM model
    def load_esm():
        global esm_model, alphabet, batch_converter
        esm_model, alphabet = esm.pretrained.load_model_and_alphabet_local(model_location=args.esm)
        esm_model.eval()
        # Move to CPU explicitly to be clear (even if already is)
        esm_model = esm_model.to("cpu")
        batch_converter = alphabet.get_batch_converter()
    run_stage("esm_load", load_esm, mon, stage_report)

    # Stage 3: load EITLEM predictor
    def load_eitlem():
        global eitlem_model
        import torch
        eitlem_model = EitlemKcatPredictor(167, 512, 1280, 10, 0.5, 10)
        sd = torch.load(args.eitlem, map_location=torch.device("cpu"))
        eitlem_model.load_state_dict(sd)
        eitlem_model.eval()
    run_stage("eitlem_load", load_eitlem, mon, stage_report)

    # Stage 4: embedding + inference with one max-length sequence
    def do_inference():
        import torch, numpy as np
        from rdkit import Chem
        from rdkit.Chem import MACCSkeys
        from torch_geometric.data import Data, Batch

        max_sequence_list = np.random.choice(list("ACDEFGHIKLMNPQRSTVWY"), size=(1022,))
        seq = ''.join(max_sequence_list)

        data = [("protein", seq)]
        _, _, batch_tokens = batch_converter(data)
        batch_lens = (batch_tokens != alphabet.padding_idx).sum(1)
        tokens_len = int(batch_lens[0].item())

        with torch.inference_mode():
            results = esm_model(batch_tokens, repr_layers=[33], return_contacts=False)
            token_reps = results["representations"][33]  # [B, L, 1280]
            rep = token_reps[0, 1:tokens_len - 1].detach().cpu().contiguous()  # [seq_len-2, 1280]

        # MACCS fingerprint for the substrate
        mol = Chem.MolFromSmiles(args.smiles)
        if mol is None:
            raise ValueError(f"Invalid SMILES: {args.smiles}")
        maccs = MACCSkeys.GenMACCSKeys(mol).ToList()  # 167 bits

        sample = Data(
            x=torch.as_tensor(maccs, dtype=torch.float32).unsqueeze(0),  # [1, 167]
            pro_emb=rep.float(),  # [seq_len-2, 1280]
        )
        batch = Batch.from_data_list([sample], follow_batch=["pro_emb"])

        with torch.inference_mode():
            _ = eitlem_model(batch)

        # Encourage freeing transient buffers
        del results, token_reps, rep, sample, batch, batch_tokens
        gc.collect()

    run_stage("embed_infer_one_sequence", do_inference, mon, stage_report)

    # Finish monitoring
    mon.stop()

    # ---------- Summary ----------
    overall_peak_sampled = mon.peak_abs_mb
    overall_peak_ru = ru_maxrss_mb()
    final_rss = rss_mb()

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
    print(f"Peak RSS (sampled): {overall_peak_sampled:.2f} MB")
    print(f"Peak RSS (ru_maxrss): {overall_peak_ru:.2f} MB")

    total_max_ram_needed = max(overall_peak_sampled, overall_peak_ru)
    print(f"\nTOTAL MAX RAM NEEDED (single process, one sequence): {total_max_ram_needed:.2f} MB")

if __name__ == "__main__":
    main()
