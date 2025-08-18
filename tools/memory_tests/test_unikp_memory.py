#!/usr/bin/env python3
"""
Peak RAM measurement for UniKP (single sample):
- Starts monitoring before heavy imports.
- Captures stage peaks: imports, SMILES model load, optional ProtT5-XL load,
  UniKP predictor load, and one full feature/predict pass.
- Reports both sampled RSS peaks and kernel ru_maxrss (peak RSS).
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

# ---------------- UniKP helpers (from run_unikp_batch.py) ----------------

def resolve_seq_ids_via_cli(sequences, seqmap_py, seqmap_cli, seqmap_db):
    """Resolve stable IDs for sequences via seqmap CLI."""
    import subprocess
    payload = "\n".join(sequences) + "\n"
    cmd = [seqmap_py, seqmap_cli, "--db", seqmap_db, "batch-get-or-create", "--stdin"]
    proc = subprocess.run(cmd, input=payload, text=True,
                          stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if proc.returncode != 0:
        raise RuntimeError(f"seqmap CLI failed (rc={proc.returncode})\nSTDOUT:\n{proc.stdout}\nSTDERR:\n{proc.stderr}")
    ids = proc.stdout.strip().splitlines()
    if len(ids) != len(sequences):
        raise RuntimeError(f"seqmap returned {len(ids)} ids for {len(sequences)} sequences")
    return ids

def load_t5_model_local(prott5_path):
    """Load local ProtT5-XL (encoder) and tokenizer on CPU."""
    import torch
    from transformers import T5Tokenizer, T5EncoderModel
    gc.collect()
    try:
        tokenizer = T5Tokenizer.from_pretrained(prott5_path, do_lower_case=False)
        model = T5EncoderModel.from_pretrained(
            prott5_path,
            low_cpu_mem_usage=True,
            torch_dtype=torch.float32,
        )
        model = model.eval().to("cpu")
        return tokenizer, model, "cpu"
    except Exception as e:
        raise RuntimeError(f"Failed to load ProtT5-XL from {prott5_path}: {e}")


def smiles_to_vec(Smiles, vocab, trfm):
    """Convert SMILES to vectors using pre-loaded models."""
    pad_index = 0
    unk_index = 1
    eos_index = 2
    sos_index = 3
    mask_index = 4

    def get_inputs(sm):
        seq_len = 220
        sm = sm.split()
        if len(sm) > 218:
            sm = sm[:109] + sm[-109:]
        ids = [vocab.stoi.get(token, unk_index) for token in sm]
        ids = [sos_index] + ids + [eos_index]
        seg = [1] * len(ids)
        padding = [pad_index] * (seq_len - len(ids))
        ids.extend(padding)
        seg.extend(padding)
        return ids, seg

    def get_array(smiles):
        x_id, x_seg = [], []
        for sm in smiles:
            a, b = get_inputs(sm)
            x_id.append(a)
            x_seg.append(b)
        return torch.tensor(x_id), torch.tensor(x_seg)

    x_split = [split(sm) for sm in Smiles]
    xid, xseg = get_array(x_split)
    X = trfm.encode(torch.t(xid))
    return X


def seq_to_vec(sequences, tokenizer, model, device, seq_vec_dir, seqmap_py, seqmap_cli, seqmap_db):
    """Return np.ndarray [N, 1024] ProtT5-XL mean-pooled embeddings; caches to disk."""
    import torch, re
    ids = resolve_seq_ids_via_cli(sequences, seqmap_py, seqmap_cli, seqmap_db)

    vecs = []
    seqs_to_embed, ids_to_embed = [], []
    for seq, sid in zip(sequences, ids):
        vec_path = os.path.join(seq_vec_dir, f"{sid}.npy")
        if os.path.exists(vec_path):
            vecs.append(np.load(vec_path))
        else:
            seqs_to_embed.append(seq)
            ids_to_embed.append(sid)

    if seqs_to_embed:
        print(f"Generating embeddings for {len(seqs_to_embed)} sequences...")
        for seq, sid in zip(seqs_to_embed, ids_to_embed):
            spaced = ' '.join(seq)
            spaced = re.sub(r"[UZOB]", "X", spaced)
            enc = tokenizer.batch_encode_plus([spaced], add_special_tokens=True, padding=True)
            input_ids = torch.tensor(enc['input_ids']).to(device)
            attention_mask = torch.tensor(enc['attention_mask']).to(device)
            with torch.inference_mode():
                hidden = model(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state  # [1, L, 1024]
            hidden = hidden.cpu().numpy()[0]
            seq_len = int((attention_mask[0] == 1).sum().item())
            seq_vec = hidden[:seq_len - 1].mean(axis=0).astype(np.float32)  # [1024]
            out_path = os.path.join(seq_vec_dir, f"{sid}.npy")
            np.save(out_path, seq_vec)
            vecs.append(seq_vec)

    return np.stack(vecs)

# ---------------- Main workflow ----------------

def test_unikp_memory():
    print("=== Peak RAM measurement: UniKP (single sample) ===")

    # --- Config (edit as needed) ---
    UNIKP_DIR = '/home/saleh/webKinPred/api/UniKP-main'
    SEQ_VEC_DIR = "/home/saleh/webKinPred/media/sequence_info/protT5xl_global"
    PROTT5XL_MODEL_PATH = '/home/saleh/webKinPred/api/UniKP-main/models/prot_t5_xl_uniref50'
    SEQMAP_PY = "/home/saleh/webKinPredEnv/bin/python"
    SEQMAP_CLI = "/home/saleh/webKinPred/tools/seqmap/main.py"
    SEQMAP_DB = "/home/saleh/webKinPred/media/sequence_info/seqmap.sqlite3"
    VOCAB_PATH = '/home/saleh/webKinPred/api/UniKP-main/vocab.pkl'
    TRFM_PATH = '/home/saleh/webKinPred/api/UniKP-main/trfm_12_23000.pkl'
    PREDICTOR_PATH = '/home/saleh/webKinPred/api/UniKP-main/models/UniKP_KCAT.pkl'

    # Test sample
    max_sequence_list = np.random.choice(list("ACDEFGHIKLMNPQRSTVWY"), size=(1000,))
    max_sequence = ''.join(max_sequence_list)# representative long sequence
    sequences = [max_sequence]
    test_smiles = 'CCO'  # ethanol

    # For reproducible memory; remove to see real-world worst case
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    os.environ["NUMEXPR_NUM_THREADS"] = "1"

    # Ensure UniKP code is importable before heavy imports
    if UNIKP_DIR not in sys.path:
        sys.path.insert(0, UNIKP_DIR)

    os.makedirs(SEQ_VEC_DIR, exist_ok=True)

    print(f"Baseline RSS before monitoring: {rss_mb():.2f} MB")

    mon = MemoryMonitor(interval_s=0.005)
    mon.start()
    stage_report = {}

    # Stage 1: imports
    def do_imports():
        global torch, pd, pickle, re, subprocess, WordVocab, TrfmSeq2seq, split
        import torch
        import pandas as pd
        import pickle
        import re
        import subprocess  # noqa: F401
        from build_vocab import WordVocab
        from pretrain_trfm import TrfmSeq2seq
        from utils import split  # noqa: F401
    run_stage("imports", do_imports, mon, stage_report)

    # Stage 2: load SMILES model (vocab + transformer)
    def load_smiles_model():
        global vocab, trfm
        import torch
        vocab = WordVocab.load_vocab(VOCAB_PATH)
        trfm = TrfmSeq2seq(len(vocab), 256, len(vocab), 4)
        sd = torch.load(TRFM_PATH, map_location=torch.device("cpu"))
        trfm.load_state_dict(sd)
        trfm.eval()
    run_stage("smiles_model_load", load_smiles_model, mon, stage_report)

    # Stage 3: decide whether ProtT5-XL is needed; load if yes
    def maybe_load_t5():
        global tokenizer, t5_model, t5_device, need_t5
        ids = resolve_seq_ids_via_cli(sequences, SEQMAP_PY, SEQMAP_CLI, SEQMAP_DB)
        need_t5 = any(not os.path.exists(os.path.join(SEQ_VEC_DIR, f"{sid}.npy")) for sid in ids)
        if need_t5:
            print("ProtT5-XL needed — loading...")
            tokenizer, t5_model, t5_device = load_t5_model_local(PROTT5XL_MODEL_PATH)
        else:
            tokenizer, t5_model, t5_device = None, None, None
            print("All sequences cached — ProtT5-XL not needed.")
    run_stage("maybe_t5_load", maybe_load_t5, mon, stage_report)

    # Stage 4: load UniKP predictor
    def load_predictor():
        global predictor
        import pickle
        with open(PREDICTOR_PATH, "rb") as f:
            predictor = pickle.load(f)
    run_stage("predictor_load", load_predictor, mon, stage_report)

    # Stage 5: end-to-end features + predict
    def do_one_prediction():
        # SMILES features (returns [N, 256] float32)
        smi_vecs = smiles_to_vec([test_smiles], vocab, trfm)

        # Sequence features (returns [N, 1024] float32), generates and caches if missing
        if t5_model is None:
            # Re-resolve IDs and load from cache
            vecs = seq_to_vec(
                sequences,
                tokenizer=None, model=None, device=None,
                seq_vec_dir=SEQ_VEC_DIR,
                seqmap_py=SEQMAP_PY, seqmap_cli=SEQMAP_CLI, seqmap_db=SEQMAP_DB
            )
        else:
            vecs = seq_to_vec(
                sequences,
                tokenizer=tokenizer, model=t5_model, device=t5_device,
                seq_vec_dir=SEQ_VEC_DIR,
                seqmap_py=SEQMAP_PY, seqmap_cli=SEQMAP_CLI, seqmap_db=SEQMAP_DB
            )

        # Concatenate and predict
        features = np.concatenate([smi_vecs.astype(np.float32), vecs.astype(np.float32)], axis=1)
        preds = predictor.predict(features)
        preds = np.power(10.0, preds)
        print(f"Predicted kcat: {float(preds[0]):.4f} s^(-1)")

        # Encourage freeing transient buffers
        del smi_vecs, vecs, features, preds
        gc.collect()

    run_stage("embed_infer_one_sample", do_one_prediction, mon, stage_report)

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
    print(f"\nTOTAL MAX RAM NEEDED (single process, one sample): {total_max_ram_needed:.2f} MB")

    # Return values similar to your other scripts (baseline + peak increase idea)
    # Here we return: (approx model load mem, peak during processing, baseline at processing start)
    # Model load mem ~ end of predictor load minus start baseline; expose as a convenience.
    model_loaded_memory = stage_report["predictor_load"]["rss_after_mb"]
    model_load_memory = model_loaded_memory - stage_report["imports"]["rss_after_mb"]
    processing_peak_inc = stage_report["embed_infer_one_sample"]["stage_peak_inc_mb"]
    processing_baseline = stage_report["embed_infer_one_sample"]["rss_after_mb"] - stage_report["embed_infer_one_sample"]["rss_delta_mb"]
    return model_load_memory, processing_peak_inc, processing_baseline

if __name__ == "__main__":
    # Ensure UniKP code path (for direct execution)
    sys.path.insert(0, '/home/saleh/webKinPred/api/UniKP-main')

    model_mem, peak_proc_mem, baseline_mem = test_unikp_memory()
    total_max_ram = baseline_mem + peak_proc_mem
    print(f"\nFINAL RESULTS:")
    print(f"UniKP - Model loading memory (approx): {model_mem:.2f} MB")
    print(f"UniKP - Peak processing memory increase: {peak_proc_mem:.2f} MB")
    print(f"UniKP - TOTAL MAX RAM NEEDED: {total_max_ram:.2f} MB")
