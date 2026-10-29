import sys
import os
import pandas as pd
import torch
import numpy as np
import re
import pickle
import gc
import warnings
from pathlib import Path
from build_vocab import WordVocab
from pretrain_trfm import TrfmSeq2seq
from utils import split
from transformers import T5Tokenizer, T5EncoderModel
from transformers.utils import logging
import subprocess

_REPO_ROOT = Path(__file__).resolve().parents[2]
_REPO_ROOT_STR = str(_REPO_ROOT)
if _REPO_ROOT_STR in sys.path:
    sys.path.remove(_REPO_ROOT_STR)
sys.path.insert(0, _REPO_ROOT_STR)

from tools.gpu_embed_service.cache_io import SpoolAsyncCommitter, resolve_missing_ids

logging.set_verbosity_error()
warnings.filterwarnings(
    "ignore",
    message=r"Trying to unpickle estimator .*",
    category=UserWarning,
)


def _env_int(name, default):
    raw = os.environ.get(name, "")
    if raw is None:
        return default
    raw = str(raw).strip()
    if not raw:
        return default
    try:
        parsed = int(raw)
    except ValueError:
        return default
    return parsed if parsed > 0 else default

# Use environment variables to determine paths
if os.environ.get("UNIKP_MEDIA_PATH"):
    # Docker environment
    SEQ_VEC_DIR = os.environ.get("UNIKP_MEDIA_PATH") + "/sequence_info/prot_t5_last/mean_vecs"
    PROTT5XL_MODEL_PATH = "/app/models/UniKP-main/models/protT5_xl/prot_t5_xl_uniref50"
    SEQMAP_PY = sys.executable  # Use current Python interpreter in Docker
    SEQMAP_CLI = os.environ.get("UNIKP_TOOLS_PATH") + "/seqmap/main.py"
    SEQMAP_DB = os.environ.get("UNIKP_MEDIA_PATH") + "/sequence_info/seqmap.sqlite3"
    VOCAB_PATH = "/app/models/UniKP-main/vocab.pkl"
    TRFM_PATH = "/app/models/UniKP-main/trfm_12_23000.pkl"
else:
    # Local environment
    SEQ_VEC_DIR = "/home/saleh/webKinPred/media/sequence_info/prot_t5_last/mean_vecs"
    PROTT5XL_MODEL_PATH = (
        "/home/saleh/webKinPred/models/UniKP-main/models/protT5_xl/prot_t5_xl_uniref50"
    )
    SEQMAP_PY = "/home/saleh/webKinPredEnv/bin/python"
    SEQMAP_CLI = "/home/saleh/webKinPred/tools/seqmap/main.py"
    SEQMAP_DB = "/home/saleh/webKinPred/media/sequence_info/seqmap.sqlite3"
    VOCAB_PATH = "/home/saleh/webKinPred/models/UniKP-main/vocab.pkl"
    TRFM_PATH = "/home/saleh/webKinPred/models/UniKP-main/trfm_12_23000.pkl"


def load_smiles_model():
    """Load SMILES model once and return components."""
    from tools.executor.policy import cache_only
    if cache_only():
        return None, None
    vocab = WordVocab.load_vocab(VOCAB_PATH)
    trfm = TrfmSeq2seq(len(vocab), 256, len(vocab), 4)
    trfm.load_state_dict(
        _torch_load_compat(TRFM_PATH, map_location=torch.device("cpu"))
    )
    trfm.eval()
    return vocab, trfm


def _torch_load_compat(path, map_location=None):
    """Prefer weights-only loading when supported; keep legacy fallback."""
    try:
        return torch.load(path, map_location=map_location, weights_only=True)
    except TypeError:
        # torch<2.0 does not support weights_only
        return torch.load(path, map_location=map_location)


def smiles_to_vec(Smiles, vocab, trfm):
    """Convert SMILES to vectors using pre-loaded models."""
    from tools.executor.policy import cache_only, molecular_vectors
    if cache_only():
        return molecular_vectors("UniKP", Smiles)
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


def load_t5_model():
    """Load T5 model once and return components."""
    from tools.executor.policy import require_cached
    require_cached(False, "protein embeddings must be scheduled")
    print("Loading T5 model...")
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    try:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        tokenizer = T5Tokenizer.from_pretrained(
            PROTT5XL_MODEL_PATH,
            do_lower_case=False,
            local_files_only=True,
        )
        model_kwargs = {"low_cpu_mem_usage": True}
        # float16 fails on CPU backends ("addmm_impl_cpu_ not implemented for 'Half'")
        # so we keep fp16 only when CUDA is available.
        if device.type == "cuda":
            model_kwargs["torch_dtype"] = torch.float16
        else:
            model_kwargs["torch_dtype"] = torch.float32

        model = T5EncoderModel.from_pretrained(
            PROTT5XL_MODEL_PATH,
            local_files_only=True,
            **model_kwargs,
        )
        model = model.to(device).eval()
        print("T5 model loaded and moved to device.")
        return tokenizer, model, device
    except Exception as e:
        from tools.executor.policy import propagate_resource_error
        propagate_resource_error(e)
        raise RuntimeError(f"Failed to load T5 model: {e}")


def resolve_seq_ids_via_cli(sequences):
    """Resolve IDs for all sequences in order (increments uses_count per occurrence)."""
    payload = "\n".join(sequences) + "\n"
    cmd = [SEQMAP_PY, SEQMAP_CLI, "--db", SEQMAP_DB, "batch-get-or-create", "--stdin"]
    proc = subprocess.run(
        cmd, input=payload, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE
    )
    if proc.returncode != 0:
        raise RuntimeError(
            f"seqmap CLI failed (rc={proc.returncode})\nSTDOUT:\n{proc.stdout}\nSTDERR:\n{proc.stderr}"
        )
    ids = proc.stdout.strip().splitlines()
    if len(ids) != len(sequences):
        raise RuntimeError(
            f"seqmap returned {len(ids)} ids for {len(sequences)} sequences"
        )
    return ids


def seq_to_vec(
    sequences,
    tokenizer=None,
    model=None,
    device=None,
    seq_ids=None,
    batch_size=1,
    planned_missing_ids=None,
):
    """Convert sequences to vectors using pre-loaded T5 model (if provided)."""
    ids = list(seq_ids) if seq_ids is not None else resolve_seq_ids_via_cli(sequences)
    if len(ids) != len(sequences):
        raise RuntimeError(
            f"seq_to_vec got {len(ids)} ids for {len(sequences)} sequences"
        )

    seq_id_to_positions = {}
    seq_id_to_example_seq = {}
    for idx, (seq, seq_id) in enumerate(zip(sequences, ids)):
        seq_id_to_positions.setdefault(seq_id, []).append(idx)
        if seq_id not in seq_id_to_example_seq:
            seq_id_to_example_seq[seq_id] = seq

    cache_dir = Path(SEQ_VEC_DIR).resolve()
    cache_dir.mkdir(parents=True, exist_ok=True)
    vec_by_seq_id = {}
    if planned_missing_ids is None:
        missing_seq_ids, ready_seq_ids = resolve_missing_ids(
            seq_id_to_positions.keys(),
            cache_dir=cache_dir,
            suffix=".npy",
        )
    else:
        missing_seq_ids = []
        for seq_id in planned_missing_ids:
            if seq_id in seq_id_to_positions and seq_id not in missing_seq_ids:
                missing_seq_ids.append(seq_id)
        ready_seq_ids = set(seq_id_to_positions.keys()) - set(missing_seq_ids)

    for seq_id in ready_seq_ids:
        vec_by_seq_id[seq_id] = np.load(cache_dir / f"{seq_id}.npy")

    if missing_seq_ids:
        if tokenizer is None or model is None or device is None:
            raise RuntimeError(
                "T5 model components not provided for embedding generation"
            )

        print(f"Generating embeddings for {len(missing_seq_ids)} sequences...")
        async_workers = max(
            1,
            int(
                os.environ.get(
                    "UNIKP_CACHE_ASYNC_WORKERS",
                    os.environ.get("GPU_EMBED_CACHE_ASYNC_WORKERS", "4"),
                )
            ),
        )
        spool_dir = Path(os.environ.get("GPU_EMBED_CACHE_SPOOL_DIR", "/dev/shm/webkinpred-gpu-cache"))
        spool_fallback = Path(os.environ.get("GPU_EMBED_CACHE_SPOOL_FALLBACK_DIR", "/tmp/webkinpred-gpu-cache"))
        with SpoolAsyncCommitter(
            max_workers=async_workers,
            spool_dir=spool_dir,
            spool_fallback_dir=spool_fallback,
        ) as committer:
            for start in range(0, len(missing_seq_ids), batch_size):
                batch_ids = missing_seq_ids[start : start + batch_size]
                spaced_batch = []
                for sid in batch_ids:
                    spaced = " ".join(seq_id_to_example_seq[sid])
                    spaced_batch.append(re.sub(r"[UZOB]", "X", spaced))

                encoded = tokenizer.batch_encode_plus(
                    spaced_batch,
                    add_special_tokens=True,
                    padding=True,
                    return_tensors="pt",
                )
                input_ids = encoded["input_ids"].to(device)
                attention_mask = encoded["attention_mask"].to(device)

                with torch.no_grad():
                    embedding = model(
                        input_ids=input_ids, attention_mask=attention_mask
                    ).last_hidden_state

                embedding_np = embedding.float().cpu().numpy()
                lengths = attention_mask.sum(dim=1).cpu().numpy()

                for row_idx, sid in enumerate(batch_ids):
                    seq_len = int(lengths[row_idx])
                    token_count = max(seq_len - 1, 1)
                    seq_vec = embedding_np[row_idx, :token_count].mean(axis=0).astype(np.float32, copy=False)
                    committer.submit_numpy(cache_dir=cache_dir, seq_id=sid, array=seq_vec)
                    vec_by_seq_id[sid] = seq_vec

    vecs = []
    for seq_id in ids:
        vec = vec_by_seq_id.get(seq_id)
        if vec is None:
            raise RuntimeError(f"Missing sequence vector for seq_id={seq_id}")
        vecs.append(vec)

    return np.stack(vecs)


def _predict_batch(model, smiles_batch, seq_vec_batch, vocab, trfm):
    smiles_vecs = smiles_to_vec(smiles_batch, vocab, trfm)
    features = np.concatenate([smiles_vecs, seq_vec_batch], axis=1)
    preds = model.predict(features)
    return np.power(10, preds)


def main(input_path, output_path, task_type):
    df = pd.read_csv(input_path)
    sequences = df["Protein Sequence"].tolist()
    smiles = df["Substrate SMILES"].tolist()
    total_predictions = len(sequences)

    # Tunable throughput controls (safe defaults).
    pred_batch_size = _env_int("UNIKP_PRED_BATCH_SIZE", 128)

    # Load SMILES model once
    print("Loading SMILES model...")
    vocab, trfm = load_smiles_model()

    # Check if we need T5 model (if any sequences need embedding)
    all_seq_ids = resolve_seq_ids_via_cli(sequences)
    planned_missing_ids, _ready_ids = resolve_missing_ids(
        all_seq_ids,
        cache_dir=Path(SEQ_VEC_DIR).resolve(),
        suffix=".npy",
    )
    need_t5_model = len(planned_missing_ids) > 0

    # Load T5 model once if needed
    tokenizer, t5_model, device = None, None, None
    if need_t5_model:
        tokenizer, t5_model, device = load_t5_model()

    # Load trained model once
    print("Loading prediction model...")
    if os.environ.get("UNIKP_MEDIA_PATH"):
        # Docker environment
        model_path = f"/app/models/UniKP-main/models/UniKP_{task_type}.pkl"
    else:
        # Local environment
        model_path = (
            f"/home/saleh/webKinPred/models/UniKP-main/models/UniKP_{task_type}.pkl"
        )
    with open(model_path, "rb") as f:
        model = pickle.load(f)
    from tools.executor.policy import limit_estimator
    model = limit_estimator(model)

    if total_predictions:
        # Sequence vectors are loaded/computed once for all rows.
        sequence_vecs = seq_to_vec(
            sequences,
            tokenizer=tokenizer,
            model=t5_model,
            device=device,
            seq_ids=all_seq_ids,
            batch_size=_env_int("UNIKP_T5_BATCH_SIZE", 1),
            planned_missing_ids=planned_missing_ids,
        )
        predictions = [None] * total_predictions

        for start in range(0, total_predictions, pred_batch_size):
            end = min(start + pred_batch_size, total_predictions)
            try:
                batch_preds = _predict_batch(
                    model=model,
                    smiles_batch=smiles[start:end],
                    seq_vec_batch=sequence_vecs[start:end],
                    vocab=vocab,
                    trfm=trfm,
                )
                predictions[start:end] = batch_preds.tolist()
            except Exception as batch_error:
                from tools.executor.policy import propagate_resource_error
                propagate_resource_error(batch_error)
                print(f"Batch {start + 1}-{end} failed: {batch_error}")
                # Keep per-row fallback so one bad sample does not fail the whole job.
                for row_idx in range(start, end):
                    try:
                        single_pred = _predict_batch(
                            model=model,
                            smiles_batch=[smiles[row_idx]],
                            seq_vec_batch=sequence_vecs[row_idx : row_idx + 1],
                            vocab=vocab,
                            trfm=trfm,
                        )
                        predictions[row_idx] = float(single_pred[0])
                    except Exception as row_error:
                        from tools.executor.policy import propagate_resource_error
                        propagate_resource_error(row_error)
                        print(f"Error processing sample {row_idx}: {row_error}")
                        predictions[row_idx] = None

            print(f"Progress: {end}/{total_predictions} predictions made", flush=True)
    else:
        predictions = []

    # Clean up models
    del vocab, trfm
    if tokenizer is not None:
        del tokenizer, t5_model
    gc.collect()

    # Output - same format as original
    df_out = pd.DataFrame({"Predicted Value": predictions})
    df_out.to_csv(output_path, index=False)


if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("Usage: python run_unikp.py <input_csv> <output_csv> <task_type>")
        sys.exit(1)

    input_csv = sys.argv[1]
    output_csv = sys.argv[2]
    task = sys.argv[3].upper()  #
    main(input_csv, output_csv, task)
