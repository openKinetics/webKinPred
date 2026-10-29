#!/usr/bin/env python3

import sys
import os
import pandas as pd
import numpy as np
import torch
import esm
import math
from rdkit import Chem
from rdkit.Chem import MACCSkeys
from torch_geometric.data import Data, Batch
import subprocess
import gc
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[3]
_REPO_ROOT_STR = str(_REPO_ROOT)
if _REPO_ROOT_STR in sys.path:
    sys.path.remove(_REPO_ROOT_STR)
sys.path.insert(0, _REPO_ROOT_STR)

from tools.gpu_embed_service.cache_io import (
    SpoolAsyncCommitter,
    remove_manifest_entries,
    resolve_missing_ids,
)

# Adjust the import paths according to your project structure
from KCM import EitlemKcatPredictor
from KMP import EitlemKmPredictor


def _env_bool(name, default=True):
    raw = str(os.environ.get(name, "")).strip().lower()
    if not raw:
        return default
    return raw in {"1", "true", "yes", "on"}

# Use environment variables if available, otherwise fall back to hardcoded paths
ESM_EMB_DIR = (
    os.environ.get("EITLEM_MEDIA_PATH", "/home/saleh/webKinPred/media")
    + "/sequence_info/esm1v"
)
SEQMAP_CLI = (
    os.environ.get("EITLEM_TOOLS_PATH", "/home/saleh/webKinPred/tools")
    + "/seqmap/main.py"
)
SEQMAP_DB = (
    os.environ.get("EITLEM_MEDIA_PATH", "/home/saleh/webKinPred/media")
    + "/sequence_info/seqmap.sqlite3"
)

# For SEQMAP_PY, use the current Python interpreter if in Docker environment, otherwise use local env
if os.environ.get("EITLEM_MEDIA_PATH"):
    # We're in Docker - use current Python interpreter
    SEQMAP_PY = sys.executable
else:
    # Local environment
    SEQMAP_PY = "/home/saleh/webKinPredEnv/bin/python"


def resolve_seq_ids_via_cli(sequences):
    """Call the seqmap CLI once to resolve IDs for all sequences (increments uses_count)."""
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


os.makedirs(ESM_EMB_DIR, exist_ok=True)
DELETE_EMBEDDINGS_AFTER_RUN = _env_bool("EITLEM_DELETE_EMBEDDINGS_AFTER_RUN", default=True)


def _torch_load_compat(path, map_location=None):
    """Prefer weights-only loading when supported; keep legacy fallback."""
    try:
        return torch.load(path, map_location=map_location, weights_only=True)
    except TypeError:
        # torch<2.0 does not support weights_only
        return torch.load(path, map_location=map_location)


def load_esm_model():
    """Load ESM model once and return it."""
    # Determine model path based on environment variables
    from tools.executor.policy import require_cached
    require_cached(False, "protein embeddings must be scheduled")
    if os.environ.get("EITLEM_MEDIA_PATH"):
        # Docker environment
        model_location = "/app/models/EITLEM/Weights/esm1v/esm1v_t33_650M_UR90S_1.pt"
    else:
        # Local environment
        model_location = (
            "/home/saleh/webKinPred/models/EITLEM/Weights/esm1v/esm1v_t33_650M_UR90S_1.pt"
        )

    esm_model, alphabet = esm.pretrained.load_model_and_alphabet_local(
        model_location=model_location
    )
    batch_converter = alphabet.get_batch_converter()
    esm_model.eval()
    return esm_model, alphabet, batch_converter


def get_sequence_embedding(
    sequence,
    seq_id,
    esm_model,
    batch_converter,
    alphabet,
    *,
    cache_dir: Path,
    missing_seq_ids: set[str],
    committer: SpoolAsyncCommitter | None,
    in_memory_cache: dict[str, np.ndarray],
):
    """Get embedding for a single sequence."""
    if seq_id in in_memory_cache:
        return in_memory_cache[seq_id]
    vec_path = cache_dir / f"{seq_id}.npy"
    if seq_id not in missing_seq_ids:
        rep = np.load(vec_path).astype(np.float32, copy=False)
        in_memory_cache[seq_id] = rep
        return rep

    if esm_model is None or batch_converter is None or alphabet is None:
        raise RuntimeError(
            f"Missing ESM1v cache file for seq_id={seq_id} and ESM model is not loaded."
        )

    # Generate embedding - handle long sequences like the original script
    sequence_for_embedding = sequence
    if len(sequence_for_embedding) > 1023:
        # take first 500 + last 500 residues
        sequence_for_embedding = (
            sequence_for_embedding[:500] + sequence_for_embedding[-500:]
        )

    data = [("protein", sequence_for_embedding)]
    _, _, batch_tokens = batch_converter(data)
    batch_lens = (batch_tokens != alphabet.padding_idx).sum(1)
    with torch.no_grad():
        results = esm_model(batch_tokens, repr_layers=[33], return_contacts=False)
    token_representations = results["representations"][33]
    tokens_len = batch_lens[0]
    rep = token_representations[0, 1 : tokens_len - 1].cpu().numpy().astype(np.float32, copy=False)
    if committer is None:
        raise RuntimeError("ESM embedding committer is not available for missing sequence cache write.")
    committer.submit_numpy(cache_dir=cache_dir, seq_id=seq_id, array=rep)
    missing_seq_ids.discard(seq_id)
    in_memory_cache[seq_id] = rep
    return rep


def main():
    seq_ids = []
    run_completed = False
    committer: SpoolAsyncCommitter | None = None
    if len(sys.argv) != 4:
        print(
            "Usage: python eitlem_prediction_script_batch.py input_file.csv output_file.csv kinetics_type"
        )
        sys.exit(1)
    input_file = sys.argv[1]
    output_file = sys.argv[2]
    kinetics_type = sys.argv[3].upper()  # 'KCAT' or 'KM'

    # Load input data
    df_input = pd.read_csv(input_file)
    sequences = df_input["Protein Sequence"].tolist()
    substrates = df_input["Substrate SMILES"].tolist()

    try:
        # Resolve all IDs up-front (counts per occurrence, duplicates included)
        seq_ids = resolve_seq_ids_via_cli(sequences)

        # Define paths to model weights based on environment variables
        if os.environ.get("EITLEM_MEDIA_PATH"):
            # Docker environment
            modelPath = {
                "KCAT": "/app/models/EITLEM/Weights/KCAT/iter8_trainR2_0.9408_devR2_0.7459_RMSE_0.7751_MAE_0.4787",
                "KM": "/app/models/EITLEM/Weights/KM/iter8_trainR2_0.9303_devR2_0.7163_RMSE_0.6960_MAE_0.4802",
            }
        else:
            # Local environment
            modelPath = {
                "KCAT": "/home/saleh/webKinPred/models/EITLEM/Weights/KCAT/iter8_trainR2_0.9408_devR2_0.7459_RMSE_0.7751_MAE_0.4787",
                "KM": "/home/saleh/webKinPred/models/EITLEM/Weights/KM/iter8_trainR2_0.9303_devR2_0.7163_RMSE_0.6960_MAE_0.4802",
            }

        if kinetics_type not in modelPath:
            print(f"Invalid kinetics type: {kinetics_type}")
            sys.exit(1)

        # Load EITLEM model
        if kinetics_type == "KCAT":
            eitlem_model = EitlemKcatPredictor(167, 512, 1280, 10, 0.5, 10)
        elif kinetics_type == "KM":
            eitlem_model = EitlemKmPredictor(167, 512, 1280, 10, 0.5, 10)

        eitlem_model.load_state_dict(
            _torch_load_compat(modelPath[kinetics_type], map_location=torch.device("cpu"))
        )
        eitlem_model.eval()

        cache_dir = Path(ESM_EMB_DIR).resolve()
        cache_dir.mkdir(parents=True, exist_ok=True)
        unique_seq_ids = set(seq_ids)
        missing_list, _ready_ids = resolve_missing_ids(
            unique_seq_ids,
            cache_dir=cache_dir,
            suffix=".npy",
        )
        missing_seq_ids = set(missing_list)
        needs_esm = len(missing_seq_ids) > 0
        if needs_esm:
            esm_model, alphabet, batch_converter = load_esm_model()
        else:
            print("All EITLEM ESM1v embeddings already on disk, skipping ESM model load.")
            esm_model, alphabet, batch_converter = None, None, None

        async_workers = max(
            1,
            int(
                os.environ.get(
                    "EITLEM_CACHE_ASYNC_WORKERS",
                    os.environ.get("GPU_EMBED_CACHE_ASYNC_WORKERS", "4"),
                )
            ),
        )
        spool_dir = Path(os.environ.get("GPU_EMBED_CACHE_SPOOL_DIR", "/dev/shm/webkinpred-gpu-cache"))
        spool_fallback = Path(os.environ.get("GPU_EMBED_CACHE_SPOOL_FALLBACK_DIR", "/tmp/webkinpred-gpu-cache"))
        committer = (
            SpoolAsyncCommitter(
                max_workers=async_workers,
                spool_dir=spool_dir,
                spool_fallback_dir=spool_fallback,
            )
            if needs_esm
            else None
        )
        in_memory_cache: dict[str, np.ndarray] = {}

        predictions = []
        total_predictions = len(sequences)

        # Process predictions one by one
        for idx, (sequence, substrate, seq_id) in enumerate(
            zip(sequences, substrates, seq_ids)
        ):
            try:
                print(f"Progress: {idx+1}/{total_predictions} predictions made", flush=True)

                # Convert substrate SMILES to molecule
                mol = Chem.MolFromSmiles(substrate)
                if mol is None:
                    raise ValueError(f"Invalid substrate SMILES: {substrate}")

                # Compute MACCS Keys
                mol_feature = MACCSkeys.GenMACCSKeys(mol).ToList()

                # Get sequence embedding
                rep = get_sequence_embedding(
                    sequence,
                    seq_id,
                    esm_model,
                    batch_converter,
                    alphabet,
                    cache_dir=cache_dir,
                    missing_seq_ids=missing_seq_ids,
                    committer=committer,
                    in_memory_cache=in_memory_cache,
                )

                sequence_rep = torch.FloatTensor(rep)
                sample = Data(
                    x=torch.FloatTensor(mol_feature).unsqueeze(0), pro_emb=sequence_rep
                )

                input_data = Batch.from_data_list([sample], follow_batch=["pro_emb"])

                # Predict kinetics value
                with torch.no_grad():
                    res = eitlem_model(input_data)
                prediction = math.pow(10, res[0].item())
                predictions.append(prediction)

                # Clean up memory
                del mol_feature, rep, sequence_rep, sample, input_data, res
                gc.collect()

            except Exception as e:

                from tools.executor.policy import propagate_resource_error

                propagate_resource_error(e)
                print(f"Error processing sample {idx}: {e}")
                predictions.append(None)  # Use None for failed predictions

        if committer is not None:
            committer.wait_for_completion()
            committer.close()
            committer = None

        if needs_esm:
            del esm_model, alphabet, batch_converter
            gc.collect()

        # Save predictions to output file
        df_output = pd.DataFrame(
            {
                "Substrate SMILES": substrates,
                "Protein Sequence": sequences,
                "Predicted Value": predictions,
            }
        )
        df_output.to_csv(output_file, index=False)
        run_completed = True
    finally:
        if committer is not None:
            try:
                committer.close()
            except Exception:
                pass
        if DELETE_EMBEDDINGS_AFTER_RUN or not run_completed:
            cleanup_seq_ids = set(seq_ids)
            for seq_id in cleanup_seq_ids:
                try:
                    os.remove(os.path.join(ESM_EMB_DIR, f"{seq_id}.npy"))
                except OSError:
                    pass
            try:
                remove_manifest_entries(Path(ESM_EMB_DIR), cleanup_seq_ids)
            except Exception:
                pass


if __name__ == "__main__":
    main()
