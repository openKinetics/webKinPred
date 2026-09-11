import pandas as pd
import numpy as np
import os
import sys
import tempfile
import time
from pathlib import Path
from typing import Dict, Optional
from tqdm import tqdm
import torch
import pickle
import gc

_REPO_ROOT = Path(__file__).resolve().parents[4]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from tools.gpu_embed_service.kinform_stream_ipc import StreamClient
# --------------------------------------------------------------------------- #
#                              HELPER FUNCTIONS                               #
# --------------------------------------------------------------------------- #
def _fetch_weights(seq_id: str, df: pd.DataFrame, key_col: str, weights_col: str) -> np.ndarray:
    """
    Return a 1-D float64 array of per-residue weights for `seq_id`.
    Raises if the sequence is missing.
    """
    row = df.loc[df[key_col] == seq_id, weights_col]
    if row.empty:
        raise ValueError(f"No weights found in {weights_col} for sequence {seq_id}")
    return np.fromiter((float(x) for x in row.iloc[0].split(",")), dtype=float)


def _truncate_first_last_array(arr: np.ndarray, keep: int = 1024) -> np.ndarray:
    if arr.shape[0] <= keep:
        return arr
    half = keep // 2
    return np.concatenate([arr[:half], arr[-half:]], axis=0)


def _align_residue_to_binding_weights(
    seq_id: str,
    residue_embedding: np.ndarray,
    weights: np.ndarray,
) -> np.ndarray:
    if residue_embedding.shape[0] == weights.shape[0]:
        return residue_embedding
    if weights.shape[0] == 1024 and residue_embedding.shape[0] > 1024:
        return _truncate_first_last_array(residue_embedding, keep=1024)
    raise ValueError(
        f"Weight length ({weights.shape[0]}) does not match embedding length "
        f"({residue_embedding.shape[0]}) for {seq_id}"
    )


def _weighted_mean(arr: np.ndarray, w: np.ndarray, normalize: bool = True) -> np.ndarray:
    """Length‑L weights → weighted mean over axis‑0."""
    w = np.asarray(w, dtype=float)
    if normalize:
        w = w / w.sum()
    return (arr * w[:, None]).sum(axis=0)


def _require_cuda_if_requested(context: str) -> None:
    raw = str(os.environ.get("KINFORM_REQUIRE_CUDA", "")).strip().lower()
    require_cuda = raw in {"1", "true", "yes", "on"}
    if require_cuda and not torch.cuda.is_available():
        raise RuntimeError(
            f"CUDA is required for {context} (KINFORM_REQUIRE_CUDA=1), "
            "but no CUDA device is available."
        )


def _save_array_atomic(path: Path, arr: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=f".{path.stem}.", suffix=".npy", dir=str(path.parent))
    os.close(fd)
    tmp_path = Path(tmp_name)
    try:
        np.save(tmp_path, arr)
        os.replace(tmp_path, path)
    finally:
        if tmp_path.exists():
            tmp_path.unlink(missing_ok=True)


def _cleanup_residue_files(residue_dir: Path, keys: list[str]) -> None:
    for key in keys:
        try:
            (residue_dir / f"{key}.npy").unlink(missing_ok=True)
        except OSError:
            pass


def get_embeddings(seq_dict, batch_size=2, model=None, id_to_seq=None, setting='mean',all_layers=False, only_save=False, layer=None, weights_df: Optional[pd.DataFrame] = None, weights_key_col: str = "PDB", weights_col: str = "Pred_BS_Scores"):
    _require_cuda_if_requested(f"{model} embedding")
    # Parse setting - can be combination like "mean+weighted"
    settings = set(s.strip() for s in setting.split("+"))
    valid_settings = {"mean", "residue", "weighted"}
    assert settings.issubset(valid_settings), f"Invalid setting: {setting}. Valid: {valid_settings}"
    """
    Get ESM-2 embeddings for sequences in seq_dict.
    input: seq_dict: dictionary with sequence IDs as keys and sequences as values
            batch_size: batch size for processing sequences
    output: dictionary with sequence IDs as keys and embeddings as values
    """
    accepted_models = ['esm2', 'esm1v', 'esmc']
    assert model in accepted_models, f"Invalid model: {model}. Accepted models: {accepted_models}"
    if "weighted" in settings:
        assert weights_df is not None, "--weights_file is required when setting includes 'weighted'"
        assert not all_layers, "weighted setting is incompatible with --all_layers"
    

    precomputed_root = Path(os.environ.get("KINFORM_MEDIA_PATH")) / "sequence_info" 
    # Get repository root relative to this file
    ROOT = Path(__file__).resolve().parent.parent.parent
    assert all([key in id_to_seq.keys() and id_to_seq[key] == value for key, value in seq_dict.items()]), "Sequences must be in id_to_seq dictionary"
    print(f"Loaded {len(seq_dict)} sequences")

    # Set up paths based on setting and layer
    paths = {}
    
    # Set up paths for each requested setting
    if "residue" in settings or all_layers:
        raise NotImplementedError("Residue embeddings extraction not implemented in this snippet.")

    if "mean" in settings:
        if layer is None:
            paths["mean"] = precomputed_root / f"{model}_last/mean_vecs"
        else:
            paths["mean"] = precomputed_root / f"{model}_layer_{layer}/mean_vecs"

    if "weighted" in settings:
        if layer is None:
            paths["weighted"] = precomputed_root / f"{model}_last/weighted_vecs"
        else:
            paths["weighted"] = precomputed_root / f"{model}_layer_{layer}/weighted_vecs"
    
    for p in paths.values():
        p.mkdir(parents=True, exist_ok=True)

    # ------------------------ skip existing files ------------------------- #
    if all_layers:
        key_to_exist = {
            key: (paths["all_layers"] / f"{key}.npy").exists()
            for key in seq_dict
        }
    else:
        # Check if all requested settings already exist for all sequences
        key_to_exist = {
            key: all((paths[s] / f"{key}.npy").exists() for s in settings)
            for key in seq_dict
        }

    if all(key_to_exist.values()):
        print(f"Skipping {model} model loading, all embeddings already exist")
        if not only_save:
            if len(settings) == 1:
                # Single setting - return flat dict
                setting_name = next(iter(settings))
                embeddings = {
                    key: np.load(paths[setting_name] / f"{key}.npy")
                    for key in seq_dict
                }
            else:
                # Multiple settings - return nested dict
                embeddings = {
                    key: {
                        s: np.load(paths[s] / f"{key}.npy")
                        for s in settings
                    }
                    for key in seq_dict
                }
            return embeddings
    else:
        import esm
        torch.cuda.empty_cache()
        not_exist = [key for key, value in key_to_exist.items() if not value]
        print(f"Generating {model} embeddings for {len(not_exist)} sequences")
        print(F"Loading {model} model...")#
        if model == 'esm1v':
            assert not all_layers, "esm1v model does not support all_layers=True"
            model, alphabet = esm.pretrained.esm1v_t33_650M_UR90S_1()
            seq_dict = {
                k: (v[:500] + v[-500:] if len(v) > 1022 else v)
                for k, v in seq_dict.items()
            }
        elif model == 'esm2':
            model, alphabet = esm.pretrained.esm2_t33_650M_UR50D()
            batch_converter = alphabet.get_batch_converter()
            model.eval()
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            model = model.to(device)
            print(f"Using device: {device}")
            print(f"Loaded model with {sum(p.numel() for p in model.parameters()) // 1e6} million parameters")

            keys = list(seq_dict.keys())
            keys = [key for key in keys if not key_to_exist[key]]
            print(f"Skipping {len(seq_dict) - len(keys)} sequences with existing embeddings")

            batches = [keys[i:i + batch_size] for i in range(0, len(keys), batch_size)]

            for batch_keys in tqdm(batches):
                batch = {key: seq_dict[key] for key in batch_keys}
                data = [(label, seq) for label, seq in batch.items()]
                batch_labels, batch_strs, batch_tokens = batch_converter(data)
                batch_tokens = batch_tokens.to(device)

                with torch.inference_mode():
                    if all_layers:
                        n_layers = 34
                        results = model(batch_tokens, repr_layers=list(range(n_layers)), return_contacts=False)
                        token_reps = results["representations"]
                        for i, (label, seq) in enumerate(data):
                            all_layer_means = [
                                token_reps[layer][i, 1:len(seq) + 1].mean(dim=0).cpu().numpy()
                                for layer in range(n_layers)
                            ]
                            all_layer_means = np.stack(all_layer_means)  # [n_layers, H]
                            np.save(paths["all_layers"] / f"{label}.npy", all_layer_means)
                    else:
                        assert layer is not None, "Layer must be specified when all_layers is False"
                        results = model(batch_tokens, repr_layers=[layer], return_contacts=False)
                        token_rep = results["representations"][layer]
                        for i, (label, seq) in enumerate(data):
                            res_emb = token_rep[i, 1:len(seq) + 1].cpu().numpy()
                            
                            # Save residue embeddings if requested
                            if "residue" in settings:
                                np.save(paths['residue'] / f'{label}.npy', res_emb)
                            
                            # Compute and save mean if requested
                            if "mean" in settings:
                                mean_emb = res_emb.mean(0)
                                np.save(paths['mean'] / f'{label}.npy', mean_emb)
                            
                            # Compute and save weighted if requested
                            if "weighted" in settings:
                                # Fetch weights for this sequence
                                weights = _fetch_weights(label, weights_df, weights_key_col, weights_col)
                                
                                weighted_res_emb = _align_residue_to_binding_weights(label, res_emb, weights)
                                
                                # Compute weighted mean
                                weighted_emb = _weighted_mean(weighted_res_emb, weights, normalize=True)
                                np.save(paths['weighted'] / f'{label}.npy', weighted_emb)

                del data, batch_tokens, results
                gc.collect()
                torch.cuda.empty_cache()

            if not only_save:
                if all_layers:
                    return {
                        key: np.load(paths["all_layers"] / f"{key}.npy")
                        for key in seq_dict
                    }
                elif len(settings) == 1:
                    # Single setting - return flat dict
                    setting_name = next(iter(settings))
                    return {
                        key: np.load(paths[setting_name] / f"{key}.npy")
                        for key in seq_dict
                    }
                else:
                    # Multiple settings - return nested dict
                    return {
                        key: {
                            s: np.load(paths[s] / f"{key}.npy")
                            for s in settings
                        }
                        for key in seq_dict
                    }
            else:
                return None


        elif model == 'esmc':
            from esm.models.esmc import ESMC
            from esm.sdk.api import ESMProtein, LogitsConfig

            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            model = ESMC.from_pretrained("esmc_600m").to(device)
            model.eval()
            config = LogitsConfig(sequence=True, return_hidden_states=True, return_embeddings=True)

            if all_layers:
                assert layer is None, "Layer argument is not used when all_layers=True"
            else:
                assert layer is not None, "Layer argument is required when all_layers=False"
            keys = list(seq_dict.keys())
            keys = [key for key in keys if not key_to_exist[key]]
            for key in tqdm(keys):  # keys already defined as sequences needing embedding
                sequence = seq_dict[key]
                protein = ESMProtein(sequence=sequence)
                # try:
                tensor = model.encode(protein)
                logits_out = model.logits(tensor, config)
                if all_layers:
                    hidden_states = logits_out.hidden_states  # [36, 1, L+2, 1152]
                    all_layer_embs = []
                    for layer_emb in hidden_states:
                        layer_emb = layer_emb.squeeze(0).to(torch.float32).cpu().numpy()  # [L+2, H]
                        cls_emb = layer_emb[0]
                        mean_emb = layer_emb[1:-1].mean(0)
                        all_layer_embs.append((cls_emb, mean_emb))
                    np.save(paths['all_layers'] / f'{key}.npy', all_layer_embs)
                else:
                    layer_emb = logits_out.hidden_states[layer]  # tensor of shape [1, L+2, H]
                    layer_emb = layer_emb.squeeze(0).to(torch.float32).cpu().numpy()  # → [L+2, H]
                    # Remove CLS and END
                    residue_emb = layer_emb[1:-1]  # → [L, H]
                    
                    # Save residue embeddings if requested
                    if "residue" in settings:
                        np.save(paths['residue'] / f'{key}.npy', residue_emb)
                    
                    # Compute and save mean if requested
                    if "mean" in settings:
                        mean_emb = residue_emb.mean(0)
                        np.save(paths['mean'] / f'{key}.npy', mean_emb)
                    
                    # Compute and save weighted if requested
                    if "weighted" in settings:
                        # Fetch weights for this sequence
                        weights = _fetch_weights(key, weights_df, weights_key_col, weights_col)
                        
                        weighted_residue_emb = _align_residue_to_binding_weights(key, residue_emb, weights)
                        # Compute weighted mean
                        weighted_emb = _weighted_mean(weighted_residue_emb, weights, normalize=True)
                        np.save(paths['weighted'] / f'{key}.npy', weighted_emb)

                #     print(f"Error processing {key}: {e}")
            if not only_save:
                if all_layers:
                    return {
                        key: np.load(paths["all_layers"] / f"{key}.npy")
                        for key in seq_dict
                    }
                elif len(settings) == 1:
                    # Single setting - return flat dict
                    setting_name = next(iter(settings))
                    return {
                        key: np.load(paths[setting_name] / f"{key}.npy")
                        for key in seq_dict
                    }
                else:
                    # Multiple settings - return nested dict
                    return {
                        key: {
                            s: np.load(paths[s] / f"{key}.npy")
                            for s in settings
                        }
                        for key in seq_dict
                    }
            else:
                return None
        else:
            raise ValueError(f"Invalid model: {model}. Accepted models: {accepted_models}")


def get_embeddings_multi_layer(
    seq_dict, layers, batch_size=2, model=None, id_to_seq=None,
    setting='mean', only_save=False,
    weights_df=None, weights_key_col="PDB", weights_col="Pred_BS_Scores"
):
    """Load *model* once and extract embeddings for all *layers* in one pass.

    For ESM2:  a single forward call with repr_layers=layers extracts all layers.
    For ESM-C: a single logits() call returns all hidden_states; we index per layer.
    """
    from collections import defaultdict

    settings = set(s.strip() for s in setting.split("+"))
    valid_settings = {"mean", "residue", "weighted"}
    assert settings.issubset(valid_settings), f"Invalid setting: {setting}"
    accepted_models = ['esm2', 'esmc']
    assert model in accepted_models, f"get_embeddings_multi_layer supports {accepted_models}"
    if "weighted" in settings:
        assert weights_df is not None, "--weights_file is required when setting includes 'weighted'"
    assert all(key in id_to_seq and id_to_seq[key] == value for key, value in seq_dict.items())

    precomputed_root = Path(os.environ.get("KINFORM_MEDIA_PATH")) / "sequence_info"

    # Build output directories for each (layer, setting) combination
    layer_paths = {}
    for layer in layers:
        prefix = f"{model}_last" if layer is None else f"{model}_layer_{layer}"
        paths = {}
        for s in settings:
            d = precomputed_root / prefix / f"{s}_vecs"
            d.mkdir(parents=True, exist_ok=True)
            paths[s] = d
        layer_paths[layer] = paths

    # Determine which keys still need at least one (layer, setting) computed
    key_to_exist = {
        key: all(
            all((layer_paths[layer][s] / f"{key}.npy").exists() for s in settings)
            for layer in layers
        )
        for key in seq_dict
    }

    if all(key_to_exist.values()):
        print(f"Skipping {model} model loading, all embeddings already exist")
        if only_save:
            return None
        # Return nested dict: key -> layer -> {setting -> array}
        result = {}
        for key in seq_dict:
            result[key] = {
                layer: {s: np.load(layer_paths[layer][s] / f"{key}.npy") for s in settings}
                for layer in layers
            }
        return result

    missing_keys = [k for k, ok in key_to_exist.items() if not ok]
    print(f"Generating {model} embeddings for {len(missing_keys)} sequence(s), layers {layers}")
    print(f"Loading {model} model...")

    import esm
    torch.cuda.empty_cache()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if model == 'esm2':
        model_obj, alphabet = esm.pretrained.esm2_t33_650M_UR50D()
        batch_converter = alphabet.get_batch_converter()
        model_obj.eval()
        model_obj = model_obj.to(device)
        print(f"Using device: {device}")

        batches = [missing_keys[i:i + batch_size] for i in range(0, len(missing_keys), batch_size)]
        for batch_keys in tqdm(batches, desc=f"ESM2 layers {layers}"):
            data = [(k, seq_dict[k]) for k in batch_keys]
            _, _, batch_tokens = batch_converter(data)
            batch_tokens = batch_tokens.to(device)
            with torch.inference_mode():
                results = model_obj(batch_tokens, repr_layers=layers, return_contacts=False)
            token_reps = results["representations"]
            for i, (label, seq) in enumerate(data):
                for layer in layers:
                    paths = layer_paths[layer]
                    if all((paths[s] / f"{label}.npy").exists() for s in settings):
                        continue
                    res_emb = token_reps[layer][i, 1:len(seq) + 1].cpu().numpy()
                    if "residue" in settings:
                        np.save(paths["residue"] / f"{label}.npy", res_emb)
                    if "mean" in settings:
                        np.save(paths["mean"] / f"{label}.npy", res_emb.mean(0))
                    if "weighted" in settings:
                        weights = _fetch_weights(label, weights_df, weights_key_col, weights_col)
                        weighted_res_emb = _align_residue_to_binding_weights(label, res_emb, weights)
                        np.save(paths["weighted"] / f"{label}.npy", _weighted_mean(weighted_res_emb, weights, normalize=True))
            del batch_tokens, results
            gc.collect()
            torch.cuda.empty_cache()
        del model_obj
        gc.collect()

    elif model == 'esmc':
        from esm.models.esmc import ESMC
        from esm.sdk.api import ESMProtein, LogitsConfig

        model_obj = ESMC.from_pretrained("esmc_600m").to(device)
        model_obj.eval()
        config = LogitsConfig(sequence=True, return_hidden_states=True, return_embeddings=True)

        for key in tqdm(missing_keys, desc=f"ESM-C layers {layers}"):
            sequence = seq_dict[key]
            protein = ESMProtein(sequence=sequence)
            tensor = model_obj.encode(protein)
            logits_out = model_obj.logits(tensor, config)
            for layer in layers:
                paths = layer_paths[layer]
                if all((paths[s] / f"{key}.npy").exists() for s in settings):
                    continue
                layer_emb = logits_out.hidden_states[layer]  # [1, L+2, H]
                layer_emb = layer_emb.squeeze(0).to(torch.float32).cpu().numpy()
                residue_emb = layer_emb[1:-1]  # [L, H]
                if "residue" in settings:
                    np.save(paths["residue"] / f"{key}.npy", residue_emb)
                if "mean" in settings:
                    np.save(paths["mean"] / f"{key}.npy", residue_emb.mean(0))
                if "weighted" in settings:
                    weights = _fetch_weights(key, weights_df, weights_key_col, weights_col)
                    weighted_residue_emb = _align_residue_to_binding_weights(key, residue_emb, weights)
                    np.save(paths["weighted"] / f"{key}.npy", _weighted_mean(weighted_residue_emb, weights, normalize=True))
        del model_obj
        gc.collect()

    if only_save:
        return None
    result = {}
    for key in seq_dict:
        result[key] = {
            layer: {s: np.load(layer_paths[layer][s] / f"{key}.npy") for s in settings}
            for layer in layers
        }
    return result


class _BackgroundWriter:
    """Drains (path, arr) write jobs on a background thread so inference never blocks on SSHFS."""

    def __init__(self) -> None:
        import queue as _q
        import threading as _t
        self._q: _q.Queue = _q.Queue()
        self._exc: Optional[BaseException] = None
        self._thread = _t.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def _loop(self) -> None:
        while True:
            item = self._q.get()
            if item is None:
                return
            path, arr = item
            try:
                _save_array_atomic(path, arr)
            except Exception as exc:
                self._exc = exc

    def submit(self, path: Path, arr: np.ndarray) -> None:
        self._q.put((path, arr.copy()))

    def join(self) -> None:
        self._q.put(None)
        self._thread.join()
        if self._exc is not None:
            raise self._exc


def stream_embeddings_multi_layer(
    seq_dict,
    *,
    layers,
    batch_size,
    model,
    id_to_seq,
    stream_socket,
    stream_job_id,
    worker_name,
    legacy_residue_write=False,
):
    _require_cuda_if_requested(f"{model} stream embedding")
    assert model in {"esm2", "esmc"}, "stream mode currently supports esm2/esmc only"
    assert all(key in id_to_seq and id_to_seq[key] == value for key, value in seq_dict.items())
    stage_name = "esm_residue" if model == "esm2" else "esmc_residue"
    stage_started_at = time.monotonic()
    show_progress = str(os.environ.get("KINFORM_STREAM_TQDM", "")).strip().lower() in {"1", "true", "yes", "on"}
    write_mean_files = str(os.environ.get("KINFORM_STREAM_WRITE_MEAN_FILES", "1")).strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }
    items_total = len(seq_dict) * len(layers)
    emitted_items = 0
    last_progress_at = stage_started_at
    progress_interval_seconds = 10.0

    precomputed_root = Path(os.environ.get("KINFORM_MEDIA_PATH")) / "sequence_info"
    layer_mean_dirs = {}
    layer_residue_dirs = {}
    layer_root_names = {}
    for layer in layers:
        root = f"{model}_last" if layer is None else f"{model}_layer_{layer}"
        layer_root_names[layer] = root
        if write_mean_files:
            mean_dir = precomputed_root / root / "mean_vecs"
            mean_dir.mkdir(parents=True, exist_ok=True)
            layer_mean_dirs[layer] = mean_dir
        if legacy_residue_write:
            residue_dir = precomputed_root / root / "residue_vecs"
            residue_dir.mkdir(parents=True, exist_ok=True)
            layer_residue_dirs[layer] = residue_dir

    stream = StreamClient(stream_socket)
    bg_writer = _BackgroundWriter() if (write_mean_files or legacy_residue_write) else None
    try:
        import esm
        model_load_started_at = time.monotonic()
        torch.cuda.empty_cache()
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        if model == "esm2":
            model_obj, alphabet = esm.pretrained.esm2_t33_650M_UR50D()
            batch_converter = alphabet.get_batch_converter()
            model_obj.eval()
            model_obj = model_obj.to(device)
            print(f"Using device: {device}")
            print(
                f"KINFORM_TIMING stage={stage_name} model_load_s={max(0.0, time.monotonic() - model_load_started_at):.3f} "
                f"seq_count={len(seq_dict)} layer_count={len(layers)}"
            )

            keys = list(seq_dict.keys())
            batches = [keys[i:i + batch_size] for i in range(0, len(keys), batch_size)]
            for batch_keys in tqdm(
                batches,
                desc=f"ESM2 stream layers {layers}",
                disable=not show_progress,
            ):
                data = [(k, seq_dict[k]) for k in batch_keys]
                _, _, batch_tokens = batch_converter(data)
                batch_tokens = batch_tokens.to(device)
                with torch.inference_mode():
                    results = model_obj(batch_tokens, repr_layers=layers, return_contacts=False)
                token_reps = results["representations"]
                for i, (label, seq) in enumerate(data):
                    for layer in layers:
                        residue_emb = token_reps[layer][i, 1:len(seq) + 1].cpu().numpy().astype(np.float32, copy=False)
                        if write_mean_files:
                            mean_vec = residue_emb.mean(axis=0).astype(np.float32, copy=False)
                            assert bg_writer is not None
                            bg_writer.submit(layer_mean_dirs[layer] / f"{label}.npy", mean_vec)
                        if legacy_residue_write:
                            assert bg_writer is not None
                            bg_writer.submit(layer_residue_dirs[layer] / f"{label}.npy", residue_emb)
                        header = {
                            "type": "RESIDUE_READY",
                            "worker": worker_name,
                            "job_id": stream_job_id,
                            "family": "esm2",
                            "root": layer_root_names[layer],
                            "seq_id": label,
                            "sequence": seq,
                            "dtype": "float32",
                            "shape": [int(x) for x in residue_emb.shape],
                        }
                        stream.send(header, residue_emb.tobytes(order="C"))
                        emitted_items += 1
                del batch_tokens, results
                now = time.monotonic()
                if now - last_progress_at >= progress_interval_seconds:
                    elapsed_s = max(0.0, now - stage_started_at)
                    rate = (emitted_items / elapsed_s) if elapsed_s > 0 else 0.0
                    print(
                        f"KINFORM_TIMING stage={stage_name} progress items={emitted_items}/{items_total} "
                        f"elapsed_s={elapsed_s:.3f} rate_items_per_s={rate:.3f}"
                    )
                    last_progress_at = now
            del model_obj
            gc.collect()
        else:
            from esm.models.esmc import ESMC
            from esm.sdk.api import ESMProtein, LogitsConfig

            model_obj = ESMC.from_pretrained("esmc_600m").to(device)
            model_obj.eval()
            config = LogitsConfig(sequence=True, return_hidden_states=True, return_embeddings=True)
            print(
                f"KINFORM_TIMING stage={stage_name} model_load_s={max(0.0, time.monotonic() - model_load_started_at):.3f} "
                f"seq_count={len(seq_dict)} layer_count={len(layers)}"
            )

            for key in tqdm(
                list(seq_dict.keys()),
                desc=f"ESM-C stream layers {layers}",
                disable=not show_progress,
            ):
                sequence = seq_dict[key]
                protein = ESMProtein(sequence=sequence)
                tensor = model_obj.encode(protein)
                logits_out = model_obj.logits(tensor, config)
                for layer in layers:
                    layer_emb = logits_out.hidden_states[layer]
                    layer_emb = layer_emb.squeeze(0).to(torch.float32).cpu().numpy()
                    residue_emb = layer_emb[1:-1].astype(np.float32, copy=False)
                    if write_mean_files:
                        mean_vec = residue_emb.mean(axis=0).astype(np.float32, copy=False)
                        assert bg_writer is not None
                        bg_writer.submit(layer_mean_dirs[layer] / f"{key}.npy", mean_vec)
                    if legacy_residue_write:
                        assert bg_writer is not None
                        bg_writer.submit(layer_residue_dirs[layer] / f"{key}.npy", residue_emb)
                    header = {
                        "type": "RESIDUE_READY",
                        "worker": worker_name,
                        "job_id": stream_job_id,
                        "family": "esmc",
                        "root": layer_root_names[layer],
                        "seq_id": key,
                        "sequence": sequence,
                        "dtype": "float32",
                        "shape": [int(x) for x in residue_emb.shape],
                    }
                    stream.send(header, residue_emb.tobytes(order="C"))
                    emitted_items += 1
                now = time.monotonic()
                if now - last_progress_at >= progress_interval_seconds:
                    elapsed_s = max(0.0, now - stage_started_at)
                    rate = (emitted_items / elapsed_s) if elapsed_s > 0 else 0.0
                    print(
                        f"KINFORM_TIMING stage={stage_name} progress items={emitted_items}/{items_total} "
                        f"elapsed_s={elapsed_s:.3f} rate_items_per_s={rate:.3f}"
                    )
                    last_progress_at = now
            del model_obj
            gc.collect()

        if bg_writer is not None:
            bg_writer.join()
        if legacy_residue_write:
            keys = list(seq_dict.keys())
            for residue_dir in layer_residue_dirs.values():
                _cleanup_residue_files(residue_dir, keys)
        stream.send(
            {
                "type": "WORKER_DONE",
                "worker": worker_name,
                "job_id": stream_job_id,
            },
            b"",
        )
        total_elapsed_s = max(0.0, time.monotonic() - stage_started_at)
        rate = (emitted_items / total_elapsed_s) if total_elapsed_s > 0 else 0.0
        print(
            f"KINFORM_TIMING stage={stage_name} summary items={emitted_items}/{items_total} "
            f"elapsed_s={total_elapsed_s:.3f} rate_items_per_s={rate:.3f}"
        )
    except Exception as exc:
        try:
            stream.send(
                {
                    "type": "WORKER_ERROR",
                    "worker": worker_name,
                    "job_id": stream_job_id,
                    "message": str(exc),
                },
                b"",
            )
        except Exception:
            pass
        raise
    finally:
        stream.close()


if __name__ == '__main__':
    """
    Extract ESM embeddings for unique sequences.
    Default: Runs for ESM2 layers 26 and 29, and ESMC layers 24 and 32.

    Usage:
        # Default behavior (residue embeddings)
        python prot_embeddings.py

        # Mean embeddings
        python prot_embeddings.py --setting mean

        # Weighted embeddings using binding site scores
        python prot_embeddings.py --setting weighted --weights_file path/to/binding_sites.tsv

        # Custom sequence file and models
        python prot_embeddings.py --seq_file path/to/sequences.txt --models esm2 esmc --layers 26 29 24 32
    """
    import argparse
    from pathlib import Path
    
    parser = argparse.ArgumentParser(
        description='Extract ESM protein embeddings',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Default: residue embeddings for ESM2 layers 26,29 and ESMC layers 24,32
  python prot_embeddings.py
  
  # Mean embeddings
  python prot_embeddings.py --setting mean --models esm2 esmc --layers 26 29 24 32
  
  # Weighted embeddings using binding site predictions
  python prot_embeddings.py --setting weighted --weights_file results/binding_sites/binding_sites_all.tsv
  
  # Both mean and weighted (computes residue once, derives both)
  python prot_embeddings.py --setting mean+weighted --weights_file results/binding_sites/binding_sites_all.tsv
  
  # All three: residue, mean, and weighted
  python prot_embeddings.py --setting residue+mean+weighted --weights_file results/binding_sites/binding_sites_all.tsv
  
  # Custom sequence file with specific models and layers
  python prot_embeddings.py --seq_file my_sequences.txt --models esm2 --layers 26 29
        """
    )
    
    parser.add_argument(
        '--seq_file',
        type=str,
        default=None,
        help='Path to text file containing sequence IDs (one per line). Default: data/unique_seq_ids.txt'
    )
    
    parser.add_argument(
        '--models',
        type=str,
        nargs='+',
        default=None,
        choices=['esm2', 'esmc', 'esm1v'],
        help='Model(s) to use for embeddings. Default: esm2 and esmc'
    )
    
    parser.add_argument(
        '--layers',
        type=int,
        nargs='+',
        default=None,
        help='Layer number(s) to extract. Default: [26, 29] for esm2 and [24, 32] for esmc'
    )
    
    parser.add_argument(
        '--batch_size',
        type=int,
        default=1,
        help='Batch size for processing. Default: 1'
    )
    
    parser.add_argument(
        '--setting',
        type=str,
        default='residue',
        help='Embedding type to extract. Can combine with "+", e.g., "mean+weighted". Options: mean, residue, weighted. Default: residue'
    )
    
    parser.add_argument(
        '--weights_file',
        type=str,
        default=None,
        help='Path to TSV file with per-residue weights (required for --setting weighted)'
    )
    
    parser.add_argument(
        '--weights_key_col',
        type=str,
        default='PDB',
        help='Column name in weights file for sequence IDs. Default: PDB'
    )
    
    parser.add_argument(
        '--weights_col',
        type=str,
        default='Pred_BS_Scores',
        help='Column name in weights file for weight values. Default: Pred_BS_Scores'
    )
    
    parser.add_argument(
        '--id_to_seq_file',
        type=str,
        default=None,
        help='Path to pickle file with sequence_id to sequence mapping. If not provided, uses default path.'
    )
    parser.add_argument(
        '--stream-mode',
        action='store_true',
        help='Emit residue embeddings to orchestrator stream instead of relying on residue files.'
    )
    parser.add_argument(
        '--stream-socket',
        type=str,
        default='',
        help='Unix socket path used for stream mode IPC.'
    )
    parser.add_argument(
        '--stream-job-id',
        type=str,
        default='',
        help='Optional job ID attached to stream events.'
    )
    parser.add_argument(
        '--worker-name',
        type=str,
        default='esm',
        help='Worker name attached to stream events.'
    )
    parser.add_argument(
        '--legacy-residue-write',
        action='store_true',
        help='In stream mode, also write residue files to disk for compatibility/fallback.'
    )
    
    args = parser.parse_args()
    
    # Validate arguments
    settings_requested = set(s.strip() for s in args.setting.split("+"))
    valid_settings = {"mean", "residue", "weighted"}
    if not settings_requested.issubset(valid_settings):
        parser.error(f"Invalid setting: {args.setting}. Valid options: mean, residue, weighted (can combine with +)")
    
    if "weighted" in settings_requested and args.weights_file is None:
        parser.error("--weights_file is required when --setting includes 'weighted'")
    
    # Determine sequence file
    if args.seq_file:
        seq_file = Path(args.seq_file)
    else:
        raise NotImplementedError("Default seq_file path is not set. Please provide --seq_file argument.")

    # Load sequence ID to sequence mapping
    if args.id_to_seq_file:
        id_to_seq_path = Path(args.id_to_seq_file)
        print(f"Loading sequence mapping from: {id_to_seq_path}")
    else:
        raise NotImplementedError("Default id_to_seq_file path is not set. Please provide --id_to_seq_file argument.")
    
    seq_id_to_seq = pickle.load(open(id_to_seq_path, 'rb'))
    
    # Load unique sequence IDs from file
    with open(seq_file, 'r') as f:
        seq_ids = [line.strip() for line in f if line.strip()]
    
    # Build sequence dictionary
    seq_dict = {s_id: seq_id_to_seq[s_id] for s_id in seq_ids if s_id in seq_id_to_seq}
    print(f"Loaded {len(seq_dict)} unique sequences from {seq_file}")
    
    # Load weights file if needed
    weights_df = None
    if "weighted" in settings_requested:
        print(f"Loading weights from {args.weights_file}")
        weights_df = pd.read_csv(args.weights_file, sep='\t')
        print(f"Loaded weights for {len(weights_df)} sequences")
    
    # Determine models and layers
    if args.models and args.layers:
        # User provided both models and layers
        models = args.models
        layers = args.layers
        # Create model-layer pairs (distribute layers across models)
        model_layer_pairs = []
        for i, layer in enumerate(layers):
            model = models[i % len(models)]
            model_layer_pairs.append((model, layer))
    elif args.models and not args.layers:
        # User provided models but not layers - use defaults per model
        model_layer_pairs = []
        for model in args.models:
            if model == 'esm2':
                model_layer_pairs.extend([('esm2', 26), ('esm2', 29)])
            elif model == 'esmc':
                model_layer_pairs.extend([('esmc', 24), ('esmc', 32)])
            elif model == 'esm1v':
                model_layer_pairs.append(('esm1v', 33))
    elif args.layers and not args.models:
        # User provided layers but not models - use esm2 by default
        model_layer_pairs = [('esm2', layer) for layer in args.layers]
    else:
        # Default behavior
        model_layer_pairs = [
            ('esm2', 26),
            ('esm2', 29),
            ('esmc', 24),
            ('esmc', 32)
        ]
    
    # Group model_layer_pairs by model so each model is loaded only once
    from collections import defaultdict
    model_to_layers = defaultdict(list)
    for model_name, layer in model_layer_pairs:
        model_to_layers[model_name].append(layer)

    if args.stream_mode:
        if not args.stream_socket:
            parser.error("--stream-socket is required in --stream-mode")
        if settings_requested != {"residue", "mean"}:
            parser.error("--stream-mode currently supports only --setting residue+mean")
        if len(model_to_layers) != 1:
            parser.error("--stream-mode requires exactly one model per invocation")
        model_name, model_layers = next(iter(model_to_layers.items()))
        if model_name not in ("esm2", "esmc"):
            parser.error("--stream-mode currently supports only esm2 or esmc models")
        print(f"\n{'='*70}")
        print(f"Streaming {model_name.upper()} residue+mean embeddings for layers {model_layers}")
        print(f"{'='*70}")
        stream_embeddings_multi_layer(
            seq_dict,
            layers=model_layers,
            batch_size=args.batch_size,
            model=model_name,
            id_to_seq=seq_id_to_seq,
            stream_socket=args.stream_socket,
            stream_job_id=args.stream_job_id,
            worker_name=args.worker_name,
            legacy_residue_write=args.legacy_residue_write,
        )
        print(f"✓ Completed {model_name.upper()} stream extraction for layers {model_layers}")
        print(f"\n{'='*70}")
        print(f"✓ All embeddings complete for {len(seq_dict)} sequences")
        print(f"{'='*70}")
        raise SystemExit(0)

    for model_name, model_layers in model_to_layers.items():
        print(f"\n{'='*70}")
        print(f"Extracting {model_name.upper()} {args.setting} embeddings for layers {model_layers}")
        print(f"{'='*70}")

        if model_name in ('esm2', 'esmc'):
            get_embeddings_multi_layer(
                seq_dict,
                layers=model_layers,
                batch_size=args.batch_size,
                model=model_name,
                id_to_seq=seq_id_to_seq,
                setting=args.setting,
                only_save=True,
                weights_df=weights_df,
                weights_key_col=args.weights_key_col,
                weights_col=args.weights_col,
            )
        else:
            # Fallback for models not supported by get_embeddings_multi_layer (e.g. esm1v)
            for layer in model_layers:
                get_embeddings(
                    seq_dict,
                    batch_size=args.batch_size,
                    model=model_name,
                    id_to_seq=seq_id_to_seq,
                    setting=args.setting,
                    all_layers=False,
                    only_save=True,
                    layer=layer,
                    weights_df=weights_df,
                    weights_key_col=args.weights_key_col,
                    weights_col=args.weights_col,
                )

        print(f"✓ Completed {model_name.upper()} layers {model_layers} {args.setting} embedding extraction")

    print(f"\n{'='*70}")
    print(f"✓ All embeddings complete for {len(seq_dict)} sequences")
    print(f"{'='*70}")
