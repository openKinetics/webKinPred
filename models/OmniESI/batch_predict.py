#!/usr/bin/env python3
"""
OmniESI prediction wrapper for webKinPred generic subprocess engine.

Contract:
    python batch_predict.py --input <input.json> --output <output.json>

Input JSON:
    {"rows": [{"sequence": "MKTAY...", "substrates": "CC(=O)O",
               "seq_id": "sid_abc123"}, ...],
     "params": {"kinetics_type": "KCAT",     # or "KM"
                "variant": "ensemble"},      # "ensemble" (OmniESI, default) or
                                             # "o2denet"  (OmniESI + O2DENet)
     "target": "kcat"}                       # alternative to params.kinetics_type

Output JSON:
    {"predictions": [0.42, null, 1.3, ...], "invalid_indices": [1, ...]}

    Predictions are in **linear kinetic units** (kcat: 1/s, Km: mM).
    The OmniESI model outputs log10-scaled values; this script applies the
    inverse transform (10**) after ensemble averaging in log space.

Embedding staging
-----------------
OmniESI requires full per-residue ESM2 token matrices — shape [seq_len, 1280]
using esm2_t33_650M_UR50D.  This is architecturally distinct from CatPred's
cache (which stores checkpoint-specific *pooled* vectors), so OmniESI uses an
ephemeral embedding family:

    <OMNIESI_EMBED_CACHE_DIR>/<seq_id>.pt   (cpu tensor, shape [seq_len, 1280])

The cache directory is injected by the platform via:
    OMNIESI_EMBED_CACHE_DIR   — path to media/sequence_info/omniesi_esm2/
                                 (set by data_path_env in SubprocessEngineConfig)

If a seq_id is present in the payload row, the shared cache is checked first.
If the file is missing the embedding is computed on-the-fly (fail-open), which
is the correct fallback when GPU precompute is unavailable.

After prediction, staged full-residue matrices for this job are deleted by
default, matching EITLEM's esm1v flow. Set OMNIESI_DELETE_EMBEDDINGS_AFTER_RUN=0
only for debugging.

If no seq_id is present in the row (e.g. local dev), staging is skipped.

Environment variables (all optional):
    OMNIESI_EMBED_CACHE_DIR
        Path to the shared omniesi_esm2 embedding cache directory.
        Default: <repo>/media/sequence_info/omniesi_esm2
    OmniESI_CACHE_DIR / OMNIESI_CACHE_DIR
        Writable root for torch hub and ESM model weights download.
        Default: <repo>/cache/omniesi
    OmniESI_ROOT
        OmniESI code root containing models.py, configs.py, utils.py,
        and scripts/embedding.py.
        Default: probed from script location, then /OmniESI.
    OmniESI_ADDITIONAL_DATA
        Path to the OmniESI_additional_data/additional_data/ directory
        that contains results/CatPred_kcat and results/CatPred_km.
        Default: probed relative to OmniESI_ROOT.
    CUDA_VISIBLE_DEVICES
        Standard mechanism to select a specific GPU on multi-GPU hosts.
    OMNIESI_DELETE_EMBEDDINGS_AFTER_RUN
        Default: 1. Delete full residue matrices after this prediction run.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import warnings
from functools import partial
from pathlib import Path
from typing import Any

import dgl
import numpy as np
import pandas as pd
import torch
from dgllife.utils import (
    CanonicalAtomFeaturizer,
    CanonicalBondFeaturizer,
    smiles_to_bigraph,
)
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Logging — stderr keeps stdout clean for Progress: lines.
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    stream=sys.stderr,
)
logger = logging.getLogger(__name__)

# ============================================================================
# Cache directory resolution — runs at import time so TORCH_HOME / ESM_HOME
# are set before any torch or ESM imports happen inside helper functions.
# ============================================================================

_REPO_ROOT = Path(__file__).resolve().parents[2]

def _resolve_weights_cache_dir() -> Path:
    """Writable root for torch hub / ESM model weight downloads."""
    cache_root = os.getenv("OmniESI_CACHE_DIR") or os.getenv("OMNIESI_CACHE_DIR")
    if cache_root:
        return Path(cache_root).resolve()
    return (_REPO_ROOT / "cache" / "omniesi").resolve()


def _resolve_embed_cache_dir() -> Path:
    """
    Shared per-residue ESM2 embedding cache.

    Injected by the platform via OMNIESI_EMBED_CACHE_DIR (set through
    data_path_env in SubprocessEngineConfig).  Falls back to a local
    subdirectory so the script remains runnable standalone.
    """
    env = os.getenv("OMNIESI_EMBED_CACHE_DIR")
    if env:
        return Path(env).resolve()
    # Fallback: local cache alongside model weights (dev / standalone use)
    return _resolve_weights_cache_dir() / "esm_embeddings"


OMNIESI_WEIGHTS_CACHE_DIR = _resolve_weights_cache_dir()
OMNIESI_EMBED_CACHE_DIR   = _resolve_embed_cache_dir()
OMNIESI_EMBED_CACHE_DIR.mkdir(parents=True, exist_ok=True)

# Must be set before torch / ESM are imported by any function below.
os.environ.setdefault("TORCH_HOME", str(OMNIESI_WEIGHTS_CACHE_DIR / "torch"))
os.environ.setdefault("ESM_HOME",   str(OMNIESI_WEIGHTS_CACHE_DIR / "esm"))

# ============================================================================
# Code-root resolution
# ============================================================================

def _resolve_code_root() -> Path:
    script_dir = Path(__file__).resolve().parent
    env_root   = Path(os.environ.get("OmniESI_ROOT", "/OmniESI")).resolve()
    candidates = [
        script_dir / "code",
        script_dir,
        env_root,
        env_root / "code",
    ]
    for candidate in candidates:
        if (candidate / "configs.py").exists() and (candidate / "models.py").exists():
            return candidate
    return env_root


ROOT_DIR = _resolve_code_root()
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

logger.debug("OmniESI code root resolved to: %s", ROOT_DIR)

# ============================================================================
# Device selection
# ============================================================================

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
DEFAULT_DELETE_EMBEDDINGS_AFTER_RUN = (
    str(os.environ.get("OMNIESI_DELETE_EMBEDDINGS_AFTER_RUN", "1")).strip().lower()
    in {"1", "true", "yes", "on"}
)
SHOW_TQDM = (
    str(os.environ.get("OMNIESI_SHOW_TQDM", "0")).strip().lower()
    in {"1", "true", "yes", "on"}
)

# ============================================================================
# Constants
# ============================================================================

SEED_LIST = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90]

# ============================================================================
# Lazy module loader / import guard
# ============================================================================

def _ensure_omniesi_code_importable() -> None:
    try:
        import models    as _m  # noqa: F401
        import utils     as _u  # noqa: F401
        import configs   as _c  # noqa: F401
    except ImportError as exc:
        raise RuntimeError(
            f"OmniESI source modules (models.py / utils.py / configs.py) are not "
            f"importable from ROOT_DIR='{ROOT_DIR}'. "
            f"Set the OmniESI_ROOT environment variable to the directory that "
            f"contains those files. Original ImportError: {exc}"
        ) from exc

    try:
        import scripts.embedding as _e  # noqa: F401
    except ImportError as exc:
        raise RuntimeError(
            f"OmniESI scripts/embedding.py is not importable from ROOT_DIR='{ROOT_DIR}'. "
            f"Original ImportError: {exc}"
        ) from exc


# ============================================================================
# Shared embedding staging I/O
#
# Cache key: seq_id  (platform-assigned stable identifier, from seqmap tool).
# File path: OMNIESI_EMBED_CACHE_DIR / <seq_id>.pt
# Stored tensor shape: [seq_len, 1280]  (cpu, float32)
#
# If seq_id is absent (local / test usage), staging is skipped entirely and
# the embedding is computed fresh each time — the fail-open path.
# ============================================================================

def _embed_cache_path(seq_id: str) -> Path:
    return OMNIESI_EMBED_CACHE_DIR / f"{seq_id}.pt"


def _torch_load_compat(path: str | Path, *, map_location: Any) -> Any:
    """
    Load a torch artifact across both newer and older torch versions.

    Newer torch accepts weights_only=True; older torch/pickle stacks may raise
    either TypeError or an Unpickler keyword error. Fall back to the legacy
    torch.load signature in those compatibility cases only.
    """
    try:
        return torch.load(str(path), map_location=map_location, weights_only=True)
    except TypeError:
        return torch.load(str(path), map_location=map_location)
    except Exception as exc:
        from tools.executor.policy import propagate_resource_error
        propagate_resource_error(exc)
        if "weights_only" in str(exc):
            return torch.load(str(path), map_location=map_location)
        raise


def _load_shared_embedding(seq_id: str | None) -> torch.Tensor | None:
    """
    Load a cached per-residue ESM2 embedding by seq_id.

    Returns tensor of shape [seq_len, 1280] on cpu, or None on cache miss /
    absent seq_id.
    """
    if not seq_id:
        return None
    cache_path = _embed_cache_path(seq_id)
    if not cache_path.exists():
        return None
    try:
        tensor = _torch_load_compat(cache_path, map_location="cpu")
        # Normalise: stored shape may be [1, seq_len, 1280] (from ESM batch dim)
        # or [seq_len, 1280].  OmniESI prepare_inputs expects [1, seq_len, 1280].
        if tensor.dim() == 2:
            tensor = tensor.unsqueeze(0)
        return tensor
    except Exception as exc:
        from tools.executor.policy import propagate_resource_error
        propagate_resource_error(exc)
        logger.warning("Embedding cache read failed for seq_id=%r: %s", seq_id, exc)
        return None


def _save_shared_embedding(seq_id: str | None, embedding: torch.Tensor) -> None:
    """
    Persist a per-residue ESM2 embedding to the shared staging directory.

    Stores shape [seq_len, 1280] (strips batch dim if present) so the file
    is as compact as possible and GPU-step writers also use this convention.
    """
    if not seq_id:
        return
    cache_path = _embed_cache_path(seq_id)
    try:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        # Strip batch dimension before persisting.
        tensor_to_save = embedding.cpu()
        if tensor_to_save.dim() == 3 and tensor_to_save.shape[0] == 1:
            tensor_to_save = tensor_to_save.squeeze(0)
        torch.save(tensor_to_save, cache_path)
    except Exception as exc:
        from tools.executor.policy import propagate_resource_error
        propagate_resource_error(exc)
        logger.warning("Embedding cache write failed for seq_id=%r: %s", seq_id, exc)


def _cleanup_shared_embeddings(seq_ids: list[str], *, cleanup_enabled: bool) -> None:
    if not cleanup_enabled:
        return
    cleanup_ids = {str(sid).strip() for sid in seq_ids if str(sid).strip()}
    for seq_id in cleanup_ids:
        try:
            _embed_cache_path(seq_id).unlink(missing_ok=True)
        except OSError:
            pass
    try:
        from tools.gpu_embed_service.cache_io import remove_manifest_entries

        remove_manifest_entries(OMNIESI_EMBED_CACHE_DIR, cleanup_ids)
    except Exception as exc:
        from tools.executor.policy import propagate_resource_error
        propagate_resource_error(exc)
        logger.warning("Embedding manifest cleanup failed: %s", exc)


def _payload_cleanup_flag(payload: dict[str, Any]) -> bool:
    """
    Resolve cleanup policy from payload params with env-backed default.

    params.cleanup_embeddings_after_run is an internal webKinPred orchestration
    flag used for multi-stage jobs that should keep shared embeddings between
    stages and clean them after the last stage.
    """
    params = payload.get("params") or {}
    if not isinstance(params, dict) or "cleanup_embeddings_after_run" not in params:
        return DEFAULT_DELETE_EMBEDDINGS_AFTER_RUN

    raw = params.get("cleanup_embeddings_after_run")
    if isinstance(raw, bool):
        return raw
    if raw is None:
        return DEFAULT_DELETE_EMBEDDINGS_AFTER_RUN
    return str(raw).strip().lower() in {"1", "true", "yes", "on"}


# ============================================================================
# ESM model — in-process singleton, keyed by device
# ============================================================================

_ESM_MODEL_CACHE: dict[torch.device, Any] = {}


def _get_esm_model(dev: torch.device) -> Any:
    """Lazy-load and cache the ESM embedding model for the given device."""
    from tools.executor.policy import require_cached
    require_cached(False, "protein embeddings must be scheduled")
    if dev not in _ESM_MODEL_CACHE:
        from scripts.embedding import ESM_model  # lazy — cached in sys.modules
        model = ESM_model()
        model.device = dev
        model.eval()
        _ESM_MODEL_CACHE[dev] = model
        logger.info("Loaded ESM model on %s", dev)
    return _ESM_MODEL_CACHE[dev]


# ============================================================================
# Additional data directory resolution
# ============================================================================

def _additional_data_dir() -> Path:
    env_value = os.environ.get("OmniESI_ADDITIONAL_DATA")
    if env_value:
        return Path(env_value).resolve()

    candidates = [
        ROOT_DIR / "OmniESI_additional_data" / "additional_data",
        ROOT_DIR.parent / "OmniESI_additional_data" / "additional_data",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()

    return candidates[0].resolve()


# ============================================================================
# Input helpers
# ============================================================================

def _safe_rows(payload: dict[str, Any]) -> list[dict[str, Any]]:
    rows = payload.get("rows")
    if not isinstance(rows, list):
        raise RuntimeError("Input payload is malformed: 'rows' must be a list")
    for i, row in enumerate(rows):
        if not isinstance(row, dict):
            raise RuntimeError(f"Input payload is malformed: row {i} is not an object")
    return list(rows)


def _substrate_to_smiles(raw: Any) -> str | None:
    if isinstance(raw, list):
        tokens = [str(item).strip() for item in raw if str(item).strip()]
    else:
        text = str(raw).strip()
        if not text:
            return None
        tokens = [tok.strip() for tok in text.split(";") if tok.strip()]

    if len(tokens) != 1:
        return None
    return tokens[0]


def _kinetics_type_from_payload(payload: dict[str, Any]) -> str:
    params = payload.get("params") or {}
    ktype  = str(params.get("kinetics_type", "")).upper().strip()
    if ktype in {"KCAT", "KM"}:
        return ktype

    target = str(payload.get("target", "")).strip()
    if target == "kcat":
        return "KCAT"
    if target == "Km":
        return "KM"

    raise RuntimeError(
        "Cannot determine kinetics type: set params.kinetics_type to 'KCAT' or 'KM', "
        "or set target to 'kcat' or 'Km'."
    )


# ============================================================================
# Input preparation for a single (smiles, sequence) pair
# ============================================================================

# Featurizers are stateless — instantiate once and reuse across all rows.
_ATOM_FEATURIZER = CanonicalAtomFeaturizer()
_BOND_FEATURIZER = CanonicalBondFeaturizer(self_loop=True)

def prepare_inputs(
    smiles: str,
    protein_seq: str,
    embedder: Any | None,     # ESM_model instance, lazy on cache miss
    seq_id: str | None,       # shared cache key; None = skip caching
    smiles_graph_cache: "dict | None" = None,
) -> tuple[Any, torch.Tensor, torch.Tensor, torch.Tensor] | None:
    """
    Build DGL molecular graph + ESM per-residue embedding for one sample.

    Embedding lookup order:
        1. Shared cache file at OMNIESI_EMBED_CACHE_DIR/<seq_id>.pt   (hit → skip GPU)
        2. On-the-fly ESM inference with the in-process model singleton
           → result written to shared cache for future jobs.

    Returns (v_d, v_p, v_d_mask, v_p_mask) on success, None on failure.
    """
    try:
        if smiles_graph_cache is not None and smiles in smiles_graph_cache:
            src, dst, cached_h, cached_edata, num_actual_nodes = smiles_graph_cache[smiles]
            v_d = dgl.graph((src, dst))
            v_d.ndata["h"] = cached_h.clone()
            for k, t in cached_edata.items():
                v_d.edata[k] = t.clone()
            v_d = dgl.batch([v_d])
            v_d_mask = torch.zeros(num_actual_nodes, dtype=torch.bool).unsqueeze(0)
        else:
            fc = partial(smiles_to_bigraph, add_self_loop=True)
            v_d = fc(
                smiles=smiles,
                node_featurizer=_ATOM_FEATURIZER,
                edge_featurizer=_BOND_FEATURIZER,
            )
            actual_node_feats = v_d.ndata.pop("h")
            num_actual_nodes  = actual_node_feats.shape[0]

            virtual_bit = torch.zeros([num_actual_nodes, 1])
            v_d.ndata["h"] = torch.cat((actual_node_feats, virtual_bit), dim=1)
            v_d = v_d.add_self_loop()

            if smiles_graph_cache is not None:
                src, dst = v_d.edges()
                smiles_graph_cache[smiles] = (
                    src.clone(),
                    dst.clone(),
                    v_d.ndata["h"].clone(),
                    {k: t.clone() for k, t in v_d.edata.items()},
                    num_actual_nodes,
                )

            v_d = dgl.batch([v_d])
            v_d_mask = torch.zeros(num_actual_nodes, dtype=torch.bool).unsqueeze(0)

        # ── Embedding: shared cache first, then on-the-fly ────────────────────
        v_p = _load_shared_embedding(seq_id)
        if v_p is None:
            # Cache miss (or no seq_id) — compute via ESM and persist.
            if embedder is None:
                embedder = _get_esm_model(device)
            v_p = embedder([protein_seq])
            v_p = v_p[:, 1 : len(protein_seq) + 1, :]   # strip BOS token
            _save_shared_embedding(seq_id, v_p)
            logger.debug(
                "Embedding computed on-the-fly for seq_id=%r (len=%d)",
                seq_id, len(protein_seq),
            )
        else:
            logger.debug("Embedding loaded from shared cache for seq_id=%r", seq_id)
            # Restore batch dim expected by model: [1, seq_len, 1280]
            if v_p.dim() == 2:
                v_p = v_p.unsqueeze(0)

        v_p_mask = torch.zeros(v_p.shape[1], dtype=torch.bool).unsqueeze(0)

        return v_d, v_p, v_d_mask, v_p_mask

    except Exception as exc:

        from tools.executor.policy import propagate_resource_error

        propagate_resource_error(exc)
        logger.debug("prepare_inputs failed for smiles=%r: %s", smiles[:40], exc)
        return None


# ============================================================================
# Checkpoint discovery
# ============================================================================

def _discover_available_seeds(weight_folder: Path) -> list[int]:
    return [
        seed for seed in SEED_LIST
        if (weight_folder / f"OmniESI_ensemble_{seed}" / "best_model_epoch.pth").exists()
    ]


def _variant_from_payload(payload: dict[str, Any]) -> str:
    """
    Resolve the prediction variant from payload params.

    "ensemble"  — vanilla OmniESI: 10-seed ensemble under fold_OmniESI (default).
    "o2denet"   — OmniESI + O2DENet: a single checkpoint under fold_O2DENet.
                  The O2DENet checkpoints share the OmniESI inference architecture
                  but carry extra training-only parameters (e.g. cross_attn), so
                  they are loaded with strict=False (see _resolve_checkpoints).
    """
    params = payload.get("params") or {}
    raw = str(params.get("variant", "")).strip().lower() if isinstance(params, dict) else ""
    return raw if raw in {"ensemble", "o2denet"} else "ensemble"


def _resolve_checkpoints(kinetics_type: str, variant: str) -> tuple[list[Path], bool]:
    """
    Resolve the checkpoint file list and the load_state_dict strictness for a
    (kinetics_type, variant) pair.

    Returns
    -------
    (checkpoint_paths, strict)
        checkpoint_paths : list of best_model_epoch.pth files to ensemble over.
                           A single-element list runs as a one-model "ensemble"
                           (geometric mean of one value == that value).
        strict           : passed to load_state_dict. False for O2DENet so the
                           training-only parameters absent from our OmniESI class
                           are ignored without error.
    """
    additional_data_dir = _additional_data_dir()
    if kinetics_type == "KCAT":
        task_dir = "CatPred_kcat"
    elif kinetics_type == "KM":
        task_dir = "CatPred_km"
    else:
        raise RuntimeError(f"Unsupported kinetics type: {kinetics_type!r}")
    task_root = additional_data_dir / "results" / task_dir

    if variant == "o2denet":
        ckpt = task_root / "fold_O2DENet" / "best_model_epoch.pth"
        if not ckpt.exists():
            raise RuntimeError(
                f"No OmniESI+O2DENet checkpoint found at: {ckpt}. "
                "Set OmniESI_ADDITIONAL_DATA to the correct additional_data/ directory."
            )
        # strict=False: O2DENet checkpoints include training-only parameters
        # (e.g. cross_attn) that are not part of the OmniESI inference graph.
        return [ckpt], False

    # Default: vanilla OmniESI 10-seed ensemble.
    weight_folder = task_root / "fold_OmniESI"
    available_seeds = _discover_available_seeds(weight_folder)
    if not available_seeds:
        raise RuntimeError(
            f"No OmniESI checkpoints found under: {weight_folder}. "
            "Set OmniESI_ADDITIONAL_DATA to the correct additional_data/ directory."
        )
    return (
        [
            weight_folder / f"OmniESI_ensemble_{seed}" / "best_model_epoch.pth"
            for seed in available_seeds
        ],
        True,
    )


# ============================================================================
# Ensemble inference with progress streaming
# ============================================================================

def predict_kinetic_parameter_ensemble(
    valid_df: pd.DataFrame,
    kinetics_type: str,
    esm_model: Any | None,
    n_total_rows: int,
    first_valid_global_idx: int,
    valid_global_indices: list[int],
    variant: str = "ensemble",
) -> list[float | None]:
    """
    Run OmniESI ensemble inference over valid_df rows.

    Loop order: row-outer, seed-inner.
    All checkpoints are loaded once upfront and held in memory for the
    duration of the batch.  Progress is emitted once per completed row
    (after all seeds have run for that row), matching the per-row progress
    semantics of other webKinPred subprocess engines.

    Log-space averaging / inverse transform
    ----------------------------------------
    OmniESI is built on the CatPred backbone and trains on log10-scaled
    kinetic targets.  Correct ensemble strategy:
        1. Collect raw log10 outputs from each seed.
        2. Average in log space  (= geometric mean in linear space).
        3. Apply inverse transform: 10 ** mean_log → linear units
           (kcat: 1/s, Km: mM).
    """
    from models import OmniESI            # lazy
    from utils import set_seed            # lazy
    from configs import get_cfg_defaults  # lazy

    cfg = get_cfg_defaults()
    cfg.merge_from_file(str(ROOT_DIR / "configs" / "model" / "OmniESI_ensemble.yaml"))
    set_seed(42)
    warnings.filterwarnings("ignore", message="invalid value encountered in divide")

    checkpoint_paths, strict = _resolve_checkpoints(kinetics_type, variant)

    n_rows        = len(valid_df)
    progress_done = -1
    progress_base = first_valid_global_idx

    def _emit(local_row_done: int) -> None:
        nonlocal progress_done
        global_done = progress_base + local_row_done
        if global_done > progress_done:
            progress_done = global_done
            if SHOW_TQDM:
                sys.stderr.write("\n")
                sys.stderr.flush()
            print(f"Progress: {global_done}/{n_total_rows}", flush=True)

    # ── Load all checkpoints once upfront ────────────────────────────────────
    # Keeps each model in GPU memory for the full batch rather than reloading
    # per row.  With 10 seeds of OmniESI the combined VRAM cost is manageable;
    # if memory is tight, reduce to a smaller seed subset before calling here.
    logger.info(
        "Loading %d OmniESI checkpoint(s) (variant=%s, strict=%s) onto %s",
        len(checkpoint_paths), variant, strict, device,
    )
    loaded_models: list[Any] = []
    for weight_path in checkpoint_paths:
        m = OmniESI(**cfg)
        state = _torch_load_compat(weight_path, map_location=device)
        load_result = m.load_state_dict(state, strict=strict)
        if not strict:
            missing    = list(getattr(load_result, "missing_keys", []) or [])
            unexpected = list(getattr(load_result, "unexpected_keys", []) or [])
            if missing:
                # Missing keys mean our architecture expected weights the
                # checkpoint did not provide — predictions would be unreliable.
                raise RuntimeError(
                    f"Checkpoint {weight_path} is missing {len(missing)} parameter(s) "
                    f"required by the OmniESI architecture, e.g. {missing[:8]}."
                )
            if unexpected:
                logger.info(
                    "Checkpoint %s loaded with %d training-only parameter(s) ignored "
                    "(not part of the inference graph), e.g. %s",
                    weight_path.name, len(unexpected), unexpected[:8],
                )
        m.to(device)
        m.eval()
        loaded_models.append(m)
    torch.backends.cudnn.benchmark = True
    logger.info("All checkpoints loaded")

    # ── Row-outer, seed-inner inference ──────────────────────────────────────
    ensemble_predictions: list[float | None] = []
    smiles_graph_cache: dict = {}

    rows_iter = tqdm(
        valid_df.iterrows(),
        total=n_rows,
        desc="OmniESI",
        file=sys.stderr,
        disable=not SHOW_TQDM,
    )
    with torch.no_grad():
        for local_idx, (_, row) in enumerate(rows_iter):
            # Build inputs once per row (embedding is shared across seeds)
            result = prepare_inputs(
                row["smiles"],
                row["sequence"],
                esm_model,
                row.get("seq_id"),
                smiles_graph_cache,
            )

            if result is None:
                ensemble_predictions.append(None)
                _emit(local_idx + 1)
                continue

            v_d, v_p, v_d_mask, v_p_mask = result
            v_d      = v_d.to(device)
            v_p      = v_p.to(device)
            v_d_mask = v_d_mask.to(device)
            v_p_mask = v_p_mask.to(device)
            node_h   = v_d.ndata["h"].clone()

            log_values: list[float] = []
            for seed_idx, m in enumerate(loaded_models):
                try:
                    # Encoder_drug pops ndata["h"], so restore it before each
                    # ensemble checkpoint reuses the same molecular graph.
                    v_d.ndata["h"] = node_h.clone()
                    output = m(v_d, v_p, v_d_mask, v_p_mask)
                    log_values.append(float(output[-1].cpu().numpy().flatten()[0]))
                except Exception as exc:
                    from tools.executor.policy import propagate_resource_error
                    propagate_resource_error(exc)
                    logger.warning(
                        "Row %d seed %d inference failed: %s",
                        local_idx, available_seeds[seed_idx], exc,
                    )

            if log_values:
                ensemble_predictions.append(float(10.0 ** float(np.mean(log_values))))
            else:
                ensemble_predictions.append(None)

            # Emit once per completed row — matches per-row progress semantics
            _emit(local_idx + 1)

    _emit(n_rows)
    return ensemble_predictions


# ============================================================================
# Top-level payload handler
# ============================================================================

def run_from_payload(payload: dict[str, Any]) -> dict[str, Any]:
    """
    Validate input, run OmniESI ensemble inference, return output dict.
    """
    _ensure_omniesi_code_importable()

    rows          = _safe_rows(payload)
    kinetics_type = _kinetics_type_from_payload(payload)
    variant       = _variant_from_payload(payload)
    cleanup_embeddings_after_run = _payload_cleanup_flag(payload)

    n_total         = len(rows)
    predictions     : list[float | None] = [None] * n_total
    invalid_indices : list[int] = []
    valid_indices   : list[int] = []
    valid_rows      : list[dict[str, Any]] = []

    for idx, row in enumerate(rows):
        seq    = str(row.get("sequence", "")).strip()
        smiles = _substrate_to_smiles(row.get("substrates", row.get("substrate", "")))
        if not seq or smiles is None:
            invalid_indices.append(idx)
        else:
            valid_indices.append(idx)
            valid_rows.append({
                "sequence": seq,
                "smiles":   smiles,
                "seq_id":   row.get("seq_id"),   # None when absent — caching skipped
            })

    if not valid_indices:
        for i in range(n_total):
            print(f"Progress: {i + 1}/{n_total}", flush=True)
        return {
            "predictions":     predictions,
            "invalid_indices": sorted(invalid_indices),
        }

    if device.type == "cuda":
        torch.cuda.empty_cache()

    valid_df = pd.DataFrame(valid_rows)
    touched_seq_ids = [
        str(row.get("seq_id", "")).strip()
        for row in valid_rows
        if str(row.get("seq_id", "")).strip()
    ]
    try:
        valid_preds = predict_kinetic_parameter_ensemble(
            valid_df               = valid_df,
            kinetics_type          = kinetics_type,
            esm_model              = None,
            n_total_rows           = n_total,
            first_valid_global_idx = valid_indices[0],
            valid_global_indices   = valid_indices,
            variant                = variant,
        )
    finally:
        _cleanup_shared_embeddings(
            touched_seq_ids,
            cleanup_enabled=cleanup_embeddings_after_run,
        )

    for local_idx, pred in enumerate(valid_preds):
        global_idx = valid_indices[local_idx]
        predictions[global_idx] = pred
        if pred is None:
            invalid_indices.append(global_idx)

    print(f"Progress: {n_total}/{n_total}", flush=True)

    return {
        "predictions":     predictions,
        "invalid_indices": sorted(set(invalid_indices)),
    }


# ============================================================================
# CLI entry point
# ============================================================================

def main() -> None:
    parser = argparse.ArgumentParser(description="OmniESI webKinPred batch adapter")
    parser.add_argument("--input",  required=True, help="Input JSON path")
    parser.add_argument("--output", required=True, help="Output JSON path")
    args = parser.parse_args()

    with open(args.input, "r", encoding="utf-8") as fh:
        payload = json.load(fh)

    result = run_from_payload(payload)

    with open(args.output, "w", encoding="utf-8") as fh:
        json.dump(result, fh)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        from tools.executor.policy import propagate_resource_error
        propagate_resource_error(exc)
        print(f"[OmniESI] FATAL: {exc}", file=sys.stderr, flush=True)
        raise
