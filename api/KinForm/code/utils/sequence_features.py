
import numpy as np
import pandas as pd
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from tqdm import tqdm

# Determine repository root relative to this file
# sequence_features.py is in code/utils/, so go up two levels to get to repo root
ROOT = Path(__file__).resolve().parent.parent.parent

# --------------------------------------------------------------------- #
def _truncate_first_last(arr: np.ndarray,
                         keep: int = 1024) -> np.ndarray:
    """
    Keep the first keep//2 and last keep//2 rows of `arr`
    (axis-0 slicing). If arr.shape[0] ≤ keep, return arr unchanged.
    Works for 1-D or 2-D NumPy arrays.
    """
    L = arr.shape[0]
    if L <= keep:
        return arr
    half = keep // 2
    return np.concatenate([arr[:half], arr[-half:]], axis=0)

def _fetch_weights(seq_id: str,
                   df: pd.DataFrame,
                   key_col: str,
                   weights_col: str) -> np.ndarray:
    """
    Return a 1-D float64 array of per-residue weights for `seq_id`.
    Raises if the sequence is missing.
    """
    row = df.loc[df[key_col] == seq_id, weights_col]
    if row.empty:
        raise ValueError(f"No weights found in {weights_col} for sequence {seq_id}")
    return np.fromiter((float(x) for x in row.iloc[0].split(",")), dtype=float)

# --------------------------------------------------------------------- #
def _fetch_cat_weights(seq_id: str,
                       df: pd.DataFrame,
                       key_col: str,
                       L: int) -> np.ndarray:
    """
    Return a catalytic-site weight vector that matches the *post-truncation*
    embedding length.

    • all_AS_probs is always length 1 024.
    • If the sequence is shorter than 1 024 residues, keep only the first L.
    • Otherwise, just return the 1 024-length vector unchanged
      (embeddings and other weights will be truncated to 1 024 later).
    """
    row = df.loc[df[key_col] == seq_id, "all_AS_probs"]
    if row.empty:
        raise ValueError(f"No catalytic weights for sequence {seq_id}")
    probs = np.asarray(row.iloc[0], dtype=float)
    if probs.shape[0] != 1024:
        raise ValueError("all_AS_probs must have length 1024")
    return probs[:L] if L <= 1024 else probs



def _weighted_mean(arr: np.ndarray, w: np.ndarray, normalize=True) -> np.ndarray:
    """Length‑L weights → weighted mean over axis‑0."""
    w = np.asarray(w, dtype=float)
    if normalize:
        w = w / w.sum()
    return (arr * w[:, None]).sum(axis=0)

def _load_residue_embeddings(seq_id: str,
                             *,
                             use_t5: bool,
                             use_esmc: bool,
                             use_esm2: bool,
                             t5_last_layer: bool = False,
                             task: str,
                             ) -> np.ndarray:
    base_dir = ROOT / "results/embeddings"

    if task == "kcat":
        t5_dir = base_dir / "prot_t5_layer_17" if not t5_last_layer else base_dir / "prot_t5_last"
        esmc_layer_dir = base_dir / "esmc_layer_24"
        esm2_dir = base_dir / "esm2_layer_26"
    elif task == "KM":
        t5_dir = base_dir / "prot_t5_layer_19" if not t5_last_layer else base_dir / "prot_t5_last"
        esmc_layer_dir = base_dir / "esmc_layer_32"
        esm2_dir = base_dir / "esm2_layer_29"
    else:
        raise ValueError(f"Unsupported task: {task}")

    emb_list: List[np.ndarray] = []

    if use_t5:
        emb_list.append(np.load(t5_dir / f"{seq_id}.npy"))
    if use_esmc:
        emb = np.load(esmc_layer_dir / f"{seq_id}.npy", allow_pickle=True)
        emb_list.append(emb)
    if use_esm2:
        emb_list.append(np.load(esm2_dir / f"{seq_id}.npy"))

    if not emb_list:
        raise ValueError("No embedding selected for loading.")
    return np.concatenate(emb_list, axis=1) if len(emb_list) > 1 else emb_list[0]


def _try_load_precomputed_vectors(
    seq_id: str,
    model_name: str,
    vec_type: str = "mean"
) -> Tuple[bool, Optional[np.ndarray]]:
    """
    Attempt to load pre-computed vectors from local storage.
    
    Parameters
    ----------
    seq_id : str
        Sequence identifier (e.g., "Sequence 755")
    model_name : str
        Name of the embedding model (e.g., "esm2_layer_26")
    vec_type : str
        Type of vector to load: "mean" or "binding"
        
    Returns
    -------
    Tuple[bool, Optional[np.ndarray]]
        (success, vector) - success is True if vector was loaded successfully
    """
    precomputed_root = ROOT / "results/protein_embeddings"
    
    try:
        if vec_type == "mean":
            vec_path = precomputed_root / model_name / "mean_vecs" / f"{seq_id}.npy"
            if vec_path.exists():
                return True, np.load(vec_path)
        elif vec_type == "binding":
            vec_path = precomputed_root / model_name / "weighted_vecs" / f"{seq_id}.npy"
            if vec_path.exists():
                return True, np.load(vec_path)
        else:
            raise ValueError(f"Unknown vec_type: {vec_type}. Must be 'mean' or 'binding'")
    except Exception:
        pass
    
    return False, None



# ---------- main extraction routine -----------------------------------------
def sequences_to_features(
    sequence_list: List[str],
    binding_site_df: pd.DataFrame,
    ec_num_df: pd.DataFrame,
    cat_sites_df: pd.DataFrame,          # <-- NEW
    use_ec_logits: bool,
    seq_to_id: Dict[str, str],
    *,
    use_t5: bool,
    use_esmc: bool,
    use_esm2: bool,
    t5_last_layer: bool = False,
    prot_rep_mode: str = "global",
    task: str = "kcat",
    use_precomputed: bool = True,
) -> np.ndarray:
    """
    Computes a pooled protein feature vector for each input sequence.

    Given a list of sequences and their corresponding per-residue
    embeddings, this function returns a (N, D) array of pooled feature
    vectors, one per sequence. Pooled vectors can be unweighted (global
    mean) or weighted by predicted binding-site scores, EC-number
    scores, or catalytic-site probabilities.

    Parameters:
    ----------
    sequence_list : List[str]
        List of amino acid sequences to featurize.
    binding_site_df : pd.DataFrame
        DataFrame containing predicted binding-site scores for each sequence.
        Required if prot_rep_mode includes 'binding'.
    ec_num_df : pd.DataFrame
        DataFrame containing EC-number-based per-residue weights or logits.
        Required if prot_rep_mode includes 'ec'.
    cat_sites_df : pd.DataFrame
        DataFrame containing 1024-length catalytic-site probability vectors.
        Required if prot_rep_mode includes 'cat'.
    use_ec_logits : bool
        If True, use 'normal_logits' column in ec_num_df; otherwise use 'weights'.
    seq_to_id : Dict[str, str]
        Maps input sequences to their corresponding embedding file IDs.
    use_t5 : bool
        Whether to include ProtT5 embeddings.
    use_esmc : bool
        Whether to include ESM-C embeddings.
    use_esm2 : bool
        Whether to include ESM-2 embeddings.
    t5_last_layer : bool, default=False
        Whether to use the final layer of ProtT5 or an intermediate one.
    prot_rep_mode : str, default="global"
        Specifies how to pool residue-level embeddings. Can be any '+'-joined
        combination of:
            - 'global'  : unweighted mean over residues
            - 'binding' : weighted by predicted binding-site scores
            - 'ec'      : weighted by EC logits or weights
            - 'cat'     : weighted by catalytic-site probabilities
        The pooled vectors are concatenated in the fixed order:
            binding → ec → cat → global
    task : str, default="kcat"
        Either 'kcat' or 'KM'. Determines which layers are loaded for embeddings.
    use_precomputed : bool, default=True
        If True, attempt to load pre-computed vectors from local storage instead
        of computing from full residue embeddings. Falls back to computing if
        pre-computed vectors are not available.

    Returns:
    -------
    np.ndarray
        Array of shape (N, D) where D depends on selected models and pooling modes.
    """
    mode_tokens = {m.strip().lower() for m in prot_rep_mode.split("+")}
    needs_cat = "cat" in mode_tokens            # convenience flag

    # ---- sanity checks (unchanged) ----------------------------------
    if "binding" in mode_tokens and binding_site_df is None:
        raise ValueError("prot_rep_mode includes 'binding' but binding_site_df is None")
    if "ec" in mode_tokens and ec_num_df is None:
        raise ValueError("prot_rep_mode includes 'ec' but ec_num_df is None")
    if needs_cat and cat_sites_df is None:
        raise ValueError("prot_rep_mode includes 'cat' but cat_sites_df is None")

    # Determine which model layers to use based on task
    if task == "kcat":
        t5_model = "prot_t5_layer_17" if not t5_last_layer else "prot_t5_last"
        esmc_model = "esmc_layer_24"
        esm2_model = "esm2_layer_26"
    elif task == "KM":
        t5_model = "prot_t5_layer_19" if not t5_last_layer else "prot_t5_last"
        esmc_model = "esmc_layer_32"
        esm2_model = "esm2_layer_29"
    else:
        raise ValueError(f"Unsupported task: {task}")
    
    # Track which models are being used
    active_models = []
    if use_t5:
        active_models.append(t5_model)
    if use_esmc:
        active_models.append(esmc_model)
    if use_esm2:
        active_models.append(esm2_model)

    features: List[np.ndarray] = []
    precomputed_hits = 0
    precomputed_misses = 0

    for seq in tqdm(sequence_list, desc="Processing sequences", ncols=100):
        seq_id = seq_to_id.get(seq)
        if seq_id is None:
            raise KeyError(f"Sequence not found in lookup: {seq[:20]}…")

        # -------- Try loading pre-computed vectors -------------------
        can_use_precomputed = (
            use_precomputed 
            and len(active_models) == 1  # Only works for single-model case
            and "ec" not in mode_tokens  # EC weighting not pre-computed
            and "cat" not in mode_tokens  # Catalytic weighting not pre-computed
        )
        
        if can_use_precomputed:
            model_name = active_models[0]
            parts_precomputed: List[np.ndarray] = []
            precomputed_success = True
            
            # Try to load each requested vector type
            for part in ("binding", "global"):
                if part not in mode_tokens:
                    continue
                
                vec_type = "mean" if part == "global" else "binding"
                success, vec = _try_load_precomputed_vectors(seq_id, model_name, vec_type)
                
                if success:
                    parts_precomputed.append(vec)
                else:
                    precomputed_success = False
                    break
            
            if precomputed_success:
                pooled = np.concatenate(parts_precomputed) if len(parts_precomputed) > 1 else parts_precomputed[0]
                features.append(pooled)
                precomputed_hits += 1
                continue
        
        # -------- Fallback: compute from full residue embeddings -----
        precomputed_misses += 1
        
        resid_emb = _load_residue_embeddings(
            seq_id,
            use_t5=use_t5,
            use_esmc=use_esmc,
            use_esm2=use_esm2,
            t5_last_layer=t5_last_layer,
            task=task,
        )

        # -------- fetch weights --------------------------------------
        if "binding" in mode_tokens:
            bs_weights = _fetch_weights(seq_id, binding_site_df,
                                        key_col="PDB",
                                        weights_col="Pred_BS_Scores")
        if "ec" in mode_tokens:
            w_key = "normal_logits" if use_ec_logits else "weights"
            ec_weights = _fetch_weights(seq_id, ec_num_df,
                                        key_col="sequence_id",
                                        weights_col=w_key)
        if needs_cat:
            cat_weights = _fetch_cat_weights(seq_id, cat_sites_df,
                                             key_col="sequence_id",
                                             L=resid_emb.shape[0])

        # -------- apply 1 024-truncation if cat is used --------------
        if needs_cat and resid_emb.shape[0] > 1024:
            resid_emb = _truncate_first_last(resid_emb)
            if "binding" in mode_tokens:
                bs_weights = _truncate_first_last(bs_weights)
            if "ec" in mode_tokens:
                ec_weights = _truncate_first_last(ec_weights)

        # -------- pooling --------------------------------------------
        parts: List[np.ndarray] = []
        for part in ("binding", "ec", "cat", "global"):
            if part not in mode_tokens:
                continue
            if part == "global":
                vec = resid_emb.mean(axis=0)
            elif part == "binding":
                vec = _weighted_mean(resid_emb, bs_weights)
            elif part == "ec":
                vec = _weighted_mean(resid_emb, ec_weights)
            elif part == "cat":
                vec = _weighted_mean(resid_emb, cat_weights, normalize=False)
            parts.append(vec)

        pooled = np.concatenate(parts) if len(parts) > 1 else parts[0]
        features.append(pooled)
    
    # Report pre-computed usage
    if use_precomputed and precomputed_hits > 0:
        total = precomputed_hits + precomputed_misses
        print(f"  ℹ Pre-computed vectors: {precomputed_hits}/{total} ({100*precomputed_hits/total:.1f}%)")

    return np.vstack(features)

def _load_single_embedding(seq_id: str, model: str, task: str, t5_last_layer: bool = False) -> np.ndarray:
    base_dir = ROOT / "results/embeddings"
    if task == "kcat":
        layer_map = {
            "t5": base_dir / "prot_t5_layer_17" if not t5_last_layer else base_dir / "prot_t5_last",
            "esm2": base_dir / "esm2_layer_26",
            "esmc": base_dir / "esmc_layer_24"
        }
    elif task == "KM":
        layer_map = {
            "t5": base_dir / "prot_t5_layer_19" if not t5_last_layer else base_dir / "prot_t5_last",
            "esm2": base_dir / "esm2_layer_29",
            "esmc": base_dir / "esmc_layer_32"
        }
    else:
        raise ValueError(f"Unsupported task: {task}")
    
    emb_path = layer_map[model] / f"{seq_id}.npy"
    if model == "esmc":
        return np.load(emb_path, allow_pickle=True)
    return np.load(emb_path)

# --------------------------------------------------------------------- #
def sequences_to_feature_blocks(
    sequence_list: List[str],
    binding_site_df: pd.DataFrame,
    ec_num_df: pd.DataFrame,
    cat_sites_df: pd.DataFrame,
    seq_to_id: Dict[str, str],
    use_ec_logits: bool,
    *,
    use_t5: bool,
    use_esmc: bool,
    use_esm2: bool,
    t5_last_layer: bool = False,
    prot_rep_mode: str = "binding+cat+global",
    task: str = "kcat",
    use_precomputed: bool = True,
) -> Tuple[List[np.ndarray], List[str]]:
    """
    Returns a list of named feature blocks for each protein sequence.

    Each feature block corresponds to a combination of model type
    (T5, ESMC, or ESM2) and pooling strategy (global, binding, ec, or cat).
    All sequences are pooled identically and returned in a fixed order.

    Parameters:
    ----------
    sequence_list : List[str]
        List of amino acid sequences to featurize.
    binding_site_df : pd.DataFrame
        DataFrame of predicted binding-site weights. Required if 'binding' is used.
    ec_num_df : pd.DataFrame
        DataFrame of EC-based per-residue weights. Required if 'ec' is used.
    cat_sites_df : pd.DataFrame
        DataFrame with fixed-length catalytic-site probabilities. Required if 'cat' is used.
    seq_to_id : Dict[str, str]
        Maps input sequences to their corresponding embedding file IDs.
    use_ec_logits : bool
        Whether to use EC logits ('normal_logits') or soft weights ('weights').
    use_t5 : bool
        Include ProtT5 representations if True.
    use_esmc : bool
        Include ESM-C representations if True.
    use_esm2 : bool
        Include ESM-2 representations if True.
    t5_last_layer : bool, default=False
        Whether to use the final layer of ProtT5 or a predefined intermediate one.
    prot_rep_mode : str, default="binding+cat+global"
        '+'-joined set of pooling modes. Any subset of:
            - 'global', 'binding', 'ec', 'cat'
        For each model, a block is generated for every selected mode.
    task : str, default="kcat"
        Either 'kcat' or 'KM', which determines which layers are loaded.
    use_precomputed : bool, default=True
        If True, attempt to load pre-computed vectors from local storage instead
        of computing from full residue embeddings. Falls back to computing if
        pre-computed vectors are not available.

    Returns:
    -------
    Tuple[List[np.ndarray], List[str]]
        - List of (N, D_block) arrays, one per (model, pooling mode) combination.
        - List of corresponding block names (e.g., 'T5_cat', 'ESMC_global').
          Order is: binding → ec → cat → global.
    """
    mode_tokens = {m.strip().lower() for m in prot_rep_mode.split("+")}
    needs_cat = "cat" in mode_tokens

    # ---- sanity checks (unchanged) ----------------------------------
    if "binding" in mode_tokens and binding_site_df is None:
        raise ValueError("prot_rep_mode includes 'binding' but binding_site_df is None")
    if "ec" in mode_tokens and ec_num_df is None:
        raise ValueError("prot_rep_mode includes 'ec' but ec_num_df is None")
    if needs_cat and cat_sites_df is None:
        raise ValueError("prot_rep_mode includes 'cat' but cat_sites_df is None")

    # Determine which model layers to use based on task
    if task == "kcat":
        model_map = {
            "T5": "prot_t5_layer_17" if not t5_last_layer else "prot_t5_last",
            "ESMC": "esmc_layer_24",
            "ESM2": "esm2_layer_26"
        }
    elif task == "KM":
        model_map = {
            "T5": "prot_t5_layer_19" if not t5_last_layer else "prot_t5_last",
            "ESMC": "esmc_layer_32",
            "ESM2": "esm2_layer_29"
        }
    else:
        raise ValueError(f"Unsupported task: {task}")

    block_dict: Dict[str, List[np.ndarray]] = {}
    precomputed_stats: Dict[str, int] = {}  # block_name -> hit_count
    total_seqs = len(sequence_list)

    def pool(rep, mode, w, normalize=True):
        return rep.mean(axis=0) if mode == "global" else _weighted_mean(rep, w, normalize=normalize)

    for seq in tqdm(sequence_list, desc="Processing sequences", ncols=100):
        seq_id = seq_to_id[seq]
        
        # Try to load pre-computed vectors if enabled and EC weighting is not used
        use_precomp_this_seq = (
            use_precomputed 
            and "ec" not in mode_tokens
            and "cat" not in mode_tokens  # Catalytic weighting not pre-computed
        )
        
        # ---- load embeddings once per model -------------------------
        emb_cache: Dict[str, np.ndarray] = {}
        
        # For each active model, try pre-computed first, then fall back
        active_models = []
        if use_t5:
            active_models.append(("T5", "t5"))
        if use_esmc:
            active_models.append(("ESMC", "esmc"))
        if use_esm2:
            active_models.append(("ESM2", "esm2"))
        
        for mdl_name, mdl_key in active_models:
            model_full_name = model_map[mdl_name]
            loaded_from_precomputed = False
            
            if use_precomp_this_seq:
                # Try to load all needed vectors for this model from pre-computed
                precomp_vecs = {}
                all_available = True
                
                for mode in ("binding", "global"):
                    if mode not in mode_tokens:
                        continue
                    vec_type = "mean" if mode == "global" else "binding"
                    success, vec = _try_load_precomputed_vectors(seq_id, model_full_name, vec_type)
                    if success:
                        precomp_vecs[mode] = vec
                    else:
                        all_available = False
                        break
                
                if all_available:
                    # All vectors available, use them directly
                    for mode, vec in precomp_vecs.items():
                        blk_name = f"{mdl_name}_{mode}"
                        block_dict.setdefault(blk_name, []).append(vec)
                        precomputed_stats[blk_name] = precomputed_stats.get(blk_name, 0) + 1
                    loaded_from_precomputed = True
            
            # Fall back to loading full embeddings if needed
            if not loaded_from_precomputed:
                emb_cache[mdl_name] = _load_single_embedding(
                    seq_id, mdl_key,
                    task=task,
                    t5_last_layer=t5_last_layer
                )

        # If we loaded any full embeddings, process them the traditional way
        if emb_cache:
            L_full = next(iter(emb_cache.values())).shape[0]

            # ---- fetch weights ------------------------------------------
            if "binding" in mode_tokens:
                bs_weights = _fetch_weights(seq_id, binding_site_df,
                                            key_col="PDB",
                                            weights_col="Pred_BS_Scores")
            if "ec" in mode_tokens:
                w_key = "normal_logits" if use_ec_logits else "weights"
                ec_weights = _fetch_weights(seq_id, ec_num_df,
                                            key_col="sequence_id",
                                            weights_col=w_key)
            if needs_cat:
                cat_weights = _fetch_cat_weights(seq_id, cat_sites_df,
                                                 key_col="sequence_id",
                                                 L=L_full)

            # ---- optional truncation ------------------------------------
            if needs_cat and L_full > 1024:
                # slice every embedding in the cache
                for mdl in emb_cache:
                    emb_cache[mdl] = _truncate_first_last(emb_cache[mdl])
                if "binding" in mode_tokens:
                    bs_weights = _truncate_first_last(bs_weights)
                if "ec" in mode_tokens:
                    ec_weights = _truncate_first_last(ec_weights)

            # ---- pooling & block collection -----------------------------
            for mode in ("binding", "ec", "cat", "global"):
                if mode not in mode_tokens:
                    continue
                if mode == "binding":
                    w = bs_weights
                elif mode == "ec":
                    w = ec_weights
                elif mode == "cat":
                    w = cat_weights
                else:
                    w = None
                for mdl, rep in emb_cache.items():
                    blk_name = f"{mdl}_{mode}"
                    normalize = mode != "cat"
                    block_dict.setdefault(blk_name, []).append(pool(rep, mode, w, normalize=normalize))

    # ---- deterministic ordering -------------------------------------
    ordered_keys: List[str] = []
    for mode in ("binding", "ec", "cat", "global"):
        for mdl in ("T5", "ESMC", "ESM2"):
            k = f"{mdl}_{mode}"
            if k in block_dict:
                ordered_keys.append(k)

    # Report pre-computed usage
    if use_precomputed and precomputed_stats:
        print("  ℹ Pre-computed vectors usage:")
        for blk_name in ordered_keys:
            if blk_name in precomputed_stats:
                hits = precomputed_stats[blk_name]
                print(f"    - {blk_name}: {hits}/{total_seqs} ({100*hits/total_seqs:.1f}%)")

    blocks = [np.vstack(block_dict[k]) for k in ordered_keys]
    return blocks, ordered_keys