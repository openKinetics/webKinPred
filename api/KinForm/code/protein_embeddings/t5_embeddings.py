#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Extract ProtT5‐XL UniRef50 embeddings.

Four operating modes:
    1. --setting mean
       → compute & save the per-sequence **mean vector of the last layer**.
    2. --setting residue [--layer N]
       → compute & save the **per-residue matrix** from either the last
         layer (default) or the user–specified layer *N* (0-based, 0-23).
    3. --setting weighted [--layer N] --weights_file WEIGHTS.tsv
       → compute weighted mean embeddings using per-residue weights.
    4. --all_layers
       → compute & save the **mean vector of every encoder layer**;
         result shape: [24, hidden_size].

Embeddings are written under:
    results/full_protein_embeddings/t5_{layer}/       (per-residue embeddings)
    results/protein_embeddings/prot_t5_last/mean_vecs/     (mean vectors, last layer)
    results/protein_embeddings/prot_t5_layer_{n}/mean_vecs/ (mean vectors, layer n)
    results/protein_embeddings/prot_t5_last/weighted_vecs/     (weighted vectors, last layer)
    results/protein_embeddings/prot_t5_layer_{n}/weighted_vecs/ (weighted vectors, layer n)

If all required files already exist the model is *not* loaded.
"""

import gc
import json
import os
import pickle
import re
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from transformers import T5EncoderModel, T5Tokenizer


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


def _weighted_mean(arr: np.ndarray, w: np.ndarray, normalize: bool = True) -> np.ndarray:
    """Length‑L weights → weighted mean over axis‑0."""
    w = np.asarray(w, dtype=float)
    if normalize:
        w = w / w.sum()
    return (arr * w[:, None]).sum(axis=0)


# --------------------------------------------------------------------------- #
#                              EMBEDDING BACK-END                             #
# --------------------------------------------------------------------------- #
def get_prot_t5_embeddings(
    seq_dict: Dict[str, str],
    *,
    batch_size = 2,
    setting  = "mean",              # 'mean' | 'residue' | 'weighted' | 'mean+weighted'
    all_layers = False,
    layer = None,           # 0-23 for residue setting
    only_save = False,
    id_to_seq = None,
    weights_df: Optional[pd.DataFrame] = None,
    weights_key_col: str = "PDB",
    weights_col: str = "Pred_BS_Scores"):

    # ----------------------- sanity checks -------------------------------- #
    # Parse setting - can be combination like "mean+weighted"
    settings = set(s.strip() for s in setting.split("+"))
    valid_settings = {"mean", "residue", "weighted"}
    assert settings.issubset(valid_settings), f"Invalid setting: {setting}. Valid: {valid_settings}"
    
    if all_layers:
        assert layer is None, "--layer is invalid when --all_layers is set"
    if "weighted" in settings:
        assert weights_df is not None, "--weights_file is required when setting includes 'weighted'"
        assert not all_layers, "weighted setting is incompatible with --all_layers"

    # ------------------------- path handling ------------------------------ #
    # Get repository root relative to this file
    ROOT = Path(__file__).resolve().parent.parent.parent
    idseq_path = ROOT / "results/sequence_id_to_sequence.pkl"
    id_to_seq = pickle.load(open(idseq_path, "rb")) if id_to_seq is None else id_to_seq
    assert all(k in id_to_seq and id_to_seq[k] == v for k, v in seq_dict.items()), (
        "Sequence mismatch between provided seq_dict and id_to_seq"
    )

    # Set up paths based on setting and layer
    paths = {}
    
    # Set up paths for each requested setting
    if "residue" in settings:
        if layer is None:
            paths["residue"] = ROOT / "results/full_protein_embeddings/t5_last"
        else:
            paths["residue"] = ROOT / f"results/full_protein_embeddings/t5_{layer}"
    
    if "mean" in settings:
        if layer is None:
            paths["mean"] = ROOT / "results/protein_embeddings/prot_t5_last/mean_vecs"
        else:
            paths["mean"] = ROOT / f"results/protein_embeddings/prot_t5_layer_{layer}/mean_vecs"
    
    if "weighted" in settings:
        if layer is None:
            paths["weighted"] = ROOT / "results/protein_embeddings/prot_t5_last/weighted_vecs"
        else:
            paths["weighted"] = ROOT / f"results/protein_embeddings/prot_t5_layer_{layer}/weighted_vecs"
    
    if all_layers:
        paths["all_layers"] = ROOT / "results/embeddings/prot_t5_all_layers"
    
    for p in paths.values():
        p.mkdir(parents=True, exist_ok=True)

    # ------------------------ skip existing files ------------------------- #
    if all_layers:
        key_to_exist = {
            k: (paths["all_layers"] / f"{k}.npy").exists()
            for k in seq_dict
        }
    else:
        # Check if all requested settings already exist for all sequences
        key_to_exist = {
            k: all((paths[s] / f"{k}.npy").exists() for s in settings)
            for k in seq_dict
        }

    if all(key_to_exist.values()):
        print("All required ProtT5 embeddings already on disk — skipping model load.")
        if only_save:
            return None
        return _load_existing_embeddings(seq_dict, paths, settings, all_layers, layer)

    # --------------------------- model load ------------------------------- #
    print("Loading ProtT5-XL UniRef50 ...")
    tokenizer = T5Tokenizer.from_pretrained("Rostlab/prot_t5_xl_uniref50", do_lower_case=False)
    model = T5EncoderModel.from_pretrained(
        "Rostlab/prot_t5_xl_uniref50", output_hidden_states=True
    )
    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    print(f"Using device: {device}")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()) // 1e6:,} M")

    # ----------------------- batching & encoding -------------------------- #
    missing_keys = [k for k, ok in key_to_exist.items() if not ok]
    print(f"Generating embeddings for {len(missing_keys)} new sequences")
    batches = [missing_keys[i:i + batch_size] for i in range(0, len(missing_keys), batch_size)]

    for batch_keys in tqdm(batches, desc="ProtT5 batches"):
        batch_seqs = [seq_dict[k] for k in batch_keys]
        # ProtT5 expects amino acids separated by space & ambiguous tokens as 'X'
        batch_strs = [
            " ".join(list(re.sub(r"[UZOB]", "X", s))) for s in batch_seqs
        ]
        token_data = tokenizer(
            batch_strs,
            return_tensors="pt",
            padding=True,
            add_special_tokens=True,
        ).to(device)

        with torch.no_grad():
            outputs = model(**token_data)
        hidden_states: Tuple[torch.Tensor, ...] = outputs.hidden_states  # len=25, 0=embeddings

        # lengths incl. <eos>; exclude from downstream
        seq_lens = (token_data["attention_mask"] == 1).sum(dim=1) - 1  # [batch]

        for idx, key in enumerate(batch_keys):
            L = seq_lens[idx].item()
            if all_layers:
                layer_means: List[np.ndarray] = []
                for hs in hidden_states[1:]:  # skip embedding layer
                    vec = hs[idx, :L].mean(dim=0).cpu().numpy()
                    layer_means.append(vec)
                stack = np.stack(layer_means)             # [24, hidden]
                np.save(paths["all_layers"] / f"{key}.npy", stack)

            else:
                # Get the appropriate layer
                if layer is None:
                    layer_tensor = hidden_states[-1]           # last layer
                else:
                    layer_tensor = hidden_states[layer + 1]    # skip embed layer
                
                residue_emb = layer_tensor[idx, :L].cpu().numpy()  # [L, H]
                
                # Save residue embeddings if requested
                if "residue" in settings:
                    np.save(paths["residue"] / f"{key}.npy", residue_emb)
                
                # Compute and save mean if requested
                if "mean" in settings:
                    mean_vec = residue_emb.mean(axis=0)
                    np.save(paths["mean"] / f"{key}.npy", mean_vec)
                
                # Compute and save weighted if requested
                if "weighted" in settings:
                    # Fetch weights for this sequence
                    weights = _fetch_weights(key, weights_df, weights_key_col, weights_col)
                    
                    # Ensure weights match the embedding length
                    if len(weights) != L:
                        raise ValueError(
                            f"Weight length ({len(weights)}) does not match embedding length ({L}) for {key}"
                        )
                    
                    # Compute weighted mean
                    weighted_vec = _weighted_mean(residue_emb, weights, normalize=True)
                    np.save(paths["weighted"] / f"{key}.npy", weighted_vec)

        # ------------------- memory hygiene per batch -------------------- #
        del token_data, outputs, hidden_states
        torch.cuda.empty_cache()
        gc.collect()

    # ---------------------- return (optional) ----------------------------- #
    if only_save:
        return None
    return _load_existing_embeddings(seq_dict, paths, settings, all_layers, layer)


# --------------------------------------------------------------------------- #
#                               HELPER ROUTINE                                #
# --------------------------------------------------------------------------- #
def _load_existing_embeddings(
    seq_dict: Dict[str, str],
    paths: Dict[str, Path],
    settings: set,
    all_layers: bool,
    layer):
    """
    Load embeddings that are now guaranteed to exist.
    """
    if all_layers:
        return {
            k: np.load(paths["all_layers"] / f"{k}.npy")
            for k in seq_dict
        }
    elif len(settings) == 1:
        # Single setting - return flat dict
        setting = next(iter(settings))
        return {
            k: np.load(paths[setting] / f"{k}.npy")
            for k in seq_dict
        }
    else:
        # Multiple settings - return nested dict
        return {
            k: {
                s: np.load(paths[s] / f"{k}.npy")
                for s in settings
            }
            for k in seq_dict
        }


# --------------------------------------------------------------------------- #
#                             SCRIPT ENTRY POINT                              #
# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    """
    Extract ProtT5 embeddings for unique sequences.
    Default: Runs for layer 19 and last layer (residue setting).
    
    Usage:
        # Default behavior (residue embeddings)
        python t5_embeddings.py
        
        # Mean embeddings
        python t5_embeddings.py --setting mean
        
        # Weighted embeddings using binding site scores
        python t5_embeddings.py --setting weighted --weights_file path/to/binding_sites.tsv
        
        # Custom sequence file and layers
        python t5_embeddings.py --seq_file path/to/sequences.txt --layers 19 None
    """
    import argparse
    from pathlib import Path
    
    parser = argparse.ArgumentParser(
        description='Extract ProtT5 protein embeddings',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Default: residue embeddings for layer 19 and last layer
  python t5_embeddings.py
  
  # Mean embeddings
  python t5_embeddings.py --setting mean --layers 19 None
  
  # Weighted embeddings using binding site predictions
  python t5_embeddings.py --setting weighted --weights_file results/binding_sites/binding_sites_all.tsv --layers 19 None
  
  # Both mean and weighted (computes residue once, derives both)
  python t5_embeddings.py --setting mean+weighted --weights_file results/binding_sites/binding_sites_all.tsv --layers 19 None
  
  # All three: residue, mean, and weighted
  python t5_embeddings.py --setting residue+mean+weighted --weights_file results/binding_sites/binding_sites_all.tsv
  
  # Custom weights column names
  python t5_embeddings.py --setting weighted --weights_file my_weights.tsv --weights_key_col sequence_id --weights_col my_weights
  
  # Single layer only
  python t5_embeddings.py --layers 19
        """
    )
    
    parser.add_argument(
        '--seq_file',
        type=str,
        default=None,
        help='Path to text file containing sequence IDs (one per line). Default: data/unique_seq_ids.txt'
    )
    
    parser.add_argument(
        '--layers',
        type=str,
        nargs='+',
        default=None,
        help='Layer number(s) to extract (0-23) or "None" for last layer. Default: [19, None]'
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
        '--all_layers',
        action='store_true',
        help='Extract embeddings from all 24 layers (overrides --layers)'
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
    
    args = parser.parse_args()
    
    # Validate arguments
    settings_requested = set(s.strip() for s in args.setting.split("+"))
    valid_settings = {"mean", "residue", "weighted"}
    if not settings_requested.issubset(valid_settings):
        parser.error(f"Invalid setting: {args.setting}. Valid options: mean, residue, weighted (can combine with +)")
    
    if "weighted" in settings_requested and args.weights_file is None:
        parser.error("--weights_file is required when --setting includes 'weighted'")
    
    # Paths relative to repository root
    ROOT = Path(__file__).resolve().parent.parent.parent
    SEQ_ID_TO_SEQ = ROOT / "results/sequence_id_to_sequence.pkl"
    
    # Determine sequence file
    if args.seq_file:
        seq_file = Path(args.seq_file)
    else:
        seq_file = ROOT / "data/unique_seq_ids.txt"
    
    # Load sequence ID to sequence mapping
    id_to_seq: Dict[str, str] = pickle.load(open(SEQ_ID_TO_SEQ, "rb"))
    
    # Load unique sequence IDs from file
    with open(seq_file, "r") as f:
        seq_ids = [line.strip() for line in f if line.strip()]
    
    assert all(sid in id_to_seq for sid in seq_ids), (
        "Some sequence IDs in the sequence file are missing from the ID-to-sequence mapping."
    )

    # Build sequence dictionary
    seq_dict = {sid: id_to_seq[sid] for sid in seq_ids}
    print(f"Loaded {len(seq_dict)} unique sequences from {seq_file}")
    
    # Load weights file if needed
    weights_df = None
    if "weighted" in settings_requested:
        print(f"Loading weights from {args.weights_file}")
        weights_df = pd.read_csv(args.weights_file, sep='\t')
        print(f"Loaded weights for {len(weights_df)} sequences")
    
    # Determine layers to process
    if args.all_layers:
        # Extract all layers mode
        print(f"\n{'='*70}")
        print(f"Extracting ProtT5 embeddings for all 24 layers")
        print(f"{'='*70}")
        
        embeddings = get_prot_t5_embeddings(
            seq_dict,
            batch_size=args.batch_size,
            setting=args.setting,
            all_layers=True,
            layer=None,
            only_save=True,
            id_to_seq=id_to_seq,
            weights_df=weights_df,
            weights_key_col=args.weights_key_col,
            weights_col=args.weights_col,
        )
        
        print(f"✓ Completed ProtT5 all-layers embedding extraction")
    else:
        # Process specific layers
        if args.layers:
            # Parse layers - convert "None" string to None, integers to int
            layers = []
            for l in args.layers:
                if l.lower() == 'none':
                    layers.append(None)
                else:
                    layers.append(int(l))
        else:
            # Default layers
            layers = [19, None]
        
        # Run embeddings for each layer
        for layer in layers:
            layer_name = "last" if layer is None else str(layer)
            print(f"\n{'='*70}")
            print(f"Extracting ProtT5 {args.setting} embeddings for layer {layer_name}")
            print(f"{'='*70}")
            
            embeddings = get_prot_t5_embeddings(
                seq_dict,
                batch_size=args.batch_size,
                setting=args.setting,
                all_layers=False,
                layer=layer,
                only_save=True,
                id_to_seq=id_to_seq,
                weights_df=weights_df,
                weights_key_col=args.weights_key_col,
                weights_col=args.weights_col,
            )
            
            print(f"✓ Completed ProtT5 layer {layer_name} {args.setting} embedding extraction")
    
    print(f"\n{'='*70}")
    print(f"✓ All ProtT5 embeddings complete for {len(seq_dict)} sequences")
    print(f"{'='*70}")
