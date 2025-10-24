import pandas as pd
import numpy as np
import os
from pathlib import Path
from typing import Dict, Optional
from tqdm import tqdm
import torch
import pickle
import gc
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


def get_embeddings(seq_dict, batch_size=2, model=None, id_to_seq=None, setting='mean',all_layers=False, only_save=False, layer=None, weights_df: Optional[pd.DataFrame] = None, weights_key_col: str = "PDB", weights_col: str = "Pred_BS_Scores"):
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
    
    # Get repository root relative to this file
    ROOT = Path(__file__).resolve().parent.parent.parent
    assert all([key in id_to_seq.keys() and id_to_seq[key] == value for key, value in seq_dict.items()]), "Sequences must be in id_to_seq dictionary"
    print(f"Loaded {len(seq_dict)} sequences")

    # Set up paths based on setting and layer
    paths = {}
    
    # Set up paths for each requested setting
    if "residue" in settings:
        if layer is None:
            paths["residue"] = ROOT / f"results/full_protein_embeddings/{model}_last"
        else:
            paths["residue"] = ROOT / f"results/full_protein_embeddings/{model}_{layer}"
    
    if "mean" in settings:
        if layer is None:
            paths["mean"] = ROOT / f"results/protein_embeddings/{model}_last/mean_vecs"
        else:
            paths["mean"] = ROOT / f"results/protein_embeddings/{model}_layer_{layer}/mean_vecs"
    
    if "weighted" in settings:
        if layer is None:
            paths["weighted"] = ROOT / f"results/protein_embeddings/{model}_last/weighted_vecs"
        else:
            paths["weighted"] = ROOT / f"results/protein_embeddings/{model}_layer_{layer}/weighted_vecs"
    
    if all_layers:
        paths["all_layers"] = ROOT / f"results/embeddings/{model}_all_layers"
    
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

                with torch.no_grad():
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
                                
                                # Ensure weights match the embedding length
                                if len(weights) != len(res_emb):
                                    raise ValueError(
                                        f"Weight length ({len(weights)}) does not match embedding length ({len(res_emb)}) for {label}"
                                    )
                                
                                # Compute weighted mean
                                weighted_emb = _weighted_mean(res_emb, weights, normalize=True)
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
                        
                        # Ensure weights match the embedding length
                        if len(weights) != len(residue_emb):
                            raise ValueError(
                                f"Weight length ({len(weights)}) does not match embedding length ({len(residue_emb)}) for {key}"
                            )
                        # Compute weighted mean
                        weighted_emb = _weighted_mean(residue_emb, weights, normalize=True)
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
    if args.id_to_seq_file:
        id_to_seq_path = Path(args.id_to_seq_file)
        print(f"Loading sequence mapping from: {id_to_seq_path}")
    else:
        id_to_seq_path = SEQ_ID_TO_SEQ
    
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
    
    # Extract embeddings for each model-layer pair
    for model, layer in model_layer_pairs:
        print(f"\n{'='*70}")
        print(f"Extracting {model.upper()} {args.setting} embeddings for layer {layer}")
        print(f"{'='*70}")
        
        embd = get_embeddings(
            seq_dict, 
            batch_size=args.batch_size, 
            model=model, 
            id_to_seq=seq_id_to_seq,
            setting=args.setting, 
            all_layers=False, 
            only_save=True,
            layer=layer,
            weights_df=weights_df,
            weights_key_col=args.weights_key_col,
            weights_col=args.weights_col,
        )
        
        print(f"✓ Completed {model.upper()} layer {layer} {args.setting} embedding extraction")
    
    print(f"\n{'='*70}")
    print(f"✓ All embeddings complete for {len(seq_dict)} sequences")
    print(f"{'='*70}")
