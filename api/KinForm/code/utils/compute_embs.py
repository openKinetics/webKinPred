from __future__ import annotations

import subprocess
import tempfile
import warnings
from pathlib import Path
from typing import Dict, Tuple, List
import pickle
# Silence warnings
warnings.filterwarnings("ignore", category=UserWarning, message=".*enable_nested_tensor.*")
# ──────────────────────────── local imports ───────────────────────── #
from config import COMPUTED_EMBEDDINGS_PATHS, ROOT, ESM_BIN, ESMC_BIN, T5_BIN
def _embs_exist(seq_id, pllm_name):
    embeddings_paths = COMPUTED_EMBEDDINGS_PATHS[pllm_name]
    for dir_path in embeddings_paths:
        for emb_type in ['mean_vecs', 'weighted_vecs']:
            emb_file = dir_path / emb_type / f"{seq_id}.npy"
            if not emb_file.exists():
                return False
    return True

def embeddings_exist(seq_ids: List[str]) -> Dict[str, List[bool]]:
    """
    Input:
        sequences : List[str]
            List of protein sequences' IDs
    Output:
        esm2_exist : List[bool]
            List indicating whether ESM-2 embeddings exist for each sequence
        esmc_exist : List[bool]
            List indicating whether ESM-C embeddings exist for each sequence
        t5_exist : List[bool]
            List indicating whether Prot-T5 embeddings exist for each sequence
    """
    pllms = ["esmc", "esm2","t5"]
    bool_lists = {pllm: [] for pllm in pllms}
    for seq_id in seq_ids:
        for pllm in pllms:
            bool_lists[pllm].append(_embs_exist(seq_id, pllm))
    return bool_lists

def _compute_esm2(sequences: List[str], seq_id_to_sequence: Dict[str, str]) -> Tuple[Dict[str, bool], Dict[str, str]]:
    seq_to_id = {v: k for k, v in seq_id_to_sequence.items()}
    seq_ids = [seq_to_id[seq] for seq in sequences]
    script_path = ROOT / "code" / "protein_embeddings" / "prot_embeddings.py"
    with tempfile.TemporaryDirectory() as tmpdir:
        temp_seq_file_path = Path(tmpdir) / "temp_sequences.txt"
        with open(temp_seq_file_path, "w") as temp_seq_file:
            for seq_id in seq_ids:
                temp_seq_file.write(f"{seq_id}\n")
        
        # Create a Python 3.7 compatible pickle file (protocol 4) with only needed sequences
        temp_id_to_seq_path = Path(tmpdir) / "temp_id_to_seq.pkl"
        temp_id_to_seq = {seq_id: seq_id_to_sequence[seq_id] for seq_id in seq_ids}
        with open(temp_id_to_seq_path, "wb") as f:
            pickle.dump(temp_id_to_seq, f, protocol=4)
        
        weights_file = ROOT / "results" / "binding_sites" / "binding_sites_all.tsv"
        setting = "mean+weighted"
        command = [ESM_BIN, str(script_path),
                   "--seq_file", str(temp_seq_file_path),
                   "--models", "esm2",
                   "--setting", setting,
                   "--weights_file", str(weights_file),
                   "--id_to_seq_file", str(temp_id_to_seq_path),  # Pass the compatible pickle
        ]
        subprocess.run(command, check=True)
    # After computation, check which succeeded
    computed_dict = {}
    reasons_dict = {}
    for seq in sequences:
        seq_id = seq_to_id[seq]
        mean_vec_paths = [ROOT / "results" / "protein_embeddings" / "esm2_layer_26" / "mean_vecs" / f"{seq_id}.npy", ROOT / "results" / "protein_embeddings" / "esm2_layer_29" / "mean_vecs" / f"{seq_id}.npy"]
        weighted_vec_paths = [ROOT / "results" / "protein_embeddings" / "esm2_layer_26" / "weighted_vecs" / f"{seq_id}.npy", ROOT / "results" / "protein_embeddings" / "esm2_layer_29" / "weighted_vecs" / f"{seq_id}.npy"]
        if all(p.exists() for p in mean_vec_paths) and all(p.exists() for p in weighted_vec_paths):
            computed_dict[seq_id] = True
            reasons_dict[seq_id] = None
        else:
            computed_dict[seq_id] = False
            reasons_dict[seq_id] = "Failed to compute ESM2 embeddings."
    return computed_dict, reasons_dict

def _compute_t5(sequences: List[str], seq_id_to_sequence: Dict[str, str]) -> Tuple[Dict[str, bool], Dict[str, str]]:
    seq_to_id = {v: k for k, v in seq_id_to_sequence.items()}
    seq_ids = [seq_to_id[seq] for seq in sequences]
    script_path = ROOT / "code" / "protein_embeddings" / "t5_embeddings.py"
    with tempfile.TemporaryDirectory() as tmpdir:
        temp_seq_file_path = Path(tmpdir) / "temp_sequences.txt"
        with open(temp_seq_file_path, "w") as temp_seq_file:
            for seq_id in seq_ids:
                temp_seq_file.write(f"{seq_id}\n")
        weights_file = ROOT / "results" / "binding_sites" / "binding_sites_all.tsv"
        setting = "mean+weighted"
        command = [T5_BIN, str(script_path),
                   "--seq_file", str(temp_seq_file_path),
                   "--setting", setting,
                   "--weights_file", str(weights_file),
        ]
        subprocess.run(command, check=True)
    # After computation, check which succeeded
    computed_dict = {}
    reasons_dict = {}
    for seq in sequences:
        seq_id = seq_to_id[seq]
        mean_vec_paths = [ROOT / "results" / "protein_embeddings" / "prot_t5_last" / "mean_vecs" / f"{seq_id}.npy", ROOT / "results" / "protein_embeddings" / "prot_t5_layer_19" / "mean_vecs" / f"{seq_id}.npy"]
        weighted_vec_paths = [ROOT / "results" / "protein_embeddings" / "prot_t5_last" / "weighted_vecs" / f"{seq_id}.npy", ROOT / "results" / "protein_embeddings" / "prot_t5_layer_19" / "weighted_vecs" / f"{seq_id}.npy"]
        if all(p.exists() for p in mean_vec_paths) and all(p.exists() for p in weighted_vec_paths):
            computed_dict[seq_id] = True
            reasons_dict[seq_id] = None
        else:
            computed_dict[seq_id] = False
            reasons_dict[seq_id] = "Failed to compute Prot-T5 embeddings."
    return computed_dict, reasons_dict

def _compute_esmc(sequences: List[str], seq_id_to_sequence: Dict[str, str]) -> Tuple[Dict[str, bool], Dict[str, str]]:
    seq_to_id = {v: k for k, v in seq_id_to_sequence.items()}
    seq_ids = [seq_to_id[seq] for seq in sequences]
    script_path = ROOT / "code" / "protein_embeddings" / "prot_embeddings.py"
    with tempfile.TemporaryDirectory() as tmpdir:
        temp_seq_file_path = Path(tmpdir) / "temp_sequences.txt"
        with open(temp_seq_file_path, "w") as temp_seq_file:
            for seq_id in seq_ids:
                temp_seq_file.write(f"{seq_id}\n")
        
        temp_id_to_seq_path = Path(tmpdir) / "temp_id_to_seq.pkl"
        temp_id_to_seq = {seq_id: seq_id_to_sequence[seq_id] for seq_id in seq_ids}
        with open(temp_id_to_seq_path, "wb") as f:
            pickle.dump(temp_id_to_seq, f, protocol=4)
        
        weights_file = ROOT / "results" / "binding_sites" / "binding_sites_all.tsv"
        setting = "mean+weighted"
        command = [ESMC_BIN, str(script_path),
                   "--seq_file", str(temp_seq_file_path),
                   "--models", "esmc",
                   "--setting", setting,
                   "--weights_file", str(weights_file),
                   "--id_to_seq_file", str(temp_id_to_seq_path),  # Pass the compatible pickle
        ]
        subprocess.run(command, check=True)
    # After computation, check which succeeded
    computed_dict = {}
    reasons_dict = {}
    for seq in sequences:
        seq_id = seq_to_id[seq]
        mean_vec_paths = [ROOT / "results" / "protein_embeddings" / "esmc_layer_24" / "mean_vecs" / f"{seq_id}.npy", ROOT / "results" / "protein_embeddings" / "esmc_layer_32" / "mean_vecs" / f"{seq_id}.npy"]
        weighted_vec_paths = [ROOT / "results" / "protein_embeddings" / "esmc_layer_24" / "weighted_vecs" / f"{seq_id}.npy", ROOT / "results" / "protein_embeddings" / "esmc_layer_32" / "weighted_vecs" / f"{seq_id}.npy"]
        if all(p.exists() for p in mean_vec_paths) and all(p.exists() for p in weighted_vec_paths):
            computed_dict[seq_id] = True
            reasons_dict[seq_id] = None
        else:
            computed_dict[seq_id] = False
            reasons_dict[seq_id] = "Failed to compute ESMC embeddings."
    return computed_dict, reasons_dict
    

def _compute_all_emb(sequences: List[str], seq_id_to_sequence: Dict[str, str]) -> Tuple[Dict[str, bool], Dict[str, str]]:
    esmc_computed, esmc_reasons = _compute_esmc(sequences, seq_id_to_sequence)
    print(esmc_computed)
    print(esmc_reasons)
    raise
    esm2_computed, esm2_reasons = _compute_esm2(sequences, seq_id_to_sequence)
    t5_computed, t5_reasons = _compute_t5(sequences, seq_id_to_sequence)
    return (esmc_computed, esmc_reasons, esm2_computed, esm2_reasons, t5_computed, t5_reasons)