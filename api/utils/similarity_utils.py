"""
Similarity analysis utility functions for protein sequence analysis.
These functions handle specific tasks following single responsibility principle.
"""
import tempfile
import os
import shutil
from typing import List, Dict, Tuple, Any
import pandas as pd
from api.utils.run_and_stream import run_and_stream

try:
    from webKinPred.config_docker import CONDA_PATH, TARGET_DBS
except ImportError:
    from webKinPred.config_local import CONDA_PATH, TARGET_DBS

TMP_DIR = os.environ.get("MMSEQS_TMP_DIR", "/tmp")
os.makedirs(TMP_DIR, exist_ok=True)


def extract_protein_sequences_from_csv(csv_file) -> List[str]:
    """
    Extract protein sequences from uploaded CSV file.
    
    Args:
        csv_file: Uploaded CSV file object
        
    Returns:
        List of protein sequences
        
    Raises:
        ValueError: If CSV doesn't contain required column or has no sequences
    """
    dataframe = pd.read_csv(csv_file)
    
    if "Protein Sequence" not in dataframe.columns:
        raise ValueError('CSV must contain a "Protein Sequence" column')
    
    sequences = dataframe["Protein Sequence"].dropna().tolist()
    
    if not sequences:
        raise ValueError("No valid protein sequences found in CSV")
    
    return sequences


def create_unique_sequence_mapping(sequences: List[str]) -> Tuple[List[str], Dict[str, str]]:
    """
    Create mapping between original sequences and unique sequences to avoid redundant analysis.
    
    Args:
        sequences: List of protein sequences (may contain duplicates)
        
    Returns:
        Tuple of (unique_sequences_list, sequence_to_id_mapping)
    """
    unique_sequences = list(dict.fromkeys(sequences))
    seq_to_unique_id = {seq: f"useq{idx}" for idx, seq in enumerate(unique_sequences)}
    return unique_sequences, seq_to_unique_id


def create_fasta_file(sequences: List[str], seq_to_id_mapping: Dict[str, str]) -> str:
    """
    Create a temporary FASTA file from protein sequences.
    
    Args:
        sequences: List of protein sequences
        seq_to_id_mapping: Mapping from sequence to unique identifier
        
    Returns:
        Path to the created FASTA file
    """
    with tempfile.NamedTemporaryFile(
        mode="w", delete=False, suffix=".fasta", dir=TMP_DIR
    ) as query_file:
        query_file_path = query_file.name
        for seq, unique_id in seq_to_id_mapping.items():
            query_file.write(f">{unique_id}\n{seq}\n")
    
    return query_file_path


def create_mmseqs_database(fasta_file_path: str, session_id: str) -> Tuple[str, str]:
    """
    Create MMseqs2 database from FASTA file.
    
    Args:
        fasta_file_path: Path to input FASTA file
        session_id: Session ID for logging
        
    Returns:
        Tuple of (query_db_path, temp_directory_path)
    """
    temp_query_dir = tempfile.mkdtemp(dir=TMP_DIR)
    query_db = os.path.join(temp_query_dir, "queryDB")
    
    run_and_stream([
        CONDA_PATH,
        "run",
        "-n",
        "mmseqs2_env",
        "mmseqs",
        "createdb",
        fasta_file_path,
        query_db,
    ], session_id=session_id)
    
    return query_db, temp_query_dir


def run_mmseqs_search(query_db: str, target_db: str, method_name: str, session_id: str) -> str:
    """
    Run MMseqs2 search against target database.
    
    Args:
        query_db: Path to query database
        target_db: Path to target database
        method_name: Name of the method (for logging)
        session_id: Session ID for logging
        
    Returns:
        Path to the result file
    """
    from api.services.progress_service import push_line
    
    tmp_dir = tempfile.mkdtemp(dir=TMP_DIR)
    result_db = os.path.join(tmp_dir, "resultDB")
    result_file = os.path.join(tmp_dir, "result.m8")
    
    try:
        push_line(session_id, f"--> [{method_name}] Running search")
        run_and_stream([
            CONDA_PATH,
            "run",
            "-n",
            "mmseqs2_env",
            "mmseqs",
            "search",
            query_db,
            target_db,
            result_db,
            tmp_dir,
            "--max-seqs",
            "5000",
            "-s",
            "7.5",
            "-e",
            "0.001",
            "-v",
            "0",
        ], session_id=session_id, fail_ok=True)
        
        push_line(session_id, f"--> [{method_name}] Converting alignments")
        run_and_stream([
            CONDA_PATH,
            "run",
            "-n",
            "mmseqs2_env",
            "mmseqs",
            "convertalis",
            query_db,
            target_db,
            result_db,
            result_file,
            "--format-output",
            "query,target,pident",
        ], session_id=session_id, fail_ok=True)
        
        return result_file
    except Exception:
        # Clean up on failure
        shutil.rmtree(tmp_dir, ignore_errors=True)
        raise


def parse_mmseqs_results(result_file: str, query_file_path: str) -> Tuple[Dict[str, float], Dict[str, float]]:
    """
    Parse MMseqs2 search results to extract identity scores.
    
    Args:
        result_file: Path to MMseqs2 result file
        query_file_path: Path to original query FASTA file
        
    Returns:
        Tuple of (max_identity_dict, mean_identity_dict)
    """
    identity_lists = {}
    
    # Parse result file if it exists
    if os.path.exists(result_file):
        with open(result_file, "r") as f:
            for line in f:
                fields = line.strip().split("\t")
                if len(fields) < 3:
                    continue
                query_id = fields[0]
                try:
                    pident = float(fields[2])
                except ValueError:
                    continue
                identity_lists.setdefault(query_id, []).append(pident)
    
    # Ensure all queries have at least 0.0 identity if no hits found
    with open(query_file_path, "r") as f:
        for line in f:
            if line.startswith(">"):
                qid = line[1:].strip()
                if qid not in identity_lists:
                    identity_lists[qid] = [0.0]
    
    # Calculate max and mean identities
    max_identity = {k: max(v) for k, v in identity_lists.items()}
    mean_identity = {k: (sum(v) / len(v)) for k, v in identity_lists.items()}
    
    return max_identity, mean_identity


def map_results_to_original_sequences(
    unique_results_max: Dict[str, float],
    unique_results_mean: Dict[str, float],
    original_sequences: List[str],
    seq_to_unique_id: Dict[str, str]
) -> Tuple[Dict[str, float], Dict[str, float]]:
    """
    Map results from unique sequences back to all original sequences.
    
    Args:
        unique_results_max: Max identity results for unique sequences
        unique_results_mean: Mean identity results for unique sequences
        original_sequences: Original list of sequences (may have duplicates)
        seq_to_unique_id: Mapping from sequence to unique identifier
        
    Returns:
        Tuple of (original_max_dict, original_mean_dict)
    """
    query_to_max = {}
    query_to_mean = {}
    
    for original_idx, seq in enumerate(original_sequences):
        original_seq_id = f"seq{original_idx + 1}"
        unique_id = seq_to_unique_id[seq]
        query_to_max[original_seq_id] = unique_results_max.get(unique_id, 0.0)
        query_to_mean[original_seq_id] = unique_results_mean.get(unique_id, 0.0)
    
    return query_to_max, query_to_mean


def calculate_identity_histogram(identity_values: Dict[str, float]) -> Tuple[Dict[str, int], Dict[str, float]]:
    """
    Calculate histogram of identity values rounded to nearest integer.
    
    Args:
        identity_values: Dictionary mapping sequence IDs to identity values
        
    Returns:
        Tuple of (count_histogram, percentage_histogram)
    """
    total_seqs = len(identity_values)
    histogram_counts = {str(i): 0 for i in range(101)}
    
    # Count occurrences for each rounded identity value
    for identity in identity_values.values():
        rounded = int(round(identity))
        rounded = max(0, min(100, rounded))  # Clamp to 0-100 range
        histogram_counts[str(rounded)] += 1
    
    # Convert to percentages
    histogram_percentages = {
        k: (v / total_seqs * 100) if total_seqs else 0.0
        for k, v in histogram_counts.items()
    }
    
    return histogram_counts, histogram_percentages


def calculate_average_similarity(identity_values: Dict[str, float]) -> float:
    """
    Calculate average similarity percentage from identity values.
    
    Args:
        identity_values: Dictionary mapping sequence IDs to identity values
        
    Returns:
        Average similarity as percentage (0-100)
    """
    total_seqs = len(identity_values)
    if not total_seqs:
        return 0.0
    
    total_similarity = sum(identity_values.values())
    return round(total_similarity / total_seqs * 100, 2)


def cleanup_temporary_files(*file_paths: str) -> None:
    """
    Clean up temporary files and directories.
    
    Args:
        file_paths: Paths to files/directories to remove
    """
    for path in file_paths:
        try:
            if os.path.isfile(path):
                os.remove(path)
            elif os.path.isdir(path):
                shutil.rmtree(path, ignore_errors=True)
        except Exception:
            pass  # Ignore cleanup errors
