"""
Similarity analysis service that orchestrates the similarity workflow.
"""
import tempfile
import os
from typing import List, Dict, Any
from api.utils.similarity_utils import (
    extract_protein_sequences_from_csv,
    create_unique_sequence_mapping,
    create_fasta_file,
    create_mmseqs_database,
    run_mmseqs_search,
    parse_mmseqs_results,
    map_results_to_original_sequences,
    calculate_identity_histogram,
    calculate_average_similarity,
    cleanup_temporary_files,
)
from api.services.progress_service import push_line

try:
    from webKinPred.config_docker import TARGET_DBS
except ImportError:
    from webKinPred.config_local import TARGET_DBS


def analyze_sequence_similarity(csv_file, session_id: str = "default") -> Dict[str, Any]:
    """
    Analyze sequence similarity against target databases.
    
    Args:
        csv_file: Uploaded CSV file containing protein sequences
        session_id: Session ID for logging
        
    Returns:
        Dictionary containing similarity analysis results
        
    Raises:
        ValueError: If CSV is invalid or contains no sequences
        Exception: If analysis fails
    """
    # Extract sequences from CSV
    input_sequences = extract_protein_sequences_from_csv(csv_file)
    
    # Create unique sequence mapping to avoid redundant analysis
    unique_sequences, seq_to_unique_id = create_unique_sequence_mapping(input_sequences)
    
    # Create temporary FASTA file
    query_file_path = create_fasta_file(unique_sequences, seq_to_unique_id)
    temp_files_to_cleanup = [query_file_path]
    
    try:
        # Create MMseqs2 database
        query_db, temp_query_dir = create_mmseqs_database(query_file_path, session_id)
        temp_files_to_cleanup.append(temp_query_dir)
        
        # Process each target database
        method_histograms = {}
        
        for method, target_db in TARGET_DBS.items():
            push_line(session_id, f"==> Processing DB: {method}")
            
            # Run similarity analysis for this method
            method_result = analyze_similarity_for_method(
                query_db, target_db, query_file_path, method, 
                input_sequences, seq_to_unique_id, session_id
            )
            
            method_histograms[method] = method_result
        
        return method_histograms
        
    finally:
        # Clean up all temporary files
        cleanup_temporary_files(*temp_files_to_cleanup)


def analyze_similarity_for_method(
    query_db: str, target_db: str, query_file_path: str, method_name: str,
    original_sequences: List[str], seq_to_unique_id: Dict[str, str], session_id: str
) -> Dict[str, Any]:
    """
    Analyze similarity for a specific method/database.
    
    Args:
        query_db: Path to query database
        target_db: Path to target database
        query_file_path: Path to original FASTA file
        method_name: Name of the method
        original_sequences: Original sequence list
        seq_to_unique_id: Sequence to unique ID mapping
        session_id: Session ID for logging
        
    Returns:
        Dictionary containing method-specific results
    """
    result_file = None
    
    try:
        # Run MMseqs2 search
        result_file = run_mmseqs_search(query_db, target_db, method_name, session_id)
        
        # Parse results to get identity scores
        unique_max_identity, unique_mean_identity = parse_mmseqs_results(
            result_file, query_file_path
        )
        
        # Map results back to original sequences
        query_to_max, query_to_mean = map_results_to_original_sequences(
            unique_max_identity, unique_mean_identity, 
            original_sequences, seq_to_unique_id
        )
        
        # Calculate histograms
        histogram_max_counts, histogram_max_perc = calculate_identity_histogram(query_to_max)
        histogram_mean_counts, histogram_mean_perc = calculate_identity_histogram(query_to_mean)
        
        # Calculate averages
        average_max_similarity = calculate_average_similarity(query_to_max)
        average_mean_similarity = calculate_average_similarity(query_to_mean)
        
        push_line(session_id, f"--> [{method_name}] Aggregated {len(query_to_max)} sequences")
        
        return {
            "histogram_max": histogram_max_perc,
            "histogram_mean": histogram_mean_perc,
            "average_max_similarity": average_max_similarity,
            "average_mean_similarity": average_mean_similarity,
            "count_max": histogram_max_counts,
            "count_mean": histogram_mean_counts,
        }
        
    finally:
        # Clean up result file and its parent directory
        if result_file and os.path.exists(result_file):
            result_dir = os.path.dirname(result_file)
            cleanup_temporary_files(result_dir)
