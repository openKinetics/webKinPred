#!/usr/bin/env python3
"""
Download and Process Protein Embeddings to Mean and Weighted Vectors

This script processes large per-residue protein embedding files by:
1. Downloading embeddings from Google Drive in batches using rclone (fast bulk transfer)
2. Computing mean and binding-site-weighted vectors for each sequence
3. Saving the compressed vectors to a local results directory
4. Cleaning up temporary files before processing the next batch

The script uses rclone for efficient batch downloading, which is much faster than
reading through a mounted filesystem. Requires rclone to be installed and configured.

The script processes multiple embedding models and provides detailed progress reporting.
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import List, Dict, Tuple, Optional
import shutil
import tempfile
from tqdm import tqdm
import argparse
import sys
import logging
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)


# ============================================================================
# Configuration
# ============================================================================

# Determine repository root relative to this file (download_vecs.py is in code/utils/, so go up two levels)
ROOT = Path(__file__).resolve().parent.parent.parent

# Source directory on Google Drive
GDRIVE_EMBEDDINGS_ROOT = Path("/home/saleh/gdrive/Recon4IMD/WP2_MetNetwork/T2.5_Kinetics/results/KinForm/embeddings")

# Rclone remote path (adjust if your rclone remote has a different name)
RCLONE_REMOTE = "gdrive:Recon4IMD/WP2_MetNetwork/T2.5_Kinetics/results/KinForm/embeddings"

# Target directory for processed vectors
LOCAL_RESULTS_ROOT = ROOT / "results/protein_embeddings"

# Embedding models and their configurations
EMBEDDING_MODELS = [
    "esm2_layer_26",
    "esm2_layer_29",
    "esmc_layer_24",
    "esmc_layer_32",
    "prot_t5_layer_19",
    "prot_t5_last"
]

# Batch size for processing (number of files per batch)
DEFAULT_BATCH_SIZE = 500


# ============================================================================
# Helper Functions - Matching sequence_features.py Logic
# ============================================================================

def _fetch_weights(
    seq_id: str,
    df: pd.DataFrame,
    key_col: str,
    weights_col: str
) -> np.ndarray:
    """
    Fetch per-residue weights for a sequence from a DataFrame.
    
    This function MUST match the implementation in sequence_features.py exactly.
    
    Parameters
    ----------
    seq_id : str
        Sequence identifier
    df : pd.DataFrame
        DataFrame containing weights
    key_col : str
        Column name for sequence IDs
    weights_col : str
        Column name for weights (comma-separated string)
        
    Returns
    -------
    np.ndarray
        1-D float64 array of per-residue weights
        
    Raises
    ------
    ValueError
        If sequence not found in DataFrame
    """
    row = df.loc[df[key_col] == seq_id, weights_col]
    if row.empty:
        raise ValueError(f"No weights found in {weights_col} for sequence {seq_id}")
    return np.fromiter((float(x) for x in row.iloc[0].split(",")), dtype=np.float64)


def _fetch_cat_weights(
    seq_id: str,
    df: pd.DataFrame,
    key_col: str,
    L: int
) -> np.ndarray:
    """
    Fetch catalytic-site weights for a sequence.
    
    This function MUST match the implementation in sequence_features.py exactly.
    
    Parameters
    ----------
    seq_id : str
        Sequence identifier
    df : pd.DataFrame
        DataFrame containing catalytic site probabilities
    key_col : str
        Column name for sequence IDs
    L : int
        Post-truncation embedding length
        
    Returns
    -------
    np.ndarray
        Float64 array of catalytic site probabilities
        
    Raises
    ------
    ValueError
        If sequence not found or all_AS_probs has wrong length
    """
    row = df.loc[df[key_col] == seq_id, "all_AS_probs"]
    if row.empty:
        raise ValueError(f"No catalytic weights for sequence {seq_id}")
    probs = np.asarray(row.iloc[0], dtype=np.float64)
    if probs.shape[0] != 1024:
        raise ValueError("all_AS_probs must have length 1024")
    return probs[:L] if L <= 1024 else probs


def _fetch_cat_weights(
    seq_id: str,
    df: pd.DataFrame,
    key_col: str,
    L: int
) -> np.ndarray:
    """
    Fetch catalytic-site weights for a sequence.
    
    This function MUST match the implementation in sequence_features.py exactly.
    
    Parameters
    ----------
    seq_id : str
        Sequence identifier
    df : pd.DataFrame
        DataFrame containing catalytic site probabilities
    key_col : str
        Column name for sequence IDs
    L : int
        Post-truncation embedding length
        
    Returns
    -------
    np.ndarray
        Float64 array of catalytic site probabilities
        
    Raises
    ------
    ValueError
        If sequence not found or all_AS_probs has wrong length
    """
    row = df.loc[df[key_col] == seq_id, "all_AS_probs"]
    if row.empty:
        raise ValueError(f"No catalytic weights for sequence {seq_id}")
    probs = np.asarray(row.iloc[0], dtype=np.float64)
    if probs.shape[0] != 1024:
        raise ValueError("all_AS_probs must have length 1024")
    return probs[:L] if L <= 1024 else probs


def _weighted_mean(
    arr: np.ndarray,
    w: np.ndarray,
    normalize: bool = True
) -> np.ndarray:
    """
    Compute weighted mean over axis-0.
    
    This function MUST match the implementation in sequence_features.py exactly.
    
    Parameters
    ----------
    arr : np.ndarray
        Array with shape (L, D)
    w : np.ndarray
        Weights with shape (L,)
    normalize : bool
        Whether to normalize weights to sum to 1
        
    Returns
    -------
    np.ndarray
        Weighted mean vector with shape (D,)
    """
    w = np.asarray(w, dtype=np.float64)
    if normalize:
        w = w / w.sum()
    return (arr * w[:, None]).sum(axis=0)


# ============================================================================
# Core Processing Functions
# ============================================================================

def compute_vectors_for_single_file(
    embedding_file: Path,
    binding_site_df: pd.DataFrame
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute mean and binding-site-weighted vectors for a single embedding file.
    
    This function implements the exact same logic as used in sequence_features.py
    to ensure vector compatibility.
    
    Parameters
    ----------
    embedding_file : Path
        Path to .npy file containing (L, D) residue embeddings
    binding_site_df : pd.DataFrame
        DataFrame with binding site scores
        
    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
        (mean_vec, binding_weighted_vec)
        Each with shape (D,) where D is embedding dimension
        
    Raises
    ------
    ValueError
        If sequence weights cannot be found
    """
    # Extract sequence ID from filename (e.g., "Sequence 755.npy" -> "Sequence 755")
    seq_id = embedding_file.stem
    
    # Load residue embeddings
    emb = np.load(embedding_file, allow_pickle=True)
    if emb.ndim != 2:
        raise ValueError(f"Expected 2D array, got shape {emb.shape} for {seq_id}")
    
    L, D = emb.shape
    
    # Fetch binding site weights
    try:
        binding_weights = _fetch_weights(
            seq_id,
            binding_site_df,
            key_col="PDB",
            weights_col="Pred_BS_Scores"
        )
    except ValueError:
        # If binding site weights not found, use uniform weights
        logger.warning(f"Binding site weights not found for {seq_id}, using uniform weights")
        binding_weights = np.ones(L, dtype=np.float64)
    
    # Compute vectors
    mean_vec = emb.mean(axis=0).astype(np.float32)
    binding_weighted_vec = _weighted_mean(
        emb,
        binding_weights,
        normalize=True
    ).astype(np.float32)
    
    return mean_vec, binding_weighted_vec


def get_all_sequence_files(model_dir: Path) -> List[Path]:
    """
    Get all .npy files in a model directory, sorted by name.
    
    Parameters
    ----------
    model_dir : Path
        Directory containing .npy embedding files
        
    Returns
    -------
    List[Path]
        Sorted list of .npy file paths
    """
    files = sorted(model_dir.glob("*.npy"))
    return files


def create_output_structure(model_name: str) -> Tuple[Path, Path, Path]:
    """
    Create directory structure for a model's output vectors.
    
    Parameters
    ----------
    model_name : str
        Name of the embedding model (e.g., 'esm2_layer_26')
        
    Returns
    -------
    Tuple[Path, Path, Path]
        (model_root, mean_vecs_dir, weighted_vecs_dir)
    """
    model_root = LOCAL_RESULTS_ROOT / model_name
    mean_vecs_dir = model_root / "mean_vecs"
    weighted_vecs_dir = model_root / "weighted_vecs"
    
    mean_vecs_dir.mkdir(parents=True, exist_ok=True)
    weighted_vecs_dir.mkdir(parents=True, exist_ok=True)
    
    return model_root, mean_vecs_dir, weighted_vecs_dir


def copy_batch_with_rclone(
    batch_files: List[Path],
    model_name: str,
    temp_dir: Path
) -> List[Path]:
    """
    Copy a batch of files using rclone for fast bulk transfer.
    
    Parameters
    ----------
    batch_files : List[Path]
        List of source file paths
    model_name : str
        Name of the embedding model
    temp_dir : Path
        Temporary directory for batch processing
        
    Returns
    -------
    List[Path]
        List of successfully copied temporary file paths
    """
    # Create a manifest file with relative paths
    manifest_file = temp_dir / "manifest.txt"
    
    with open(manifest_file, "w") as f:
        for src_file in batch_files:
            # Get relative path: model_name/filename.npy
            relative_path = f"{model_name}/{src_file.name}"
            f.write(f"{relative_path}\n")
    
    # Run rclone copy with the manifest
    try:
        cmd = [
            "rclone", "copy",
            RCLONE_REMOTE,
            str(temp_dir),
            "--files-from", str(manifest_file),
            "--transfers", "16",
            "--checkers", "64",
            "--fast-list"
        ]
        
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            check=True
        )
        
        # Clean up manifest
        manifest_file.unlink()
        
        # Return list of successfully copied files
        # Files will be in temp_dir/model_name/
        model_temp_dir = temp_dir / model_name
        if model_temp_dir.exists():
            copied_files = list(model_temp_dir.glob("*.npy"))
            return copied_files
        else:
            logger.error(f"Model directory not created by rclone: {model_temp_dir}")
            return []
            
    except subprocess.CalledProcessError as e:
        logger.error(f"rclone copy failed: {e}")
        logger.error(f"stderr: {e.stderr}")
        return []
    except Exception as e:
        logger.error(f"Error during rclone copy: {e}")
        return []


def process_batch(
    batch_files: List[Path],
    model_name: str,
    temp_dir: Path,
    binding_site_df: pd.DataFrame,
    mean_vecs_dir: Path,
    weighted_vecs_dir: Path,
    progress_bar: tqdm
) -> Tuple[int, int]:
    """
    Process a single batch of embedding files.
    
    Parameters
    ----------
    batch_files : List[Path]
        List of source files to process
    model_name : str
        Name of the embedding model
    temp_dir : Path
        Temporary directory for batch processing
    binding_site_df : pd.DataFrame
        DataFrame with binding site scores
    mean_vecs_dir : Path
        Output directory for mean vectors
    weighted_vecs_dir : Path
        Output directory for weighted vectors
    progress_bar : tqdm
        Progress bar to update
        
    Returns
    -------
    Tuple[int, int]
        (success_count, error_count)
    """
    success_count = 0
    error_count = 0
    
    # Use rclone to copy batch to temp directory
    temp_batch_files = copy_batch_with_rclone(batch_files, model_name, temp_dir)
    
    if not temp_batch_files:
        logger.error(f"Failed to copy any files in batch for {model_name}")
        progress_bar.update(len(batch_files))
        return 0, len(batch_files)
    
    # Log if some files failed to copy
    if len(temp_batch_files) < len(batch_files):
        missing_count = len(batch_files) - len(temp_batch_files)
        logger.warning(f"{missing_count} files failed to copy via rclone")
        error_count += missing_count
    
    # Process each file in the batch
    for temp_file in temp_batch_files:
        try:
            # Compute vectors
            mean_vec, binding_weighted_vec = compute_vectors_for_single_file(
                temp_file,
                binding_site_df
            )
            
            # Save mean vector
            mean_output_path = mean_vecs_dir / temp_file.name
            np.save(mean_output_path, mean_vec)
            
            # Save binding-weighted vector
            weighted_output_path = weighted_vecs_dir / temp_file.name
            np.save(weighted_output_path, binding_weighted_vec)
            
            success_count += 1
            
        except Exception as e:
            logger.error(f"Error processing {temp_file.name} from {model_name}: {e}")
            error_count += 1
        finally:
            # Clean up temp file
            try:
                temp_file.unlink()
            except:
                pass
        
        progress_bar.update(1)
    
    # Clean up the model subdirectory created by rclone
    model_temp_dir = temp_dir / model_name
    if model_temp_dir.exists():
        try:
            shutil.rmtree(model_temp_dir)
        except Exception as e:
            logger.warning(f"Failed to clean up temp directory {model_temp_dir}: {e}")
    
    return success_count, error_count


def process_model(
    model_name: str,
    binding_site_df: pd.DataFrame,
    batch_size: int,
    skip_existing: bool = True
) -> Dict[str, int]:
    """
    Process all embeddings for a single model.
    
    Parameters
    ----------
    model_name : str
        Name of the embedding model
    binding_site_df : pd.DataFrame
        DataFrame with binding site scores
    batch_size : int
        Number of files to process per batch
    skip_existing : bool
        If True, skip files that already have processed vectors
        
    Returns
    -------
    Dict[str, int]
        Statistics dictionary with success, error, and skipped counts
    """
    logger.info(f"\n{'='*80}")
    logger.info(f"Processing model: {model_name}")
    logger.info(f"{'='*80}")
    
    # Setup directories
    source_dir = GDRIVE_EMBEDDINGS_ROOT / model_name
    if not source_dir.exists():
        logger.error(f"Source directory not found: {source_dir}")
        return {"success": 0, "error": 0, "skipped": 0, "total": 0}
    
    model_root, mean_vecs_dir, weighted_vecs_dir = create_output_structure(model_name)
    
    # Get all files
    all_files = get_all_sequence_files(source_dir)
    total_files = len(all_files)
    
    if total_files == 0:
        logger.warning(f"No .npy files found in {source_dir}")
        return {"success": 0, "error": 0, "skipped": 0, "total": 0}
    
    logger.info(f"Found {total_files:,} embedding files")
    
    # Filter out already processed files if requested
    files_to_process = []
    skipped_count = 0
    
    if skip_existing:
        for f in all_files:
            mean_output = mean_vecs_dir / f.name
            weighted_output = weighted_vecs_dir / f.name
            if mean_output.exists() and weighted_output.exists():
                skipped_count += 1
            else:
                files_to_process.append(f)
        logger.info(f"Skipping {skipped_count:,} already processed files")
    else:
        files_to_process = all_files
    
    if not files_to_process:
        logger.info("All files already processed!")
        return {
            "success": total_files - skipped_count,
            "error": 0,
            "skipped": skipped_count,
            "total": total_files
        }
    
    logger.info(f"Processing {len(files_to_process):,} files in batches of {batch_size}")
    
    # Process in batches
    success_count = 0
    error_count = 0
    
    with tqdm(total=len(files_to_process), desc=f"{model_name}", unit="file", ncols=100) as pbar:
        # Create batches
        for i in range(0, len(files_to_process), batch_size):
            batch_files = files_to_process[i:i + batch_size]
            batch_num = i // batch_size + 1
            total_batches = (len(files_to_process) + batch_size - 1) // batch_size
            
            pbar.set_description(f"{model_name} [Batch {batch_num}/{total_batches}]")
            
            # Create temporary directory for this batch
            with tempfile.TemporaryDirectory(prefix=f"kinform_batch_{model_name}_") as temp_dir:
                temp_path = Path(temp_dir)
                batch_success, batch_error = process_batch(
                    batch_files,
                    model_name,
                    temp_path,
                    binding_site_df,
                    mean_vecs_dir,
                    weighted_vecs_dir,
                    pbar
                )
                success_count += batch_success
                error_count += batch_error
    
    # Summary
    logger.info(f"\n{model_name} Summary:")
    logger.info(f"  ✓ Successfully processed: {success_count:,}")
    logger.info(f"  ✗ Errors: {error_count:,}")
    logger.info(f"  ⊘ Skipped (already done): {skipped_count:,}")
    logger.info(f"  Total: {total_files:,}")
    
    return {
        "success": success_count,
        "error": error_count,
        "skipped": skipped_count,
        "total": total_files
    }


def load_dataframes(
    binding_site_path: Optional[Path] = None
) -> pd.DataFrame:
    """
    Load binding site DataFrame.
    
    Parameters
    ----------
    binding_site_path : Optional[Path]
        Path to binding site predictions directory or single file (TSV/CSV).
        If directory, all prediction*.tsv files will be concatenated.
        
    Returns
    -------
    pd.DataFrame
        binding_site_df
    """
    if binding_site_path is None:
        # Default: look in results/binding_sites/
        binding_site_path = ROOT / "results/binding_sites/binding_sites_all.tsv"
    
    if binding_site_path.exists():
        # Single file
        logger.info(f"Using binding site file: {binding_site_path}")
        binding_site_df = pd.read_csv(binding_site_path, sep="\t")
    else:
        logger.warning(f"Binding site path not found: {binding_site_path}")
        binding_site_df = pd.DataFrame(columns=["PDB", "Pred_BS_Scores"])
    
    logger.info(f"Loaded {len(binding_site_df):,} binding site entries")
    
    return binding_site_df


# ============================================================================
# Main Execution
# ============================================================================

def main():
    """Main execution function."""
    parser = argparse.ArgumentParser(
        description="Download and process protein embeddings to mean and weighted vectors",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Process all models with default batch size (500)
  python download_vecs.py
  
  # Process specific models with custom batch size
  python download_vecs.py --models esm2_layer_26 esmc_layer_32 --batch-size 1000
  
  # Force reprocessing of all files
  python download_vecs.py --no-skip-existing
  
  # Provide custom data paths
  python download_vecs.py --binding-sites results/binding_sites/prediction.tsv
        """
    )
    
    parser.add_argument(
        "--models",
        nargs="+",
        default=EMBEDDING_MODELS,
        choices=EMBEDDING_MODELS,
        help="Embedding models to process (default: all)"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=DEFAULT_BATCH_SIZE,
        help=f"Number of files per batch (default: {DEFAULT_BATCH_SIZE})"
    )
    parser.add_argument(
        "--binding-sites",
        type=Path,
        default=None,
        help="Path to binding site predictions (directory or single file). "
             "If directory, all prediction*.tsv files will be concatenated. "
             "Default: {ROOT / 'results/binding_sites'}/"
    )
    parser.add_argument(
        "--no-skip-existing",
        action="store_true",
        help="Reprocess files even if outputs already exist"
    )
    parser.add_argument(
        "--parallel",
        action="store_true",
        help="Process multiple models in parallel (experimental)"
    )
    parser.add_argument(
        "--rclone-remote",
        type=str,
        default=None,
        help=f"Rclone remote path (default: gdrive:Recon4IMD/WP2_MetNetwork/T2.5_Kinetics/results/KinForm/embeddings)"
    )
    
    args = parser.parse_args()
    
    # Update global rclone remote if specified
    global RCLONE_REMOTE
    if args.rclone_remote:
        RCLONE_REMOTE = args.rclone_remote
    
    # Validate source directory
    if not GDRIVE_EMBEDDINGS_ROOT.exists():
        logger.error(f"Source directory not found: {GDRIVE_EMBEDDINGS_ROOT}")
        logger.error("Please ensure Google Drive is mounted and path is correct")
        sys.exit(1)
    
    # Create output root
    LOCAL_RESULTS_ROOT.mkdir(parents=True, exist_ok=True)
    
    # Load dataframes
    logger.info("Loading binding site data...")
    binding_site_df = load_dataframes(args.binding_sites)
    
    # Process models
    logger.info(f"\n{'='*80}")
    logger.info(f"Starting processing of {len(args.models)} model(s)")
    logger.info(f"{'='*80}")
    
    results = {}
    
    if args.parallel and len(args.models) > 1:
        logger.info("Processing models in parallel...")
        with ThreadPoolExecutor(max_workers=min(len(args.models), 3)) as executor:
            futures = {
                executor.submit(
                    process_model,
                    model,
                    binding_site_df,
                    args.batch_size,
                    not args.no_skip_existing
                ): model
                for model in args.models
            }
            
            for future in as_completed(futures):
                model = futures[future]
                try:
                    results[model] = future.result()
                except Exception as e:
                    logger.error(f"Model {model} failed with exception: {e}")
                    results[model] = {"success": 0, "error": 0, "skipped": 0, "total": 0}
    else:
        # Sequential processing
        for model in args.models:
            results[model] = process_model(
                model,
                binding_site_df,
                args.batch_size,
                not args.no_skip_existing
            )
    
    # Final summary
    logger.info(f"\n{'='*80}")
    logger.info("FINAL SUMMARY")
    logger.info(f"{'='*80}")
    
    total_success = sum(r["success"] for r in results.values())
    total_error = sum(r["error"] for r in results.values())
    total_skipped = sum(r["skipped"] for r in results.values())
    total_files = sum(r["total"] for r in results.values())
    
    for model, stats in results.items():
        logger.info(f"\n{model}:")
        logger.info(f"  ✓ Success: {stats['success']:,}")
        logger.info(f"  ✗ Errors: {stats['error']:,}")
        logger.info(f"  ⊘ Skipped: {stats['skipped']:,}")
    
    logger.info(f"\nOverall:")
    logger.info(f"  ✓ Total success: {total_success:,}")
    logger.info(f"  ✗ Total errors: {total_error:,}")
    logger.info(f"  ⊘ Total skipped: {total_skipped:,}")
    logger.info(f"  Total files: {total_files:,}")
    
    success_rate = (total_success / total_files * 100) if total_files > 0 else 0
    logger.info(f"  Success rate: {success_rate:.2f}%")
    
    logger.info(f"\nOutput directory: {LOCAL_RESULTS_ROOT}")
    logger.info(f"{'='*80}\n")
    
    return 0 if total_error == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
