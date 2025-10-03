"""
Job-specific utility functions for job submission and management.
"""
import os
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any
from django.conf import settings
from django.utils import timezone
from api.utils import get_experimental 


def validate_prediction_parameters(
    prediction_type: str, 
    kcat_method: Optional[str] = None, 
    km_method: Optional[str] = None
) -> Optional[str]:
    """
    Validate prediction type and method parameters.
    
    Args:
        prediction_type: Type of prediction ('kcat', 'Km', or 'both')
        kcat_method: Kcat prediction method if applicable
        km_method: Km prediction method if applicable
        
    Returns:
        Error message if validation fails, None if valid
    """
    if prediction_type not in ["kcat", "Km", "both"]:
        return 'Invalid prediction type. Expected "kcat", "Km", or "both".'
    
    if prediction_type in ["kcat", "both"]:
        if kcat_method not in ["TurNup", "DLKcat", "EITLEM", "UniKP"]:
            return "Invalid kcat method"
    
    if prediction_type in ["Km", "both"]:
        if km_method not in ["EITLEM", "UniKP"]:
            return "Invalid Km method"
    
    return None


def validate_sequence_handling_option(handle_long_seq: str) -> Optional[str]:
    """
    Validate the sequence handling option parameter.
    
    Args:
        handle_long_seq: Option for handling long sequences
        
    Returns:
        Error message if invalid, None if valid
    """
    if handle_long_seq not in ["truncate", "skip"]:
        return 'Invalid handleLongSeq value. Expected "truncate" or "skip".'
    return None


def determine_required_columns(prediction_type: str, kcat_method: str, km_method: str) -> List[str]:
    """
    Determine required columns based on prediction parameters.
    
    Args:
        prediction_type: Type of prediction
        kcat_method: Kcat method if applicable
        km_method: Km method if applicable
        
    Returns:
        List of required column names
    """
    required_columns = ["Protein Sequence"]
    
    if prediction_type in ["kcat", "both"]:
        if kcat_method == "TurNup":
            required_columns.extend(["Substrates", "Products"])
        elif kcat_method in ["DLKcat", "EITLEM", "UniKP"]:
            required_columns.append("Substrate")
    
    if prediction_type == "Km":
        if km_method in ["EITLEM", "UniKP"]:
            required_columns.append("Substrate")
    
    return required_columns


def create_job_directory(public_id: str) -> str:
    """
    Create directory structure for a job.
    
    Args:
        public_id: Public ID of the job
        
    Returns:
        Path to the created job directory
    """
    job_dir = os.path.join(settings.MEDIA_ROOT, "jobs", str(public_id))
    os.makedirs(job_dir, exist_ok=True)
    return job_dir


def save_job_input_file(file, job_dir: str) -> str:
    """
    Save input CSV file to job directory.
    
    Args:
        file: File object from request
        job_dir: Directory to save the file in
        
    Returns:
        Path to the saved file
    """
    file_path = os.path.join(job_dir, "input.csv")
    
    # Reset file pointer and read/save data
    file.seek(0)
    input_df = pd.read_csv(file)
    input_df.dropna(how="all", inplace=True)  # Remove empty rows
    input_df.to_csv(file_path, index=False)
    
    return file_path


def get_experimental_results(
    use_experimental: bool,
    kcat_method: str,
    dataframe: pd.DataFrame,
    prediction_type: str
) -> Optional[Dict[str, Any]]:
    """
    Get experimental results if requested and applicable.
    
    Args:
        use_experimental: Whether to use experimental data
        kcat_method: Kcat method being used
        dataframe: Input DataFrame
        prediction_type: Type of prediction
        
    Returns:
        Experimental results dictionary or None
    """
    if not use_experimental or kcat_method == "TurNup":
        return None
    
    if "Substrate" not in dataframe.columns:
        return None
    
    return get_experimental.lookup_experimental(
        dataframe["Protein Sequence"].tolist(),
        dataframe["Substrate"].tolist(),
        param_type=prediction_type,
    )


def extract_job_parameters_from_request(request) -> Dict[str, Any]:
    """
    Extract job parameters from HTTP request.
    
    Args:
        request: Django HTTP request object
        
    Returns:
        Dictionary containing extracted parameters
    """
    return {
        'use_experimental': request.POST.get("useExperimental") == "true",
        'prediction_type': request.POST.get("predictionType"),
        'kcat_method': request.POST.get("kcatMethod"),
        'km_method': request.POST.get("kmMethod"),
        'handle_long_sequences': request.POST.get("handleLongSequences"),
    }


def create_rate_limit_headers(daily_limit: int, remaining: int, ttl: int) -> Dict[str, str]:
    """
    Create rate limiting headers for HTTP response.
    
    Args:
        daily_limit: Daily limit for requests
        remaining: Remaining requests for today
        ttl: Time to live until reset
        
    Returns:
        Dictionary of headers
    """
    return {
        "X-RateLimit-Limit": str(daily_limit),
        "X-RateLimit-Remaining": str(max(0, remaining)),
        "X-RateLimit-Reset": str(ttl),
    }


def create_job_status_response_data(job) -> Dict[str, Any]:
    """
    Create response data dictionary for job status endpoint.
    
    Args:
        job: Job model instance
        
    Returns:
        Dictionary containing job status data
    """
    return {
        "public_id": job.public_id,
        "status": job.status,
        "submission_time": job.submission_time,
        "completion_time": job.completion_time,
        "server_time": timezone.now(),
        "elapsed_seconds": (
            int(max(0, (job.completion_time - job.submission_time).total_seconds()))
            if job.completion_time
            else int(max(0, (timezone.now() - job.submission_time).total_seconds()))
        ),
        "error_message": job.error_message,
        "total_molecules": job.total_molecules,
        "molecules_processed": job.molecules_processed,
        "invalid_molecules": job.invalid_molecules,
        "total_predictions": job.total_predictions,
        "predictions_made": job.predictions_made,
    }
