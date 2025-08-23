"""
Job service that orchestrates job submission and management workflows.
"""
from typing import Dict, Any, Optional, Tuple
from django.http import JsonResponse

from api.models import Job
from api.utils.validation_utils import (
    parse_csv_file,
    validate_required_columns,
)
from api.utils.job_utils import (
    validate_prediction_parameters,
    validate_sequence_handling_option,
    determine_required_columns,
    create_job_directory,
    save_job_input_file,
    get_experimental_results,
    create_rate_limit_headers,
    create_job_status_response_data,
)
from api.utils.quotas import (
    reserve_or_reject,
    get_client_ip,
    DAILY_LIMIT,
    get_or_create_user,
)
from api.tasks import (
    run_dlkcat_predictions,
    run_turnup_predictions,
    run_eitlem_predictions,
    run_unikp_predictions,
    run_both_predictions,
)


def process_job_submission(request, file) -> Tuple[Optional[JsonResponse], Optional[Dict[str, Any]]]:
    """
    Process job submission with validation and job creation.
    
    Args:
        request: Django HTTP request
        file: Uploaded CSV file
        
    Returns:
        Tuple of (error_response, success_data). If successful, error_response is None.
        If failed, success_data is None and error_response contains the error.
    """
    print("Received job submission request")
    
    # Extract job parameters from request
    from api.utils.job_utils import extract_job_parameters_from_request
    params = extract_job_parameters_from_request(request)
    
    # Validate parameters
    param_error = validate_prediction_parameters(
        params['prediction_type'], 
        params['kcat_method'], 
        params['km_method']
    )
    if param_error:
        return JsonResponse({"error": param_error}, status=400), None
    
    seq_handling_error = validate_sequence_handling_option(params['handle_long_sequences'])
    if seq_handling_error:
        return JsonResponse({"error": seq_handling_error}, status=400), None
    
    # Parse CSV file
    try:
        dataframe = parse_csv_file(file)
    except Exception as e:
        return JsonResponse({"error": f"Error reading file: {str(e)}"}, status=400), None
    
    # Validate required columns
    required_columns = determine_required_columns(
        params['prediction_type'], 
        params['kcat_method'], 
        params['km_method']
    )
    
    column_error = validate_required_columns(dataframe, required_columns)
    if column_error:
        return JsonResponse({"error": column_error}, status=400), None
    
    # Handle quota management
    ip_address = get_client_ip(request)
    quota_response = handle_quota_validation(ip_address, len(dataframe))
    if quota_response:
        return quota_response, None
    
    # Create and save job
    try:
        user = get_or_create_user(ip_address)
    except Exception as e:
        print(f"Error creating/updating user: {e}")
        user = None
    
    # Get experimental results if needed
    experimental_results = get_experimental_results(
        params['use_experimental'],
        params['kcat_method'],
        dataframe,
        params['prediction_type']
    )
    
    # Create job record
    job = create_job_record(params, ip_address, len(dataframe), user)
    
    # Save input file
    job_dir = create_job_directory(job.public_id)
    save_job_input_file(file, job_dir)
    
    # Dispatch prediction tasks
    dispatch_prediction_task(job.public_id, params, experimental_results)
    
    return None, {
        "message": "Job submitted successfully",
        "public_id": job.public_id
    }


def handle_quota_validation(ip_address: str, requested_rows: int) -> Optional[JsonResponse]:
    """
    Handle quota validation and return error response if quota exceeded.
    
    Args:
        ip_address: Client IP address
        requested_rows: Number of rows being requested
        
    Returns:
        JsonResponse with error if quota exceeded, None if allowed
    """
    allowed, remaining, ttl = reserve_or_reject(ip_address, requested_rows)
    
    rate_headers = create_rate_limit_headers(DAILY_LIMIT, remaining, ttl)
    
    if not allowed:
        error_response = JsonResponse({
            "error": (
                f"Upload rejected: daily limit exceeded. "
                f"{remaining} predictions remaining today; this upload requires {requested_rows}."
            )
        }, status=429)
        
        # Add rate limiting headers
        for key, value in rate_headers.items():
            error_response[key] = value
        
        return error_response
    
    return None


def create_job_record(params: Dict[str, Any], ip_address: str, requested_rows: int, user) -> Job:
    """
    Create and save a new job record.
    
    Args:
        params: Job parameters dictionary
        ip_address: Client IP address
        requested_rows: Number of rows in the request
        user: User model instance
        
    Returns:
        Created Job instance
    """
    job = Job(
        prediction_type=params['prediction_type'],
        kcat_method=params['kcat_method'],
        km_method=params['km_method'],
        status="Pending",
        handle_long_sequences=params['handle_long_sequences'],
        ip_address=ip_address,
        requested_rows=requested_rows,
        user=user,
    )
    job.save()
    print("Saved Job:", job.public_id)
    return job


def dispatch_prediction_task(public_id: str, params: Dict[str, Any], experimental_results: Optional[Dict]) -> None:
    """
    Dispatch appropriate prediction task based on parameters.
    
    Args:
        public_id: Job public ID
        params: Job parameters
        experimental_results: Experimental results if available
    """
    prediction_type = params['prediction_type']
    kcat_method = params['kcat_method']
    km_method = params['km_method']
    
    if prediction_type == "both":
        run_both_predictions.delay(public_id, experimental_results)
    elif prediction_type == "kcat":
        method_to_func = {
            "DLKcat": run_dlkcat_predictions,
            "TurNup": run_turnup_predictions,
            "EITLEM": run_eitlem_predictions,
            "UniKP": run_unikp_predictions,
        }
        pred_func = method_to_func.get(kcat_method)
        if pred_func:
            pred_func.delay(public_id, experimental_results)
    elif prediction_type == "Km":
        method_to_func = {
            "EITLEM": run_eitlem_predictions,
            "UniKP": run_unikp_predictions,
        }
        pred_func = method_to_func.get(km_method)
        if pred_func:
            pred_func.delay(public_id, experimental_results)
            print("Dispatching task to Celery:", prediction_type, kcat_method, km_method)


def get_job_status_data(job: Job) -> Dict[str, Any]:
    """
    Get formatted job status data.
    
    Args:
        job: Job model instance
        
    Returns:
        Dictionary containing job status information
    """
    return create_job_status_response_data(job)
