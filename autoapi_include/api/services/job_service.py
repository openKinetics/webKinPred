"""
Job service that orchestrates job submission and management workflows.
"""

import logging
from typing import Any, Dict, Optional, Tuple

from api.models import Job
from api.services.about_stats_service import mark_about_stats_cache_stale
from api.tasks import run_multi_prediction, run_recon_xkg_cache_prediction
from api.utils.job_utils import (
    canonical_prediction_type,
    create_job_directory,
    create_job_status_response_data,
    create_rate_limit_headers,
    get_experimental_results,
    save_job_input_file,
    validate_prediction_parameters,
    validate_required_columns_for_methods,
    validate_sequence_handling_option,
)
from api.utils.quotas import (
    get_client_ip,
    get_or_create_user,
    get_user_daily_limit,
    reserve_or_reject,
)
from api.utils.validation_utils import (
    parse_csv_file,
    validate_column_emptiness,
    validate_products_column,
)
from django.http import JsonResponse

_log = logging.getLogger(__name__)


def process_job_submission(
    request, file
) -> Tuple[Optional[JsonResponse], Optional[Dict[str, Any]]]:
    """
    Process a job submission from the web UI.

    Extracts parameters from the Django request object (form fields + IP) and
    delegates to the shared ``process_job_submission_from_params`` function.

    Args:
        request: Django HTTP request (POST with multipart form data).
        file:    Uploaded CSV file extracted from request.FILES.

    Returns:
        Tuple of (error_response, success_data).
        On success: (None, {"message": ..., "public_id": ...})
        On failure: (JsonResponse with error, None)
    """
    from api.utils.job_utils import extract_job_parameters_from_request

    params = extract_job_parameters_from_request(request)
    if params.get("_parse_error"):
        return JsonResponse({"error": params["_parse_error"]}, status=400), None

    ip_address = get_client_ip(request)
    return process_job_submission_from_params(params, file, ip_address)


def process_job_submission_from_params(
    params: Dict[str, Any],
    file,
    ip_address: str,
    *,
    quota_subject: str | None = None,
    daily_limit: int | None = None,
    user=None,
) -> Tuple[Optional[JsonResponse], Optional[Dict[str, Any]]]:
    """
    Core job-submission logic, decoupled from the HTTP request object.

    This function is called by both the web-UI view (via
    ``process_job_submission``) and the public API v1 submit endpoint.  It
    accepts an explicit params dict and request IP so it can be used
    regardless of how the caller obtained those values.

    Args:
        params:     Dict with keys:
                      targets              – e.g. ["kcat"], ["Km", "kcat/Km"]
                      methods              – e.g. {"kcat":"DLKcat","Km":"UniKP"}
                      handle_long_sequences– "truncate" or "skip"
                      use_experimental     – bool
                      include_similarity_columns – bool
                      canonicalize_substrates – bool
                      disable_gpu_precompute – bool, internal benchmark toggle
        file:       A file-like object (Django InMemoryUploadedFile or
                    io.BytesIO) containing the CSV data.
        ip_address: Request source IP to store on the Job record.
        quota_subject: Identifier used for rate-limit accounting (defaults
                       to ``ip_address`` when omitted).
        daily_limit: Optional explicit daily limit for ``quota_subject``.
        user: Optional ApiUser instance to attach to the created Job.

    Returns:
        Tuple of (error_response, success_data).
        On success: (None, {"message": ..., "public_id": ...})
        On failure: (JsonResponse with error, None)
    """
    # --- Validate parameters ---------------------------------------------------

    param_error = validate_prediction_parameters(
        params["targets"],
        params["methods"],
    )
    if param_error:
        return JsonResponse({"error": param_error}, status=400), None

    seq_handling_error = validate_sequence_handling_option(params["handle_long_sequences"])
    if seq_handling_error:
        return JsonResponse({"error": seq_handling_error}, status=400), None

    # --- Parse and validate the CSV --------------------------------------------

    try:
        dataframe = parse_csv_file(file)
    except Exception as e:
        return JsonResponse({"error": f"Could not read CSV file: {e}"}, status=400), None

    required_columns_error = validate_required_columns_for_methods(
        dataframe,
        params["targets"],
        params["methods"],
    )
    if required_columns_error:
        return JsonResponse({"error": required_columns_error}, status=400), None

    # Ensure key columns are not mostly empty.
    substrate_column = (
        "Substrate"
        if "Substrate" in dataframe.columns
        else "Substrates" if "Substrates" in dataframe.columns else None
    )
    if substrate_column:
        emptiness_error = validate_column_emptiness(dataframe, substrate_column)
        if emptiness_error:
            return JsonResponse({"error": emptiness_error}, status=400), None

    sequence_error = validate_column_emptiness(dataframe, "Protein Sequence")
    if sequence_error:
        return JsonResponse({"error": sequence_error}, status=400), None

    product_errors = validate_products_column(dataframe)
    if product_errors:
        first = product_errors[0]
        position = first.get("position")
        location = f"row {first['row']}"
        if position is not None:
            location += f", product {position}"
        return JsonResponse(
            {
                "error": (
                    f"Invalid Products value at {location}: {first.get('value')!r}. "
                    "Products must be semicolon-separated SMILES or InChI strings."
                )
            },
            status=400,
        ), None

    # --- Quota -----------------------------------------------------------------

    quota_subject = quota_subject or ip_address
    quota_response = handle_quota_validation(
        quota_subject,
        len(dataframe),
        daily_limit=daily_limit,
    )
    if quota_response:
        return quota_response, None

    # --- Create job record and dispatch task -----------------------------------

    try:
        if user is None:
            user = get_or_create_user(ip_address)
    except Exception as e:
        _log.warning(
            "Could not create or update ApiUser",
            extra={
                "event": "job.api_user_sync_failed",
                "ip_address": ip_address,
                "exception_type": type(e).__name__,
            },
            exc_info=True,
        )
        user = None

    experimental_results = get_experimental_results(
        params["use_experimental"],
        params["methods"],
        params["targets"],
        dataframe,
    )

    job = create_job_record(
        params,
        ip_address,
        len(dataframe),
        user,
        quota_subject=quota_subject,
    )
    # New jobs can affect About-page totals once processed; mark cache stale.
    mark_about_stats_cache_stale()

    job_dir = create_job_directory(job.public_id)
    save_job_input_file(file, job_dir)

    if params.get("recon_xkg", False):
        dispatch_recon_xkg_cache_task(job.public_id, params, experimental_results)
    else:
        dispatch_prediction_task(job.public_id, params, experimental_results)

    return None, {
        "message": "Job submitted successfully",
        "public_id": job.public_id,
        "completed_immediately": False,
    }


def handle_quota_validation(
    quota_subject: str,
    requested_rows: int,
    *,
    daily_limit: int | None = None,
) -> Optional[JsonResponse]:
    """
    Handle quota validation and return error response if quota exceeded.

    Args:
        quota_subject: Quota subject identifier (IP or API-key subject).
        requested_rows: Number of rows being requested
        daily_limit: Optional explicit limit for this subject.

    Returns:
        JsonResponse with error if quota exceeded, None if allowed
    """
    allowed, remaining, ttl = reserve_or_reject(
        quota_subject,
        requested_rows,
        daily_limit=daily_limit,
    )

    header_limit = daily_limit if daily_limit is not None else get_user_daily_limit(quota_subject)
    rate_headers = create_rate_limit_headers(header_limit, remaining, ttl)

    if not allowed:
        error_response = JsonResponse(
            {
                "error": (
                    f"Upload rejected: daily limit exceeded. "
                    f"{remaining} predictions remaining today; "
                    f"this upload requires {requested_rows}."
                )
            },
            status=429,
        )

        for key, value in rate_headers.items():
            error_response[key] = value

        return error_response

    return None


def create_job_record(
    params: Dict[str, Any],
    ip_address: str,
    requested_rows: int,
    user,
    *,
    quota_subject: str | None = None,
) -> Job:
    """
    Create and save a new job record.

    Args:
        params: Job parameters dictionary
        ip_address: Request source IP
        requested_rows: Number of rows in the request
        user: User model instance
        quota_subject: Identifier used for quota accounting.

    Returns:
        Created Job instance
    """
    job = Job(
        prediction_type=canonical_prediction_type(params["targets"]),
        kcat_method=params["methods"].get("kcat"),
        km_method=params["methods"].get("Km"),
        kcat_km_method=params["methods"].get("kcat/Km"),
        canonicalize_substrates=params.get("canonicalize_substrates", True),
        recon_xkg=params.get("recon_xkg", False),
        status="Pending",
        handle_long_sequences=params["handle_long_sequences"],
        ip_address=ip_address,
        quota_subject=quota_subject or ip_address,
        requested_rows=requested_rows,
        user=user,
    )
    job.save()
    _log.info(
        "Job record created",
        extra={
            "event": "job.created",
            "job_public_id": job.public_id,
            "prediction_type": job.prediction_type,
            "requested_rows": requested_rows,
            "kcat_method": job.kcat_method,
            "km_method": job.km_method,
            "kcat_km_method": job.kcat_km_method,
        },
    )
    return job


def dispatch_prediction_task(
    public_id: str,
    params: Dict[str, Any],
    experimental_results: Optional[dict],
) -> None:
    """
    Dispatch the appropriate Celery prediction task based on job parameters.

    Uses one generic multi-target task that resolves each method at runtime.

    Args:
        public_id: Job public ID
        params: Job parameters
        experimental_results: Pre-fetched experimental results or None
    """
    targets = params["targets"]
    methods = params["methods"]
    canonicalize_substrates = params.get("canonicalize_substrates", True)
    include_similarity_columns = params.get("include_similarity_columns", True)
    disable_gpu_precompute = params.get("disable_gpu_precompute", False)
    recon_xkg = params.get("recon_xkg", False)

    result = run_multi_prediction.delay(
        public_id,
        targets,
        methods,
        experimental_results or {},
        canonicalize_substrates,
        include_similarity_columns,
        disable_gpu_precompute,
        recon_xkg,
    )
    _log.info(
        "Prediction task dispatched",
        extra={
            "event": "job.task_dispatched",
            "job_public_id": public_id,
            "celery_task_id": result.id,
            "targets": targets,
            "methods": methods,
            "canonicalize_substrates": canonicalize_substrates,
            "include_similarity_columns": include_similarity_columns,
            "disable_gpu_precompute": disable_gpu_precompute,
            "recon_xkg": recon_xkg,
        },
    )


def dispatch_recon_xkg_cache_task(
    public_id: str,
    params: Dict[str, Any],
    experimental_results: Optional[dict],
) -> None:
    """Dispatch ReconXKG cache preflight/assembly to the dedicated cache queue."""
    targets = params["targets"]
    methods = params["methods"]
    canonicalize_substrates = params.get("canonicalize_substrates", True)
    include_similarity_columns = params.get("include_similarity_columns", True)
    disable_gpu_precompute = params.get("disable_gpu_precompute", False)

    result = run_recon_xkg_cache_prediction.apply_async(
        args=[
            public_id,
            targets,
            methods,
            experimental_results or {},
            canonicalize_substrates,
            include_similarity_columns,
            disable_gpu_precompute,
        ],
        queue="webkinpred-cache",
    )
    _log.info(
        "ReconXKG cache task dispatched",
        extra={
            "event": "job.recon_xkg_cache_task_dispatched",
            "job_public_id": public_id,
            "celery_task_id": result.id,
            "targets": targets,
            "methods": methods,
            "canonicalize_substrates": canonicalize_substrates,
            "include_similarity_columns": include_similarity_columns,
            "disable_gpu_precompute": disable_gpu_precompute,
            "recon_xkg": True,
        },
    )


def get_job_status_data(job: Job) -> Dict[str, Any]:
    """
    Get formatted job status data.

    Args:
        job: Job model instance

    Returns:
        Dictionary containing job status information
    """
    return create_job_status_response_data(job)
