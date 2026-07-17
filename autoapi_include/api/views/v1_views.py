"""
Public REST API — version 1 endpoints.

All endpoints live under the /api/v1/ prefix (see api/urls_v1.py).

Authentication
--------------
Most endpoints require an API key delivered as a Bearer token:

    Authorization: Bearer ak_<your_key>

The two exceptions are /health/ and /methods/, which are intentionally public
so that clients can discover available methods and check server status without
needing credentials.

Quota
-----
Quota accounting is keyed by API key identity (not source IP), so requests
from different hosts still share one quota bucket per key. The effective
daily limit for authenticated requests is the higher of:
  - the owning user's IP-level limit, and
  - the API key's optional custom limit.
"""

import io
import json
from typing import Any

import pandas as pd
from api.models import Job
from api.services.embedding_progress_service import get_embedding_progress
from api.services.gpu_embed_service import get_gpu_status
from api.services.gpu_precompute_status_service import get_gpu_precompute_status
from api.services.job_progress_service import get_active_stage_embedding, get_progress_summary
from api.services.job_service import process_job_submission_from_params
from api.services.result_service import serialize_result_csv
from api.services.similarity_service import analyze_sequence_similarity
from api.services.validation_service import validate_input_file
from api.utils.api_auth import require_api_key
from api.utils.job_utils import coerce_bool_param
from api.utils.quotas import get_quota_usage
from api.utils.recon_xkg import coerce_recon_xkg, resolve_recon_xkg
from django.http import FileResponse, JsonResponse
from django.utils import timezone
from django.utils.text import slugify
from django.views.decorators.csrf import csrf_exempt
from django.views.decorators.http import require_GET

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _json_error(message: str, status: int = 400) -> JsonResponse:
    """Return a consistently formatted error response."""
    return JsonResponse({"error": message}, status=status)


def _quota_dict(quota_subject: str, daily_limit: int | None = None) -> dict:
    """Return quota data in the camelCase shape expected by API clients."""
    usage = get_quota_usage(quota_subject, daily_limit=daily_limit)
    return {
        "limit": usage["limit"],
        "used": usage["used"],
        "remaining": usage["remaining"],
        "resetsInSeconds": usage["reset_in_seconds"],
    }


# ---------------------------------------------------------------------------
# GET /api/v1/health/
# ---------------------------------------------------------------------------


@csrf_exempt
def api_health(request):
    """
    Health check endpoint — no authentication required.

    Returns a simple JSON object confirming the service is up.
    """
    return JsonResponse(
        {
            "status": "ok",
            "service": "Open Kinetics Predictor API",
            "version": "1",
            "timestamp": timezone.now().isoformat(),
        }
    )


# ---------------------------------------------------------------------------
# GET /api/v1/gpu/status/
# ---------------------------------------------------------------------------


@csrf_exempt
@require_GET
def api_gpu_status(request):
    """
    Return a minimal GPU availability snapshot for the frontend.

    Internal fields (gpu_name, vram, active_jobs, raw GPU response) are
    deliberately stripped — the browser only needs to know online/offline.
    """
    status = get_gpu_status()
    public = {
        "configured": bool(status.get("configured", False)),
        "online": bool(status.get("online", False)),
        "mode": str(status.get("mode", "cpu")),
    }
    return JsonResponse(public)


# ---------------------------------------------------------------------------
# GET /api/v1/methods/
# ---------------------------------------------------------------------------


@csrf_exempt
def api_list_methods(request):
    """
    List all available prediction methods and their requirements.

    No authentication required — this endpoint acts as living documentation
    so clients can discover what methods exist and what CSV columns they need
    before writing any code.  The response is generated directly from the
    method registry, so it is always up to date when new methods are added.

    Response shape
    --------------
    {
      "methods": {
        "<method_key>": {
          "displayName":       str,
          "authors":           str,
          "publicationTitle":  str,
          "citationUrl":       str,
          "repoUrl":           str,
          "moreInfo":          str,
          "supports":          list[str],   // e.g. ["kcat", "Km", "kcat/Km"]
          "inputFormat":       str,         // backend contract: "single" or "multi"
          "acceptedCsvTypes":  list[str],   // UI/API input schemas accepted by the method
          "acceptedCsvTypesByTarget": object,
          "inputBehaviorByTarget": object,  // expanded_pair/native_multi/native_full_reaction
          "maxSeqLen":         int | null,  // null means no limit
          "requiredColumns":   list[str],   // includes "Protein Sequence"
          "substrateFormat":   str,
        },
        ...
      },
      "predictionTypes": ["kcat", "Km", "kcat/Km"],
      "longSequenceOptions": { "truncate": "...", "skip": "..." },
      "notes": { ... }
    }
    """
    from api.methods.registry import all_methods

    registry = all_methods()

    # Build the full method detail object for each key.
    def _method_obj(key, desc):
        max_len = None if desc.max_seq_len == float("inf") else int(desc.max_seq_len)
        required_cols = ["Protein Sequence"] + list(desc.col_to_kwarg.keys())
        accepted_by_target = {
            target: desc.accepted_csv_types_for_target(target)
            for target in desc.supports
        }
        csv_type_order = ["single", "multi", "full_reaction"]
        accepted_csv_types = [
            csv_type
            for csv_type in csv_type_order
            if any(csv_type in accepted for accepted in accepted_by_target.values())
        ]
        behavior_by_target = {
            target: desc.input_behavior(target)
            for target in desc.supports
        }
        if any(behavior == "native_full_reaction" for behavior in behavior_by_target.values()):
            substrate_fmt = (
                "Full reaction: semicolon-separated SMILES or InChI strings in "
                "Substrates and Products"
            )
        elif any(behavior == "native_multi" for behavior in behavior_by_target.values()):
            substrate_fmt = (
                "CatPred kcat consumes semicolon-separated Substrates as one combined "
                "input; CatPred Km predicts each substrate separately"
            )
        else:
            substrate_fmt = (
                "Use Substrate for one molecular input, or semicolon-separated Substrates. "
                "For Substrates inputs, kcat is reduced by maximum; Km and kcat/Km remain "
                "ordered arrays"
            )
        return {
            "id": key,
            "displayName": desc.display_name,
            "authors": desc.authors,
            "publicationTitle": desc.publication_title,
            "citationUrl": desc.citation_url,
            "repoUrl": desc.repo_url,
            "moreInfo": desc.more_info,
            "supports": desc.supports,
            "inputFormat": desc.input_format,
            "acceptedCsvTypes": accepted_csv_types,
            "acceptedCsvTypesByTarget": accepted_by_target,
            "inputBehaviorByTarget": behavior_by_target,
            "maxSeqLen": max_len,
            "requiredColumns": required_cols,
            "substrateFormat": substrate_fmt,
        }

    # Group methods by the prediction targets they support.
    methods_payload: dict[str, list[dict[str, Any]]] = {"kcat": [], "Km": [], "kcat/Km": []}
    for key, desc in registry.items():
        obj = _method_obj(key, desc)
        if "kcat" in desc.supports:
            methods_payload["kcat"].append(obj)
        if "Km" in desc.supports:
            methods_payload["Km"].append(obj)
        if "kcat/Km" in desc.supports:
            methods_payload["kcat/Km"].append(obj)

    return JsonResponse(
        {
            "methods": methods_payload,
            "predictionTypes": ["kcat", "Km", "kcat/Km"],
            "longSequenceOptions": {
                "truncate": "Shorten sequences that exceed the model's maximum length.",
                "skip": "Omit rows where the sequence exceeds the model's maximum length.",
            },
            "notes": {
                "targets": (
                    "Submit target selection via 'targets' (list) and method "
                    "selection via 'methods' (object mapping each selected target "
                    "to a method key)."
                ),
                "substrateLists": (
                    "Single-substrate methods accept semicolon-separated values in 'Substrates'. "
                    "Reaction kcat uses the maximum successful child prediction; Km and "
                    "kcat/Km use ordered JSON arrays. CatPred kcat consumes the complete "
                    "substrate set natively. TurNup additionally consumes 'Products'; other "
                    "methods preserve but ignore them."
                ),
                "quota": "Each input reaction row counts once against your daily quota.",
            },
        }
    )


# ---------------------------------------------------------------------------
# GET /api/v1/quota/
# ---------------------------------------------------------------------------


@require_api_key
def api_quota(request):
    """
    Return the current quota status for the authenticated key owner.

    Quota is tracked per-day (resetting at midnight UTC) and keyed by
    authenticated API key.
    """
    return JsonResponse(
        _quota_dict(
            request.api_quota_subject,
            daily_limit=request.api_daily_limit,
        )
    )


# ---------------------------------------------------------------------------
# POST /api/v1/submit/
# ---------------------------------------------------------------------------


@csrf_exempt
@require_api_key
def api_submit_job(request):
    """
    Submit a new prediction job.

    Accepts two content types:

    1. multipart/form-data — upload a CSV file plus form fields:
         file                  (required) — the CSV file
         targets               (required) — JSON array from
                                             ["kcat", "Km", "kcat/Km"]
         methods               (required) — JSON object mapping selected target
                                             to method key
         handleLongSequences   (optional, default "truncate") — "truncate" or "skip"
         useExperimental       (optional, default "false")   — "true" or "false"
         includeSimilarityColumns (optional, default "true") — "true" or "false"
         canonicalizeSubstrates (optional, default "true")  — "true" or "false"
         disableGpuPrecompute (optional, default "false")   — internal benchmark toggle

    2. application/json — send data directly as a JSON body:
         {
           "targets": ["kcat"],
           "methods": {"kcat": "DLKcat"},
           "handleLongSequences": "truncate",
           "useExperimental": false,
           "includeSimilarityColumns": true,
           "canonicalizeSubstrates": true,
           "disableGpuPrecompute": false,
           "data": [
             {"Protein Sequence": "MKTL...", "Substrates": "CC(=O)O;O"},
             ...
           ]
         }

    Both paths call the same underlying job-submission service, so validation,
    quota accounting, and task dispatch behave identically.

    On success, returns a JSON object with:
      - jobId       — use this to poll /status/ and download /result/
      - status      — Pending; poll statusUrl for the current persisted job state
      - statusUrl   — convenience URL for polling
      - resultUrl   — convenience URL for downloading results
      - quota       — your remaining quota after this submission
    """
    if request.method != "POST":
        return _json_error("This endpoint only accepts POST requests.", 405)

    content_type = request.content_type or ""

    if "application/json" in content_type:
        csv_file, params, error = _parse_json_body(request)
    else:
        csv_file, params, error = _parse_multipart_body(request)

    if error:
        return error

    # Resolve the (undocumented) recon_xkg flag here, where the authenticated
    # API key is available. Non-allowlisted requests are silently downgraded to
    # a normal job — no distinctive error — and only audited server-side.
    params["recon_xkg"] = resolve_recon_xkg(
        request.api_key,
        params.get("recon_xkg", False),
    )

    error_response, success_data = process_job_submission_from_params(
        params,
        csv_file,
        request.api_request_ip,
        quota_subject=request.api_quota_subject,
        daily_limit=request.api_daily_limit,
        user=request.api_user,
    )

    if error_response:
        return error_response

    if success_data is None:
        return _json_error("Job submission failed unexpectedly.", status=500)

    public_id = success_data["public_id"]

    response_data = {
        "jobId": public_id,
        "status": "Pending",
        "statusUrl": f"/api/v1/status/{public_id}/",
        "resultUrl": f"/api/v1/result/{public_id}/",
        "quota": _quota_dict(
            request.api_quota_subject,
            daily_limit=request.api_daily_limit,
        ),
    }
    return JsonResponse(response_data, status=201)


def _parse_multipart_body(request):
    """
    Extract CSV file and parameters from a multipart/form-data request.

    Returns (csv_file, params, error).  On success, error is None.
    """
    csv_file = request.FILES.get("file")

    if not csv_file:
        return None, None, _json_error("No file provided. Include 'file' as a multipart field.")

    if not csv_file.name.lower().endswith(".csv"):
        return None, None, _json_error("The uploaded file must have a .csv extension.")

    params = {
        "targets": [],
        "methods": {},
        "handle_long_sequences": request.POST.get("handleLongSequences", "truncate"),
        "use_experimental": coerce_bool_param(
            request.POST.get("useExperimental"),
            default=False,
        ),
        "include_similarity_columns": coerce_bool_param(
            request.POST.get("includeSimilarityColumns"),
            default=True,
        ),
        "canonicalize_substrates": coerce_bool_param(
            request.POST.get("canonicalizeSubstrates"),
            default=True,
        ),
        "disable_gpu_precompute": coerce_bool_param(
            request.POST.get("disableGpuPrecompute"),
            default=False,
        ),
        # Undocumented: kept out of the public schema/docs on purpose. Holds the
        # *requested* value here; authorization is resolved in the view.
        "recon_xkg": coerce_recon_xkg(request.POST.get("recon_xkg")),
    }

    targets_raw = request.POST.get("targets", "")
    methods_raw = request.POST.get("methods", "")
    try:
        params["targets"] = json.loads(targets_raw) if targets_raw else []
    except json.JSONDecodeError:
        return (
            None,
            None,
            _json_error(
                'Invalid \'targets\' value. Expected a JSON array, for example: ["kcat", "Km"].'
            ),
        )

    try:
        params["methods"] = json.loads(methods_raw) if methods_raw else {}
    except json.JSONDecodeError:
        return (
            None,
            None,
            _json_error(
                "Invalid 'methods' value. Expected a JSON object, for example: "
                '{"kcat":"DLKcat","Km":"UniKP"}.'
            ),
        )

    return csv_file, params, None


def _parse_json_body(request):
    """
    Extract CSV data and parameters from an application/json request body.

    The caller supplies the data as an array of row objects under the key
    "data".  We convert this to an in-memory CSV file so the same downstream
    validation and processing code can handle it without modification.

    Returns (csv_file, params, error).  On success, error is None.
    """
    try:
        body = json.loads(request.body)
    except (json.JSONDecodeError, UnicodeDecodeError):
        return None, None, _json_error("Request body is not valid JSON.")

    rows = body.get("data")

    if not rows or not isinstance(rows, list):
        return (
            None,
            None,
            _json_error(
                "'data' must be a non-empty array of row objects. "
                "Each object should map column names to values, e.g. "
                '{"Protein Sequence": "MKTL...", "Substrate": "CC(=O)O"}.'
            ),
        )

    if len(rows) > 10_000:
        return (
            None,
            None,
            _json_error(
                f"JSON body submission is limited to 10,000 rows. "
                f"You submitted {len(rows):,}.  Use CSV file upload for larger datasets."
            ),
        )

    # Convert the list of dicts to an in-memory CSV file-like object.
    try:
        df = pd.DataFrame(rows)
        buffer = io.StringIO()
        df.to_csv(buffer, index=False)
        buffer.seek(0)
        csv_bytes = io.BytesIO(buffer.getvalue().encode("utf-8"))
        csv_bytes.name = "input.csv"  # downstream code may inspect .name
    except Exception as e:
        return None, None, _json_error(f"Could not convert 'data' to CSV: {e}")

    params = {
        "targets": body.get("targets", []),
        "methods": body.get("methods", {}),
        "handle_long_sequences": body.get("handleLongSequences", "truncate"),
        "use_experimental": coerce_bool_param(
            body.get("useExperimental"),
            default=False,
        ),
        "include_similarity_columns": coerce_bool_param(
            body.get("includeSimilarityColumns"),
            default=True,
        ),
        "canonicalize_substrates": coerce_bool_param(
            body.get("canonicalizeSubstrates"),
            default=True,
        ),
        "disable_gpu_precompute": coerce_bool_param(
            body.get("disableGpuPrecompute"),
            default=False,
        ),
        # Undocumented (see multipart parser). Requested value only; the view
        # resolves authorization before dispatch.
        "recon_xkg": coerce_recon_xkg(body.get("recon_xkg")),
    }

    return csv_bytes, params, None


# ---------------------------------------------------------------------------
# GET /api/v1/status/<public_id>/
# ---------------------------------------------------------------------------


@require_api_key
def api_job_status(request, public_id):
    """
    Return the current status of a prediction job.

    Poll this endpoint until status is "Completed" or "Failed", then
    fetch your results from the resultUrl.

    Possible status values:
      Pending    — job is queued and waiting for a worker
      Processing — worker is running predictions
      Completed  — predictions are done; results are ready to download
      Failed     — something went wrong; see the 'error' field for details
    """
    try:
        job = Job.objects.get(public_id=public_id)
    except Job.DoesNotExist:
        return _json_error(f"No job found with id '{public_id}'.", status=404)

    now = timezone.now()
    elapsed = (
        int((job.completion_time - job.submission_time).total_seconds())
        if job.completion_time
        else int((now - job.submission_time).total_seconds())
    )

    queue_seconds: int | None
    compute_seconds: int | None
    if job.start_time:
        queue_seconds = int(max(0, (job.start_time - job.submission_time).total_seconds()))
        if job.completion_time:
            compute_seconds = int(max(0, (job.completion_time - job.start_time).total_seconds()))
        else:
            compute_seconds = int(max(0, (now - job.start_time).total_seconds()))
    else:
        queue_seconds = max(0, elapsed) if job.status == "Pending" else None
        compute_seconds = None

    from api.utils.job_utils import get_queue_position

    progress: dict[str, Any] = {
        "moleculesTotal": job.total_molecules,
        "moleculesProcessed": job.molecules_processed,
        "predictionsTotal": job.total_predictions,
        "predictionsMade": job.predictions_made,
        "invalidRows": job.invalid_rows,
    }
    data: dict[str, Any] = {
        "jobId": job.public_id,
        "status": job.status,
        "submittedAt": job.submission_time.isoformat(),
        "elapsedSeconds": max(0, elapsed),
        "queueSeconds": queue_seconds,
        "computeSeconds": compute_seconds,
        "queuePosition": get_queue_position(job),
        "progress": progress,
    }

    try:
        stage_summary = get_progress_summary(job)
    except Exception:
        stage_summary = {
            "stages": [],
            "active_stage_index": None,
            "completed_stage_count": 0,
            "total_stage_count": 0,
        }
    data["progressStages"] = stage_summary["stages"]
    data["activeStageIndex"] = stage_summary["active_stage_index"]
    data["completedStageCount"] = stage_summary["completed_stage_count"]
    data["totalStageCount"] = stage_summary["total_stage_count"]
    progress["stages"] = stage_summary["stages"]
    progress["activeStageIndex"] = stage_summary["active_stage_index"]
    progress["completedStageCount"] = stage_summary["completed_stage_count"]
    progress["totalStageCount"] = stage_summary["total_stage_count"]

    embedding_progress = get_embedding_progress(job.public_id)
    if not embedding_progress:
        try:
            embedding_progress = get_active_stage_embedding(job)
        except Exception:
            embedding_progress = None
    if embedding_progress:
        progress["embedding"] = {
            "enabled": bool(embedding_progress.get("enabled", True)),
            "state": embedding_progress.get("state", "running"),
            "methodKey": embedding_progress.get("methodKey")
            or embedding_progress.get("method_key"),
            "method_key": embedding_progress.get("method_key")
            or embedding_progress.get("methodKey"),
            "target": embedding_progress.get("target"),
            "total": int(embedding_progress.get("total", 0) or 0),
            "cachedAlready": int(embedding_progress.get("cachedAlready", 0) or 0),
            "cached_already": int(embedding_progress.get("cached_already", 0) or 0),
            "needComputation": int(embedding_progress.get("needComputation", 0) or 0),
            "need_computation": int(embedding_progress.get("need_computation", 0) or 0),
            "computed": int(embedding_progress.get("computed", 0) or 0),
            "remaining": int(embedding_progress.get("remaining", 0) or 0),
            "updatedAt": embedding_progress.get("updatedAt"),
        }

    try:
        gpu_precompute = get_gpu_precompute_status(job.public_id)
    except Exception:
        gpu_precompute = None
    if gpu_precompute:
        data["gpuPrecompute"] = {
            "methodKey": gpu_precompute.get("methodKey")
            or gpu_precompute.get("method_key"),
            "method_key": gpu_precompute.get("method_key")
            or gpu_precompute.get("methodKey"),
            "target": gpu_precompute.get("target"),
            "attempted": bool(gpu_precompute.get("attempted", False)),
            "usedGpu": bool(
                gpu_precompute.get("usedGpu", gpu_precompute.get("used_gpu", False))
            ),
            "used_gpu": bool(
                gpu_precompute.get("used_gpu", gpu_precompute.get("usedGpu", False))
            ),
            "completed": bool(gpu_precompute.get("completed", False)),
            "failed": bool(gpu_precompute.get("failed", False)),
            "reason": gpu_precompute.get("reason"),
            "updatedAt": gpu_precompute.get("updatedAt"),
            "events": gpu_precompute.get("events", []),
        }

    if job.status == "Completed" and job.completion_time is not None:
        data["completedAt"] = job.completion_time.isoformat()
        data["resultUrl"] = f"/api/v1/result/{public_id}/"

    if job.status == "Failed":
        data["error"] = job.error_message or "An unknown error occurred."

    return JsonResponse(data)


# ---------------------------------------------------------------------------
# GET /api/v1/result/<public_id>/
# ---------------------------------------------------------------------------


@require_api_key
def api_download_result(request, public_id):
    """
    Download the prediction results for a completed job.

    By default, returns a CSV file attachment.  Add ?format=json to receive
    the same data as a JSON object instead — useful when you want to parse
    results directly without writing an intermediate file.

    The response includes all columns from the input CSV plus:
      - A prediction column (e.g. "kcat (1/s)" or "KM (mM)")
      - A "Source" column indicating whether the value came from a model
        or from the experimental database
      - An "Extra Info" column with additional details

    For ``Substrates`` inputs, Km and direct kcat/Km cells contain JSON-encoded
    arrays. They remain strings in CSV and in the JSON result envelope so both
    formats represent the same output exactly.

    Returns 409 Conflict if the job has not yet completed.
    Returns 404 if the job does not exist.
    Returns 500 if the output file is missing (should not normally occur).
    """
    try:
        job = Job.objects.get(public_id=public_id)
    except Job.DoesNotExist:
        return _json_error(f"No job found with id '{public_id}'.", status=404)

    if job.status != "Completed":
        return _json_error(
            f"Results are not yet available — job status is '{job.status}'. "
            "Poll /status/ until the job is Completed.",
            status=409,
        )

    if not job.output_file:
        return _json_error(
            "The output file for this job is missing. Please contact support.",
            status=500,
        )

    output_path = job.output_file.path

    # --- JSON format ----------------------------------------------------------
    if request.GET.get("format") == "json":
        try:
            result = serialize_result_csv(output_path)
        except Exception as e:
            return _json_error(f"Could not read output file: {e}", status=500)

        return JsonResponse({"jobId": public_id, **result})

    # --- Default: CSV download ------------------------------------------------
    try:
        file_handle = open(output_path, "rb")
    except OSError as e:
        return _json_error(f"Could not open output file: {e}", status=500)

    filename = f"webkinpred-{slugify(public_id)}-results.csv"
    return FileResponse(
        file_handle,
        as_attachment=True,
        filename=filename,
        content_type="text/csv",
    )


# ---------------------------------------------------------------------------
# POST /api/v1/validate/
# ---------------------------------------------------------------------------


@csrf_exempt
@require_api_key
def api_validate(request):
    """
    Validate a CSV file (or inline JSON data) without submitting a prediction
    job.  Checks substrate SMILES/InChI strings, protein amino-acid sequences,
    and sequence-length limits for every available model.

    Optionally runs MMseqs2 sequence similarity analysis against each method's
    training database when runSimilarity=true.  This is a synchronous call that
    blocks until the analysis is complete — it may take several minutes for
    large inputs.  No quota is consumed by this endpoint.

    Accepts the same two content-type formats as /submit/:

    1. multipart/form-data
         file           (required) — the CSV file
         runSimilarity  (optional, default "false") — "true" to include similarity

    2. application/json
         {
           "data":          [...],      // required — array of row objects
           "runSimilarity": false       // optional
         }

    Response fields:
      rowCount          — total number of data rows in the input
      invalidSubstrates — list of rows with unparseable SMILES/InChI strings
      invalidProteins   — list of rows with invalid amino-acid sequences
      lengthViolations  — per-model violation counts (all registered methods + Server)
      lengthLimits      — per-model max sequence lengths from descriptors
                          (null means no limit), plus Server
      similarity        — null when not requested; otherwise a dict keyed by
                          method name, each containing histogram_max,
                          histogram_mean, average_max_similarity,
                          average_mean_similarity, count_max, count_mean
    """
    if request.method != "POST":
        return _json_error("This endpoint only accepts POST requests.", 405)

    content_type = request.content_type or ""

    if "application/json" in content_type:
        csv_file, run_similarity, error = _parse_validate_json_body(request)
    else:
        csv_file, run_similarity, error = _parse_validate_multipart_body(request)

    if error:
        return error

    # Count rows (fast peek — parse once, then reset file pointer)
    try:
        csv_file.seek(0)
        row_count = len(pd.read_csv(csv_file))
        csv_file.seek(0)
    except Exception as e:
        return _json_error(f"Could not parse CSV: {e}")

    # Run substrate + protein + length validation
    validation_result = validate_input_file(csv_file)

    if "error" in validation_result:
        return _json_error(
            validation_result["error"],
            status=validation_result.get("status_code", 400),
        )

    # Optionally run MMseqs2 sequence similarity analysis
    similarity = None
    if run_similarity:
        try:
            csv_file.seek(0)
            similarity = analyze_sequence_similarity(csv_file, session_id="api_validate")
        except Exception as e:
            # Surface the error as a structured field rather than a 500 —
            # the validation results are still useful even if similarity fails.
            similarity = {"error": str(e)}

    return JsonResponse(
        {
            "rowCount": row_count,
            "invalidSubstrates": validation_result["invalid_substrates"],
            "invalidProteins": validation_result["invalid_proteins"],
            "lengthViolations": validation_result["length_violations"],
            "lengthLimits": validation_result["length_limits"],
            "similarity": similarity,
        }
    )


def _parse_validate_multipart_body(request):
    """
    Extract CSV file and runSimilarity flag from a multipart/form-data request.

    Returns (csv_file, run_similarity, error).  On success, error is None.
    """
    csv_file = request.FILES.get("file")

    if not csv_file:
        return None, False, _json_error("No file provided. Include 'file' as a multipart field.")

    if not csv_file.name.lower().endswith(".csv"):
        return None, False, _json_error("The uploaded file must have a .csv extension.")

    run_similarity = request.POST.get("runSimilarity", "false").lower() == "true"
    return csv_file, run_similarity, None


def _parse_validate_json_body(request):
    """
    Extract inline data and runSimilarity flag from an application/json request.

    Returns (csv_file, run_similarity, error).  On success, error is None.
    """
    try:
        body = json.loads(request.body)
    except (json.JSONDecodeError, UnicodeDecodeError):
        return None, False, _json_error("Request body is not valid JSON.")

    rows = body.get("data")

    if not rows or not isinstance(rows, list):
        return (
            None,
            False,
            _json_error(
                "'data' must be a non-empty array of row objects. "
                "Each object should map column names to values, e.g. "
                '{"Protein Sequence": "MKTL...", "Substrate": "CC(=O)O"}.'
            ),
        )

    if len(rows) > 10_000:
        return (
            None,
            False,
            _json_error(
                f"JSON body submission is limited to 10,000 rows. "
                f"You submitted {len(rows):,}.  Use CSV file upload for larger datasets."
            ),
        )

    try:
        df = pd.DataFrame(rows)
        buffer = io.StringIO()
        df.to_csv(buffer, index=False)
        buffer.seek(0)
        csv_bytes = io.BytesIO(buffer.getvalue().encode("utf-8"))
        csv_bytes.name = "input.csv"
    except Exception as e:
        return None, False, _json_error(f"Could not convert 'data' to CSV: {e}")

    run_similarity = bool(body.get("runSimilarity", False))
    return csv_bytes, run_similarity, None
