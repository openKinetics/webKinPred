"""
Job-specific utility functions for job submission and management.
"""

import json
import os
import pandas as pd
from typing import Dict, List, Optional, Any
from django.conf import settings
from django.utils import timezone
from api.utils import get_experimental
from api.services.embedding_progress_service import get_embedding_progress
from api.services.job_progress_service import get_active_stage_embedding, get_progress_summary
from api.utils.sequence_expansion import split_sequence_list
from api.utils.substrate_expansion import split_substrate_list

TARGET_ORDER = ["kcat", "Km", "kcat/Km"]
VALID_TARGETS = set(TARGET_ORDER)


def coerce_bool_param(value: Any, default: bool = False) -> bool:
    """
    Coerce common HTML/JSON boolean representations to a Python bool.
    """
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "on"}
    if isinstance(value, (int, float)):
        return bool(value)
    return default


def canonicalise_targets(targets: List[str]) -> List[str]:
    """
    Return deduplicated targets in canonical display/execution order.
    """
    out: List[str] = []
    for target in TARGET_ORDER:
        if target in targets and target not in out:
            out.append(target)
    return out


def canonical_prediction_type(targets: List[str]) -> str:
    """
    Build a compact, human-readable prediction_type label for persisted jobs.
    """
    ordered = canonicalise_targets(targets)
    return "+".join(ordered)


def validate_prediction_parameters(
    targets: List[str],
    methods: Dict[str, str],
) -> Optional[str]:
    """
    Validate target and method parameters against the method registry.

    Returns an error message if validation fails, None if valid.
    """
    from api.methods.registry import all_methods

    if not isinstance(targets, list) or not targets:
        return (
            'Invalid targets. Expected a non-empty list with values from "kcat", "Km", "kcat/Km".'
        )

    if not isinstance(methods, dict):
        return "Invalid methods payload. Expected an object mapping target names to method keys."

    invalid_targets = [t for t in targets if t not in VALID_TARGETS]
    if invalid_targets:
        return (
            "Invalid target(s): "
            + ", ".join(map(str, invalid_targets))
            + '. Allowed targets: "kcat", "Km", "kcat/Km".'
        )

    extra_method_keys = [k for k in methods.keys() if k not in VALID_TARGETS]
    if extra_method_keys:
        return (
            "Invalid method mapping keys: "
            + ", ".join(map(str, extra_method_keys))
            + '. Allowed keys: "kcat", "Km", "kcat/Km".'
        )

    registry = all_methods()
    for target in canonicalise_targets(targets):
        method_key = methods.get(target)
        if not isinstance(method_key, str) or not method_key.strip():
            return f"Missing method selection for target '{target}'."

        desc = registry.get(method_key)
        if desc is None or target not in desc.supports:
            valid = sorted(k for k, d in registry.items() if target in d.supports)
            return (
                f"Invalid method '{method_key}' for target '{target}'. "
                f"Available {target} methods: {', '.join(valid)}."
            )

    return None


def validate_sequence_handling_option(handle_long_seq: str) -> Optional[str]:
    """
    Validate the sequence handling option parameter.

    Returns an error message if invalid, None if valid.
    """
    if handle_long_seq not in ("truncate", "skip"):
        return 'Invalid handleLongSeq value. Expected "truncate" or "skip".'
    return None


def determine_required_columns(
    targets: List[str],
    methods: Dict[str, str],
) -> List[str]:
    """
    Determine strict required columns for the selected target/method set.

    The result always includes "Protein Sequence".  Additional columns are
    derived from each selected descriptor's col_to_kwarg mapping.
    """
    from api.methods.registry import get

    required: set[str] = {"Protein Sequence"}

    for target in canonicalise_targets(targets):
        method_key = methods.get(target)
        if not method_key:
            continue
        try:
            desc = get(method_key)
            required.update(desc.col_to_kwarg.keys())
        except KeyError:
            pass

    return list(required)


def validate_required_columns_for_methods(
    dataframe: pd.DataFrame,
    targets: List[str],
    methods: Dict[str, str],
) -> Optional[str]:
    """
    Validate CSV columns for selected methods.

    Pair-based methods accept either ``Substrate`` or ``Substrates``. CatPred
    kcat accepts semicolon-separated ``Substrates`` natively, while TurNup
    requires both ``Substrates`` and ``Products``.
    """
    from api.methods.registry import get

    has_substrate = "Substrate" in dataframe.columns
    has_substrates = "Substrates" in dataframe.columns
    has_products = "Products" in dataframe.columns
    if has_substrate and has_substrates:
        return "Cannot have both 'Substrate' and 'Substrates' columns."
    if has_products and not has_substrates:
        return "'Products' requires a 'Substrates' column."

    missing: set[str] = set()
    if "Protein Sequence" not in dataframe.columns:
        missing.add("Protein Sequence")

    for target in canonicalise_targets(targets):
        method_key = methods.get(target)
        if not method_key:
            continue
        try:
            desc = get(method_key)
        except KeyError:
            continue

        behavior = desc.input_behavior(target)
        if behavior == "native_full_reaction":
            if not has_substrates:
                missing.add("Substrates")
            if not has_products:
                missing.add("Products")
        elif behavior == "native_multi":
            if not has_substrates:
                missing.add("Substrates")
        else:
            if not (has_substrate or has_substrates):
                missing.add("Substrate or Substrates")
            for col in desc.col_to_kwarg.keys():
                if col != "Substrate" and col not in dataframe.columns:
                    missing.add(col)

    if not missing:
        return None

    ordered = sorted(missing, key=lambda c: (c != "Protein Sequence", c))
    return f"Missing required columns: {', '.join(ordered)}"


def create_job_directory(public_id: str) -> str:
    """
    Create directory structure for a job.

    Returns the path to the created job directory.
    """
    job_dir = os.path.join(settings.MEDIA_ROOT, "jobs", str(public_id))
    os.makedirs(job_dir, exist_ok=True)
    return job_dir


def save_job_input_file(file, job_dir: str) -> str:
    """
    Save the input CSV file to the job directory.

    Returns the path to the saved file.
    """
    file_path = os.path.join(job_dir, "input.csv")
    file.seek(0)
    input_df = pd.read_csv(file)
    input_df.dropna(how="all", inplace=True)
    input_df.to_csv(file_path, index=False)
    return file_path


def get_experimental_results(
    use_experimental: bool,
    methods: Dict[str, str],
    targets: List[str],
    dataframe: pd.DataFrame,
) -> Optional[Dict[str, list[dict[str, Any]]]]:
    """
    Look up experimental kinetic values when the user has opted in.

    Experimental lookup is skipped for native full-reaction methods (TurNup).
    For pair-based methods, ``Substrates`` rows are expanded positionally and
    each protein/substrate pair is looked up independently.

    Returns a dict keyed by target ("kcat", "Km"), or None.
    """
    if not use_experimental:
        return None

    selected = set(targets)
    out: Dict[str, list[dict[str, Any]]] = {}

    from api.methods.registry import get

    for target, param_type in (("kcat", "kcat"), ("Km", "Km")):
        if target not in selected:
            continue
        method_key = methods.get(target)
        if not method_key:
            continue
        try:
            desc = get(method_key)
        except KeyError:
            continue
        # Native multi/full-reaction targets do not have a pair lookup key.
        if desc.input_behavior(target) != "expanded_pair":
            continue

        raw_sequences = dataframe["Protein Sequence"].tolist()
        if "Substrate" in dataframe.columns:
            lookup_sequences: list[str] = []
            lookup_substrates: list[Any] = []
            positions: list[tuple[int, int, int]] = []
            substrates = dataframe["Substrate"].tolist()
            for reaction_idx, raw_sequence in enumerate(raw_sequences):
                for sequence_idx, sequence in enumerate(split_sequence_list(raw_sequence)):
                    lookup_sequences.append(sequence)
                    lookup_substrates.append(substrates[reaction_idx])
                    positions.append((reaction_idx, sequence_idx, 0))
            results = get_experimental.lookup_experimental(
                lookup_sequences,
                lookup_substrates,
                param_type=param_type,
            )
            if len(results) != len(positions):
                raise ValueError(
                    f"Experimental lookup returned {len(results)} result(s) for "
                    f"{len(positions)} protein/substrate pair(s)."
                )
            enriched: list[dict[str, Any]] = []
            for result, (reaction_idx, sequence_idx, substrate_idx) in zip(
                results,
                positions,
                strict=True,
            ):
                item = dict(result)
                item["reaction_idx"] = reaction_idx
                item["sequence_idx"] = sequence_idx
                item["substrate_idx"] = substrate_idx
                enriched.append(item)
            out[target] = enriched
            continue
        if "Substrates" not in dataframe.columns:
            continue

        lookup_sequences: list[str] = []
        lookup_substrates: list[str] = []
        positions: list[tuple[int, int, int]] = []
        substrate_values = dataframe["Substrates"].tolist()
        for reaction_idx, raw_sequence in enumerate(raw_sequences):
            sequences = split_sequence_list(raw_sequence)
            substrates = split_substrate_list(substrate_values[reaction_idx])
            for sequence_idx, sequence in enumerate(sequences):
                for substrate_idx, substrate in enumerate(substrates):
                    lookup_sequences.append(sequence)
                    lookup_substrates.append(substrate)
                    positions.append((reaction_idx, sequence_idx, substrate_idx))
        results = get_experimental.lookup_experimental(
            lookup_sequences,
            lookup_substrates,
            param_type=param_type,
        )
        if len(results) != len(positions):
            raise ValueError(
                f"Experimental lookup returned {len(results)} result(s) for "
                f"{len(positions)} protein/substrate pair(s)."
            )
        enriched: list[dict[str, Any]] = []
        for result, (reaction_idx, sequence_idx, substrate_idx) in zip(
            results,
            positions,
            strict=True,
        ):
            item = dict(result)
            item["reaction_idx"] = reaction_idx
            item["sequence_idx"] = sequence_idx
            item["substrate_idx"] = substrate_idx
            enriched.append(item)
        out[target] = enriched

    return out or None


def extract_job_parameters_from_request(request) -> Dict[str, Any]:
    """
    Extract job parameters from an HTTP request.

    Returns a parameters dictionary used by process_job_submission_from_params.
    """
    parse_error = ""
    targets: Any = request.POST.get("targets", "[]")
    methods: Any = request.POST.get("methods", "{}")

    try:
        if isinstance(targets, str):
            targets = json.loads(targets) if targets.strip() else []
    except json.JSONDecodeError:
        parse_error = (
            'Invalid \'targets\' format. Expected a JSON array, for example: ["kcat", "Km"].'
        )
        targets = []

    try:
        if isinstance(methods, str):
            methods = json.loads(methods) if methods.strip() else {}
    except json.JSONDecodeError:
        parse_error = (
            "Invalid 'methods' format. Expected a JSON object, for example: "
            '{"kcat":"DLKcat","Km":"UniKP"}.'
        )
        methods = {}

    return {
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
        "targets": targets if isinstance(targets, list) else [],
        "methods": methods if isinstance(methods, dict) else {},
        "handle_long_sequences": request.POST.get("handleLongSequences", "truncate"),
        "_parse_error": parse_error,
    }


def create_rate_limit_headers(daily_limit: int, remaining: int, ttl: int) -> Dict[str, str]:
    """
    Create standard rate-limiting headers for HTTP responses.
    """
    return {
        "X-RateLimit-Limit": str(daily_limit),
        "X-RateLimit-Remaining": str(max(0, remaining)),
        "X-RateLimit-Reset": str(ttl),
    }


def get_queue_position(job) -> Optional[int]:
    """
    Return the 1-based queue position for a Pending job.
    Returns None if the job is not Pending.
    """
    from api.models import Job as JobModel

    if job.status != "Pending":
        return None
    ahead = JobModel.objects.filter(
        status="Pending",
        submission_time__lt=job.submission_time,
    ).count()
    return ahead + 1


def create_job_status_response_data(job) -> Dict[str, Any]:
    """
    Create a response data dictionary for the job-status endpoint.
    """
    now = timezone.now()

    if job.completion_time:
        elapsed_seconds = int(max(0, (job.completion_time - job.submission_time).total_seconds()))
    else:
        elapsed_seconds = int(max(0, (now - job.submission_time).total_seconds()))

    queue_seconds: int | None
    compute_seconds: int | None
    if job.start_time:
        queue_seconds = int(max(0, (job.start_time - job.submission_time).total_seconds()))
        if job.completion_time:
            compute_seconds = int(max(0, (job.completion_time - job.start_time).total_seconds()))
        else:
            compute_seconds = int(max(0, (now - job.start_time).total_seconds()))
    else:
        queue_seconds = elapsed_seconds if job.status == "Pending" else None
        compute_seconds = None

    data = {
        "public_id": job.public_id,
        "status": job.status,
        "prediction_type": job.prediction_type,
        "kcat_method": job.kcat_method,
        "km_method": job.km_method,
        "kcat_km_method": job.kcat_km_method,
        "submission_time": job.submission_time,
        "completion_time": job.completion_time,
        "server_time": now,
        "elapsed_seconds": elapsed_seconds,
        "queue_seconds": queue_seconds,
        "compute_seconds": compute_seconds,
        "queue_position": get_queue_position(job),
        "error_message": job.error_message,
        "total_molecules": job.total_molecules,
        "molecules_processed": job.molecules_processed,
        "invalid_rows": job.invalid_rows,
        "total_predictions": job.total_predictions,
        "predictions_made": job.predictions_made,
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
    data["progress_stages"] = stage_summary["stages"]
    data["active_stage_index"] = stage_summary["active_stage_index"]
    data["completed_stage_count"] = stage_summary["completed_stage_count"]
    data["total_stage_count"] = stage_summary["total_stage_count"]

    embedding_progress = get_embedding_progress(job.public_id)
    if not embedding_progress:
        try:
            embedding_progress = get_active_stage_embedding(job)
        except Exception:
            embedding_progress = None
    if embedding_progress:
        data["embedding_progress"] = embedding_progress
    return data
