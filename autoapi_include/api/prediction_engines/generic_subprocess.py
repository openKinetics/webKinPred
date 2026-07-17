"""
Generic subprocess prediction engine.

This engine is intended for most new methods: the contributor only writes a
prediction script and method descriptor. No custom Python engine module is
required unless the method needs bespoke behavior.
"""

from __future__ import annotations

import json
import math
import os
import re
import logging
import subprocess
from typing import Any

from api.methods.base import MethodDescriptor, PredictionError
from api.models import Job
from api.prediction_engines.runtime_paths import (
    DATA_PATHS,
    PREDICTION_SCRIPTS,
    PYTHON_PATHS,
)
from api.services.embedding_plan_service import (
    resolve_media_and_tools,
    resolve_seq_ids_via_cli,
)
from api.services.gpu_embed_service import run_gpu_precompute_if_available
from api.services.job_progress_service import (
    increment_stage_validation,
    reset_stage_prediction_metrics,
    set_stage_prediction_total,
)

from api.prediction_engines.subprocess_runner import run_prediction_subprocess
from api.utils.convert_to_mol import convert_to_mol
from webKinPred.settings import MEDIA_ROOT
_log = logging.getLogger(__name__)
_SEQ_ID_ATTACH_METHOD_KEYS = {"OmniESI", "OmniESI-O2DENet", "RealKcat", "IECata"}


def run_generic_subprocess_prediction(
    desc: MethodDescriptor,
    sequences: list[str],
    public_id: str,
    target: str,
    disable_gpu_precompute: bool = False,
    **kwargs,
) -> tuple[list, dict[int, str]]:
    """
    Execute a method via the built-in generic subprocess engine.

    Expected prediction-script contract
    -----------------------------------
    Command:
      python <script> <extra_args...> --input <input.json> --output <output.json>

    Input JSON:
      {
        "method": "<method key>",
        "target": "kcat" | "Km",
        "public_id": "<job id>",
        "rows": [
          {"sequence": "...", "...": "..."},
          ...
        ],
        "params": {"...": "..."}
      }

    Output JSON:
      {"predictions": [...], "invalid_indices": [...], "invalid_reasons": {...}}   OR just [...]
    """
    cfg = desc.subprocess
    if cfg is None:
        raise PredictionError(f"{desc.display_name} is not configured with a subprocess engine.")

    job = Job.objects.get(public_id=public_id)
    _initialise_job_progress(job, len(sequences), method_key=desc.key, target=target)

    row_kwarg_names = list(dict.fromkeys(desc.col_to_kwarg.values()))
    per_row_inputs = _extract_row_inputs(
        method_label=desc.display_name,
        row_kwarg_names=row_kwarg_names,
        sequences=sequences,
        call_kwargs=kwargs,
    )
    static_params = {key: value for key, value in kwargs.items() if key not in row_kwarg_names}

    predictions: list[Any] = [None] * len(sequences)
    valid_rows, valid_indices, invalid_reasons = _validate_rows(
        sequences=sequences,
        per_row_inputs=per_row_inputs,
        input_format=desc.input_format,
        desc=desc,
        job=job,
        method_key=desc.key,
        target=target,
    )

    set_stage_prediction_total(
        job_public_id=public_id,
        target=target,
        method_key=desc.key,
        total_predictions=len(valid_indices),
    )

    if not valid_indices:
        return predictions, invalid_reasons

    python_path, script_path = _resolve_subprocess_paths(desc)
    env = _build_subprocess_env(desc)
    valid_sequences = [str(row.get("sequence", "")) for row in valid_rows]
    embedding_sequences = valid_sequences if desc.embeddings_used else None

    if _should_attach_seq_ids(desc):
        _attach_seq_ids_to_rows(desc=desc, rows=valid_rows, sequences=valid_sequences, env=env)

    if embedding_sequences:
        _gpu = run_gpu_precompute_if_available(
            job_public_id=public_id,
            method_key=desc.key,
            target=target,
            valid_sequences=embedding_sequences,
            env=env,
            disabled=disable_gpu_precompute,
        )
        if _gpu.attempted and not _gpu.completed:
            _log.warning(
                "GPU precompute incomplete for %s job %s: %s (used_gpu=%s, failed=%s)",
                desc.key, public_id, _gpu.reason, _gpu.used_gpu, _gpu.failed,
            )
        if _gpu.failed and cfg.fail_on_gpu_precompute_failure:
            reason = str(_gpu.reason or "").strip().lower()
            if "timeout" in reason or "timed out" in reason:
                raise PredictionError(
                    f"{desc.display_name} could not complete because GPU precompute timed out. "
                    "Please try again later."
                )
            raise PredictionError(
                f"{desc.display_name} could not complete because GPU precompute failed. "
                "Please try again later."
            )

    job_dir = os.path.join(MEDIA_ROOT, "jobs", str(public_id))
    safe_method = re.sub(r"[^A-Za-z0-9_-]+", "_", desc.key)
    input_file = os.path.join(job_dir, f"{safe_method}_input_{public_id}.json")
    output_file = os.path.join(job_dir, f"{safe_method}_output_{public_id}.json")

    payload = {
        "method": desc.key,
        "target": target,
        "public_id": public_id,
        "rows": valid_rows,
        "params": static_params,
    }

    try:
        with open(input_file, "w", encoding="utf-8") as f:
            json.dump(payload, f)
    except OSError as e:
        raise PredictionError(
            f"{desc.display_name} could not write its input file. "
            "Please contact support if this persists."
        ) from e

    command = [
        python_path,
        script_path,
        *cfg.extra_args,
        cfg.input_flag,
        input_file,
        cfg.output_flag,
        output_file,
    ]

    try:
        run_prediction_subprocess(
            command=command,
            job=job,
            env=env,
            label=desc.display_name,
            method_key=desc.key,
            target=target,
            valid_sequences=embedding_sequences,
        )
    except subprocess.CalledProcessError as e:
        _cleanup(input_file, output_file)
        if e.returncode in (-9, 137):
            raise PredictionError(
                f"{desc.display_name} ran out of memory. "
                "Try reducing the number of rows or the sequence lengths."
            ) from e
        raise PredictionError(
            f"{desc.display_name} encountered an internal error and could not complete. "
            "Please verify your input and try again."
        ) from e
    except Exception as e:
        _cleanup(input_file, output_file)
        if isinstance(e, PredictionError):
            raise
        raise PredictionError(
            f"{desc.display_name} encountered an unexpected error. "
            "Please verify your input and try again."
        ) from e

    try:
        pred_subset, invalid_subset = _read_output(desc.display_name, output_file)
    except PredictionError:
        _cleanup(input_file, output_file)
        raise

    if len(pred_subset) != len(valid_rows):
        _cleanup(input_file, output_file)
        raise PredictionError(
            f"{desc.display_name} produced {len(pred_subset)} prediction(s) "
            f"for {len(valid_rows)} valid input row(s)."
        )

    for local_idx, value in enumerate(pred_subset):
        global_idx = valid_indices[local_idx]
        predictions[global_idx] = _normalise_prediction(value)

    # Merge method-reported invalids (indices into valid_rows) into the reason dict
    for local_idx, reason in invalid_subset.items():
        if 0 <= local_idx < len(valid_indices):
            seq_idx = valid_indices[local_idx]
            invalid_reasons.setdefault(seq_idx, reason)

    _cleanup(input_file, output_file)
    return predictions, invalid_reasons


def _initialise_job_progress(job: Job, total_rows: int, method_key: str, target: str) -> None:
    reset_stage_prediction_metrics(
        job_public_id=job.public_id,
        target=target,
        method_key=method_key,
        total_rows=total_rows,
    )


def _extract_row_inputs(
    method_label: str,
    row_kwarg_names: list[str],
    sequences: list[str],
    call_kwargs: dict[str, Any],
) -> dict[str, list[Any]]:
    n_rows = len(sequences)
    out: dict[str, list[Any]] = {}

    for key in row_kwarg_names:
        values = call_kwargs.get(key)
        if not isinstance(values, list):
            raise PredictionError(f"{method_label} input mapping is invalid for '{key}'.")
        if len(values) != n_rows:
            raise PredictionError(f"{method_label} input mapping length mismatch for '{key}'.")
        out[key] = values

    return out


def _validate_rows(
    sequences: list[str],
    per_row_inputs: dict[str, list[Any]],
    input_format: str,
    desc: MethodDescriptor,
    job: Job,
    method_key: str,
    target: str,
) -> tuple[list[dict[str, Any]], list[int], dict[int, str]]:
    cfg = desc.subprocess
    assert cfg is not None

    valid_rows: list[dict[str, Any]] = []
    valid_indices: list[int] = []
    invalid_reasons: dict[int, str] = {}

    allowed = set(cfg.allowed_amino_acids)

    for idx, seq in enumerate(sequences):
        row = {"sequence": seq}
        for key, values in per_row_inputs.items():
            row[key] = values[idx]

        is_valid = True
        reason = ""

        if cfg.validate_sequence:
            if not isinstance(seq, str) or not seq or any(c not in allowed for c in seq):
                is_valid = False
                reason = "Invalid protein sequence (unsupported amino acid characters)"

        if is_valid and cfg.validate_chemistry:
            if not _chemistry_is_valid(row, input_format):
                is_valid = False
                reason = "Invalid substrate (not a valid SMILES or InChI)"

        if is_valid:
            valid_indices.append(idx)
            valid_rows.append(row)
            increment_stage_validation(
                job_public_id=job.public_id,
                target=target,
                method_key=method_key,
                processed_inc=1,
                invalid_inc=0,
            )
        else:
            invalid_reasons[idx] = reason
            increment_stage_validation(
                job_public_id=job.public_id,
                target=target,
                method_key=method_key,
                processed_inc=1,
                invalid_inc=1,
            )

    return valid_rows, valid_indices, invalid_reasons


def _split_tokens(value: Any) -> list[str]:
    if value is None:
        return []

    if isinstance(value, (list, tuple)):
        out: list[str] = []
        for item in value:
            out.extend(_split_tokens(item))
        return out

    text = str(value).strip()
    if not text:
        return []
    if text.lower() in {"none", "nan"}:
        return []

    semicolon_tokens = [tok.strip() for tok in text.split(";") if tok.strip()]
    tokens_out: list[str] = []
    for token in semicolon_tokens:
        if token.startswith("InChI="):
            tokens_out.append(token)
            continue
        # Support multi-component entries (e.g. "A.B") in single substrate fields.
        dot_parts = [part.strip() for part in token.split(".") if part.strip()]
        tokens_out.extend(dot_parts if dot_parts else [token])
    return tokens_out


def _chemistry_is_valid(row: dict[str, Any], input_format: str) -> bool:
    chem_fields: list[tuple[str, Any]] = []
    for key in ("substrates", "substrate", "products", "product"):
        if key in row:
            chem_fields.append((key, row[key]))

    if input_format == "multi":
        has_substrates = any(k in row for k in ("substrates", "substrate"))
        has_products = any(k in row for k in ("products", "product"))
        if not (has_substrates and has_products):
            return False

    if not chem_fields:
        # Sequence-only methods are allowed.
        return True

    for _key, value in chem_fields:
        tokens = _split_tokens(value)
        if not tokens:
            return False
        for token in tokens:
            if convert_to_mol(token) is None:
                return False

    return True


def _resolve_subprocess_paths(desc: MethodDescriptor) -> tuple[str, str]:
    cfg = desc.subprocess
    assert cfg is not None

    python_path = cfg.python_path or (
        PYTHON_PATHS.get(cfg.python_path_key, "") if cfg.python_path_key else ""
    )
    script_path = cfg.script_path or (
        PREDICTION_SCRIPTS.get(cfg.script_key, "") if cfg.script_key else ""
    )

    if not python_path:
        raise PredictionError(
            f"{desc.display_name} is not configured correctly (missing python path)."
        )
    if not script_path:
        raise PredictionError(
            f"{desc.display_name} is not configured correctly (missing prediction script path)."
        )

    return python_path, script_path


def _build_subprocess_env(desc: MethodDescriptor) -> dict[str, str]:
    cfg = desc.subprocess
    assert cfg is not None

    env = os.environ.copy()

    for env_var, data_key in cfg.data_path_env.items():
        path = DATA_PATHS.get(data_key)
        if path:
            env[env_var] = path

    for env_var, value in cfg.extra_env.items():
        env[env_var] = str(value)

    return env


def _should_attach_seq_ids(desc: MethodDescriptor) -> bool:
    """Return True when a subprocess method reads shared embedding caches by seq_id."""
    return str(getattr(desc, "key", "")).strip() in _SEQ_ID_ATTACH_METHOD_KEYS


def _attach_seq_ids_to_rows(
    *,
    desc: MethodDescriptor,
    rows: list[dict[str, Any]],
    sequences: list[str],
    env: dict[str, str],
) -> None:
    """Attach shared seqmap IDs for subprocess scripts that consume them.

    Most generic subprocess adapters resolve sequence IDs internally. Methods
    that share GPU-precomputed cache families (OmniESI, RealKcat, IECata)
    intentionally consume the platform seq_id directly in subprocess payloads
    so local fallback and remote precompute use exactly the same cache key.
    """
    if not rows:
        return

    try:
        media_path, tools_path = resolve_media_and_tools(desc.key, env)
        seq_ids = resolve_seq_ids_via_cli(sequences, tools_path, media_path)
    except Exception as exc:
        _log.warning(
            "Could not attach seq_ids for %s; embedding cache will be skipped in subprocess: %s",
            desc.key,
            exc,
        )
        return

    if len(seq_ids) != len(rows):
        _log.warning(
            "Could not attach seq_ids for %s; got %d id(s) for %d row(s)",
            desc.key,
            len(seq_ids),
            len(rows),
        )
        return

    for row, seq_id in zip(rows, seq_ids):
        row["seq_id"] = seq_id


def _read_output(method_label: str, output_file: str) -> tuple[list[Any], dict[int, str]]:
    try:
        with open(output_file, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception as e:
        raise PredictionError(
            f"{method_label} completed but its output file could not be read. "
            "Please contact support if this persists."
        ) from e

    if isinstance(data, list):
        return data, {}

    if isinstance(data, dict):
        preds = data.get("predictions")
        invalid = data.get("invalid_indices", [])
        invalid_reasons = data.get("invalid_reasons", {})

        if not isinstance(preds, list):
            raise PredictionError(
                f"{method_label} output format is invalid: 'predictions' must be a list."
            )

        invalid_out: dict[int, str] = {}
        if isinstance(invalid, list):
            for idx in invalid:
                try:
                    invalid_out[int(idx)] = "Prediction could not be made"
                except (TypeError, ValueError):
                    continue
        if isinstance(invalid_reasons, dict):
            for idx, reason in invalid_reasons.items():
                try:
                    local_idx = int(idx)
                except (TypeError, ValueError):
                    continue
                reason_text = str(reason).strip()
                if reason_text:
                    invalid_out[local_idx] = reason_text
        return preds, invalid_out

    raise PredictionError(
        f"{method_label} output format is invalid. "
        "Expected a JSON list or an object with 'predictions'."
    )


def _normalise_prediction(value: Any) -> Any:
    if value is None:
        return None

    if isinstance(value, float) and math.isnan(value):
        return None

    if isinstance(value, str):
        trimmed = value.strip()
        if trimmed == "":
            return None
        if trimmed.lower() in {"none", "nan"}:
            return None
        return value

    return value


def _cleanup(*paths: str) -> None:
    for path in paths:
        try:
            if os.path.exists(path):
                os.remove(path)
        except OSError:
            pass
