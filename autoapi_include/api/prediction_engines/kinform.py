# api/prediction_engines/kinform.py
#
# Prediction engine for the KinForm model (H and L variants).
#
# Wraps the KinForm prediction script in a subprocess call.  KinForm uses
# multiple protein embedding models (ESM2, ESMC, ProtT5, Pseq2Sites) that
# are each loaded once per job via dedicated conda environments.

import json
import logging
import os
import subprocess
from collections import defaultdict, deque

import pandas as pd

_log = logging.getLogger(__name__)

from api.methods.base import PredictionError
from api.models import Job
from api.prediction_engines.subprocess_runner import run_prediction_subprocess
from api.prediction_engines.runtime_paths import (
    DATA_PATHS,
    PREDICTION_SCRIPTS,
    PYTHON_PATHS,
)
from api.services.gpu_embed_service import run_gpu_precompute_if_available
from api.services.job_progress_service import (
    increment_stage_validation,
    reset_stage_prediction_metrics,
    set_stage_prediction_total,
)
from api.utils.convert_to_mol import convert_to_mol, substrate_as_smiles
from webKinPred.settings import MEDIA_ROOT

_AMINO_ACIDS = set("ACDEFGHIKLMNPQRSTVWY")

# KinForm CLI task names (passed as --task argument)
_TASK_FLAG = {"KCAT": "kcat", "KM": "KM"}


def _is_missing_prediction(prediction) -> bool:
    """Return True when a raw model output should be treated as missing."""
    if prediction is None:
        return True
    if isinstance(prediction, str):
        return prediction.strip().lower() in {"", "none", "nan", "null"}
    try:
        return bool(pd.isna(prediction))
    except Exception:
        return False


def kinform_predictions(
    sequences: list[str],
    public_id: str,
    substrates: list[str],
    model_variant: str = "H",
    kinetics_type: str = "KCAT",
    canonicalize_substrates: bool = True,
    **kwargs,
) -> tuple[list, dict[int, str]]:
    """
    Run the KinForm model on the given protein sequences and substrates.

    Parameters
    ----------
    sequences : list[str]
        Pre-filtered protein sequences.
    public_id : str
        Job identifier for progress tracking.
    substrates : list[str]
        Substrate SMILES or InChI strings, one per sequence.
    model_variant : str
        ``"H"`` (high similarity) or ``"L"`` (low similarity).
    kinetics_type : str
        ``"KCAT"`` (default) or ``"KM"``.

    Returns
    -------
    predictions : list
        Predicted values (float) or None for invalid rows.
    invalid_reasons : dict[int, str]
        Maps row index to a human-readable reason for rows that could not
        be processed.

    Raises
    ------
    PredictionError
        On subprocess failure or any unrecoverable error.
    """
    assert model_variant in {"H", "L"}, "model_variant must be 'H' or 'L'"
    model_key = f"KinForm-{model_variant}"
    stage_target = "kcat" if kinetics_type.upper() == "KCAT" else "Km"
    _log.info(
        "Prediction method started",
        extra={
            "event": "prediction.method_started",
            "job_public_id": public_id,
            "method_key": model_key,
            "target": stage_target,
            "kinetics_type": kinetics_type,
        },
    )

    job = Job.objects.get(public_id=public_id)
    reset_stage_prediction_metrics(
        job_public_id=public_id,
        target=stage_target,
        method_key=model_key,
        total_rows=len(sequences),
    )

    python_path = PYTHON_PATHS.get("KinForm", "")
    prediction_script = PREDICTION_SCRIPTS.get("KinForm", "")
    job_dir = os.path.join(MEDIA_ROOT, "jobs", str(public_id))
    input_file = os.path.join(job_dir, f"input_{public_id}.csv")
    output_file = os.path.join(job_dir, f"output_{public_id}.csv")

    # Pass embedding environment python paths so KinForm can invoke each
    # embedding model once per job without reloading.
    env = os.environ.copy()
    if DATA_PATHS.get("media"):
        env["KINFORM_MEDIA_PATH"] = DATA_PATHS["media"]
    if DATA_PATHS.get("tools"):
        env["KINFORM_TOOLS_PATH"] = DATA_PATHS["tools"]
    if DATA_PATHS.get("KinForm"):
        env["KINFORM_DATA"] = DATA_PATHS["KinForm"]
    if PYTHON_PATHS.get("esm2"):
        env["KINFORM_ESM_PATH"] = PYTHON_PATHS["esm2"]
    if PYTHON_PATHS.get("esmc"):
        env["KINFORM_ESMC_PATH"] = PYTHON_PATHS["esmc"]
    if PYTHON_PATHS.get("t5"):
        env["KINFORM_T5_PATH"] = PYTHON_PATHS["t5"]
    if PYTHON_PATHS.get("pseq2sites"):
        env["KINFORM_PSEQ2SITES_PATH"] = PYTHON_PATHS["pseq2sites"]

    valid_indices: list[int] = []
    invalid_reasons: dict[int, str] = {}
    valid_smiles: list[str] = []
    valid_sequences: list[str] = []
    predictions: list = [None] * len(sequences)
    disable_gpu_precompute = bool(kwargs.get("disable_gpu_precompute", False))

    # ── Validate inputs molecule by molecule ──────────────────────────────────
    for idx, (seq, substrate) in enumerate(zip(sequences, substrates)):
        seq_valid = all(c in _AMINO_ACIDS for c in seq)
        mol = convert_to_mol(substrate)

        if mol and seq_valid:
            valid_smiles.append(
                substrate_as_smiles(
                    substrate,
                    canonicalize=canonicalize_substrates,
                    preserve_raw_smiles_when_possible=True,
                )
            )
            valid_sequences.append(seq)
            valid_indices.append(idx)
        else:
            reason = (
                "Invalid protein sequence (unsupported amino acid characters)"
                if not seq_valid
                else "Invalid substrate (not a valid SMILES or InChI)"
            )
            _log.debug(
                "Prediction row invalid",
                extra={
                    "event": "prediction.row_invalid",
                    "job_public_id": public_id,
                    "method_key": model_key,
                    "target": stage_target,
                    "row_index": idx,
                    "reason": reason,
                },
            )
            invalid_reasons[idx] = reason
            increment_stage_validation(
                job_public_id=public_id,
                target=stage_target,
                method_key=model_key,
                processed_inc=1,
                invalid_inc=1,
            )
            continue

        increment_stage_validation(
            job_public_id=public_id,
            target=stage_target,
            method_key=model_key,
            processed_inc=1,
            invalid_inc=0,
        )

    set_stage_prediction_total(
        job_public_id=public_id,
        target=stage_target,
        method_key=model_key,
        total_predictions=len(valid_indices),
    )

    if not valid_indices:
        return predictions, invalid_reasons

    _gpu = run_gpu_precompute_if_available(
        job_public_id=public_id,
        method_key=model_key,
        target=stage_target,
        valid_sequences=valid_sequences,
        env=env,
        disabled=disable_gpu_precompute,
    )
    if _gpu.attempted and not _gpu.completed:
        _log.warning(
            "GPU precompute incomplete for %s job %s: %s (used_gpu=%s, failed=%s)",
            model_key, public_id, _gpu.reason, _gpu.used_gpu, _gpu.failed,
        )

    # ── Write JSON input file (KinForm expects JSON, not CSV) ─────────────────
    try:
        json_input = [
            {"smiles": smiles, "sequence": seq, "Sequence": seq}
            for smiles, seq in zip(valid_smiles, valid_sequences)
        ]
        with open(input_file, "w") as f:
            json.dump(json_input, f)
    except OSError as e:
        raise PredictionError(
            f"{model_key} could not write its input file. Please contact support if this persists."
        ) from e

    # ── Run prediction subprocess ─────────────────────────────────────────────
    try:
        run_prediction_subprocess(
            command=[
                python_path,
                prediction_script,
                "--mode",
                "predict",
                "--task",
                _TASK_FLAG[kinetics_type],
                "--model_config",
                f"KinForm-{model_variant}",
                "--save_results",
                output_file,
                "--data_path",
                input_file,
            ],
            job=job,
            env=env,
            label=model_key,
            method_key=model_key,
            target=stage_target,
            valid_sequences=valid_sequences,
        )
    except subprocess.CalledProcessError as e:
        _cleanup(input_file, output_file)
        if e.returncode in (-9, 137):
            raise PredictionError(
                f"{model_key} ran out of memory. "
                "Try reducing the number of rows or the sequence lengths."
            ) from e
        raise PredictionError(
            f"{model_key} encountered an internal error and could not complete. "
            "Please verify your input and try again."
        ) from e
    except Exception as e:
        _cleanup(input_file, output_file)
        if isinstance(e, PredictionError):
            raise
        raise PredictionError(
            f"{model_key} encountered an unexpected error. Please verify your input and try again."
        ) from e

    # ── Read output CSV ───────────────────────────────────────────────────────
    try:
        df_output = pd.read_csv(output_file)
        if "y_pred" not in df_output.columns:
            raise ValueError("missing required output column 'y_pred'")
    except Exception as e:
        _cleanup(input_file, output_file)
        raise PredictionError(
            f"{model_key} completed but its output file could not be read. "
            "Please contact support if this persists."
        ) from e

    expected = len(valid_indices)
    reported = len(df_output)
    post_prediction_invalids = 0
    matched_local: set[int] = set()

    # Prefer key-based mapping to avoid positional drift if KinForm internally
    # drops rows (e.g., stricter filtering than the wrapper).
    can_key_map = {"sequence", "smiles"}.issubset(set(df_output.columns))
    if can_key_map:
        key_to_local: dict[tuple[str, str], deque[int]] = defaultdict(deque)
        for local_idx, (seq, smi) in enumerate(zip(valid_sequences, valid_smiles)):
            key_to_local[(str(seq), str(smi))].append(local_idx)

        unmatched_rows = 0
        for out_row in df_output.itertuples(index=False):
            seq_out = str(getattr(out_row, "sequence", ""))
            smi_out = str(getattr(out_row, "smiles", ""))
            pred = getattr(out_row, "y_pred", None)
            queue = key_to_local.get((seq_out, smi_out))
            if not queue:
                unmatched_rows += 1
                continue

            local_idx = queue.popleft()
            matched_local.add(local_idx)
            global_idx = valid_indices[local_idx]
            if _is_missing_prediction(pred):
                predictions[global_idx] = None
                invalid_reasons[global_idx] = "Prediction could not be made"
                post_prediction_invalids += 1
            else:
                predictions[global_idx] = pred

        if unmatched_rows:
            _log.warning(
                "%s output contained %d unmatched rows for job %s (reported=%d)",
                model_key,
                unmatched_rows,
                public_id,
                reported,
            )
    else:
        raw_values = df_output["y_pred"].tolist()
        upto = min(expected, reported)
        for local_idx in range(upto):
            pred = raw_values[local_idx]
            matched_local.add(local_idx)
            global_idx = valid_indices[local_idx]
            if _is_missing_prediction(pred):
                predictions[global_idx] = None
                invalid_reasons[global_idx] = "Prediction could not be made"
                post_prediction_invalids += 1
            else:
                predictions[global_idx] = pred

    missing_local = [i for i in range(expected) if i not in matched_local]
    if missing_local:
        _log.warning(
            "%s output shorter than expected for job %s: expected=%d, reported=%d, unmatched_inputs=%d",
            model_key,
            public_id,
            expected,
            reported,
            len(missing_local),
        )
        for local_idx in missing_local:
            global_idx = valid_indices[local_idx]
            predictions[global_idx] = None
            invalid_reasons[global_idx] = "Prediction output missing"
            post_prediction_invalids += 1

    if reported > expected:
        _log.warning(
            "%s output longer than expected for job %s: expected=%d, reported=%d",
            model_key,
            public_id,
            expected,
            reported,
        )

    if post_prediction_invalids:
        increment_stage_validation(
            job_public_id=public_id,
            target=stage_target,
            method_key=model_key,
            processed_inc=0,
            invalid_inc=post_prediction_invalids,
        )

    _cleanup(input_file, output_file)
    return predictions, invalid_reasons


def _cleanup(*paths: str) -> None:
    for path in paths:
        try:
            if os.path.exists(path):
                os.remove(path)
        except OSError:
            pass
