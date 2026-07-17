# api/prediction_engines/unikp.py
#
# Prediction engine for the UniKP model.
#
# Wraps the UniKP prediction script in a subprocess call.  Handles molecule
# validation, progress reporting, and user-friendly error messages.

import os
import subprocess

import numpy as np
import pandas as pd

import logging

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

_log = logging.getLogger(__name__)
from api.utils.convert_to_mol import convert_to_mol, substrate_as_smiles
from webKinPred.settings import MEDIA_ROOT

_AMINO_ACIDS = set("ACDEFGHIKLMNPQRSTVWY")


def unikp_predictions(
    sequences: list[str],
    public_id: str,
    substrates: list[str],
    kinetics_type: str = "KCAT",
    canonicalize_substrates: bool = True,
    **kwargs,
) -> tuple[list, dict[int, str]]:
    """
    Run the UniKP model on the given protein sequences and substrates.

    Parameters
    ----------
    sequences : list[str]
        Pre-filtered protein sequences.
    public_id : str
        Job identifier for progress tracking.
    substrates : list[str]
        Substrate SMILES or InChI strings, one per sequence.
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
    disable_gpu_precompute = bool(kwargs.get("disable_gpu_precompute", False))
    stage_target = "kcat" if kinetics_type.upper() == "KCAT" else "Km"
    _log.info(
        "Prediction method started",
        extra={
            "event": "prediction.method_started",
            "job_public_id": public_id,
            "method_key": "UniKP",
            "target": stage_target,
            "kinetics_type": kinetics_type,
        },
    )

    job = Job.objects.get(public_id=public_id)
    reset_stage_prediction_metrics(
        job_public_id=public_id,
        target=stage_target,
        method_key="UniKP",
        total_rows=len(sequences),
    )

    python_path = PYTHON_PATHS.get("UniKP", "")
    prediction_script = PREDICTION_SCRIPTS.get("UniKP", "")
    job_dir = os.path.join(MEDIA_ROOT, "jobs", str(public_id))
    input_file = os.path.join(job_dir, f"input_{public_id}.csv")
    output_file = os.path.join(job_dir, f"output_{public_id}.csv")

    env = os.environ.copy()
    if DATA_PATHS.get("media"):
        env["UNIKP_MEDIA_PATH"] = DATA_PATHS["media"]
    if DATA_PATHS.get("tools"):
        env["UNIKP_TOOLS_PATH"] = DATA_PATHS["tools"]

    valid_indices: list[int] = []
    invalid_reasons: dict[int, str] = {}
    valid_smiles: list[str] = []
    valid_sequences: list[str] = []
    predictions: list = [None] * len(sequences)

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
                    "method_key": "UniKP",
                    "target": stage_target,
                    "row_index": idx,
                    "reason": reason,
                },
            )
            invalid_reasons[idx] = reason
            increment_stage_validation(
                job_public_id=public_id,
                target=stage_target,
                method_key="UniKP",
                processed_inc=1,
                invalid_inc=1,
            )
            continue

        increment_stage_validation(
            job_public_id=public_id,
            target=stage_target,
            method_key="UniKP",
            processed_inc=1,
            invalid_inc=0,
        )

    set_stage_prediction_total(
        job_public_id=public_id,
        target=stage_target,
        method_key="UniKP",
        total_predictions=len(valid_indices),
    )

    if not valid_indices:
        return predictions, invalid_reasons

    _gpu = run_gpu_precompute_if_available(
        job_public_id=public_id,
        method_key="UniKP",
        target=stage_target,
        valid_sequences=valid_sequences,
        env=env,
        disabled=disable_gpu_precompute,
    )
    if _gpu.attempted and not _gpu.completed:
        _log.warning(
            "GPU precompute incomplete for UniKP job %s: %s (used_gpu=%s, failed=%s)",
            public_id, _gpu.reason, _gpu.used_gpu, _gpu.failed,
        )

    # ── Write CSV input file ──────────────────────────────────────────────────
    try:
        df_input = pd.DataFrame(
            {
                "Substrate SMILES": valid_smiles,
                "Protein Sequence": valid_sequences,
            }
        )
        df_input.to_csv(input_file, index=False)
    except OSError as e:
        raise PredictionError(
            "UniKP could not write its input file. Please contact support if this persists."
        ) from e

    # ── Run prediction subprocess ─────────────────────────────────────────────
    try:
        run_prediction_subprocess(
            command=[python_path, prediction_script, input_file, output_file, kinetics_type],
            job=job,
            env=env,
            label="UniKP",
            method_key="UniKP",
            target=stage_target,
            valid_sequences=valid_sequences,
        )
    except subprocess.CalledProcessError as e:
        _cleanup(input_file, output_file)
        if e.returncode in (-9, 137):
            raise PredictionError(
                "UniKP ran out of memory. Try reducing the number of rows or the sequence lengths."
            ) from e
        raise PredictionError(
            "UniKP encountered an internal error and could not complete. "
            "Please verify your input and try again."
        ) from e
    except Exception as e:
        _cleanup(input_file, output_file)
        if isinstance(e, PredictionError):
            raise
        raise PredictionError(
            "UniKP encountered an unexpected error. Please verify your input and try again."
        ) from e

    # ── Read output CSV ───────────────────────────────────────────────────────
    try:
        df_output = pd.read_csv(output_file)
        raw_values = df_output["Predicted Value"].tolist()
    except Exception as e:
        _cleanup(input_file, output_file)
        raise PredictionError(
            "UniKP completed but its output file could not be read. "
            "Please contact support if this persists."
        ) from e

    for local_idx, pred in enumerate(raw_values):
        global_idx = valid_indices[local_idx]
        predictions[global_idx] = None if pred in ("None", "", np.nan, "nan") else pred

    _cleanup(input_file, output_file)
    return predictions, invalid_reasons


def _cleanup(*paths: str) -> None:
    for path in paths:
        try:
            if os.path.exists(path):
                os.remove(path)
        except OSError:
            pass
