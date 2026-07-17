# api/prediction_engines/dlkcat.py
#
# Prediction engine for the DLKcat model.
#
# Wraps the DLKcat prediction script in a subprocess call.  Handles molecule
# validation, progress reporting, and user-friendly error messages.

import logging
import os
import subprocess

import numpy as np
from rdkit import Chem

from api.methods.base import PredictionError
from api.models import Job
from api.prediction_engines.subprocess_runner import run_prediction_subprocess
from api.prediction_engines.runtime_paths import (
    DATA_PATHS,
    PREDICTION_SCRIPTS,
    PYTHON_PATHS,
)
from api.services.job_progress_service import (
    increment_stage_validation,
    reset_stage_prediction_metrics,
    set_stage_prediction_total,
)
from api.utils.convert_to_mol import convert_to_mol, substrate_as_smiles
from webKinPred.settings import MEDIA_ROOT

_AMINO_ACIDS = set("ACDEFGHIKLMNPQRSTVWY")
_log = logging.getLogger(__name__)


def dlkcat_predictions(
    sequences: list[str],
    public_id: str,
    substrates: list[str],
    canonicalize_substrates: bool = True,
    **kwargs,
) -> tuple[list, dict[int, str]]:
    """
    Run the DLKcat model on the given protein sequences and substrates.

    Parameters
    ----------
    sequences : list[str]
        Pre-filtered protein sequences.
    public_id : str
        Job identifier for progress tracking.
    substrates : list[str]
        Substrate SMILES or InChI strings, one per sequence.

    Returns
    -------
    predictions : list
        Predicted kcat values (float) or None for invalid rows.
    invalid_reasons : dict[int, str]
        Maps row index to a human-readable reason for rows that could not
        be processed.

    Raises
    ------
    PredictionError
        On subprocess failure or any unrecoverable error.
    """
    _log.info(
        "Prediction method started",
        extra={
            "event": "prediction.method_started",
            "job_public_id": public_id,
            "method_key": "DLKcat",
            "target": "kcat",
        },
    )

    job = Job.objects.get(public_id=public_id)
    reset_stage_prediction_metrics(
        job_public_id=public_id,
        target="kcat",
        method_key="DLKcat",
        total_rows=len(sequences),
    )

    python_path = PYTHON_PATHS.get("DLKcat", "")
    prediction_script = PREDICTION_SCRIPTS.get("DLKcat", "")
    job_dir = os.path.join(MEDIA_ROOT, "jobs", str(public_id))
    input_file = os.path.join(job_dir, f"input_{public_id}.tsv")
    output_file = os.path.join(job_dir, f"output_{public_id}.tsv")

    env = os.environ.copy()
    if DATA_PATHS.get("DLKcat"):
        env["DLKCAT_DATA_PATH"] = DATA_PATHS["DLKcat"]
    if DATA_PATHS.get("DLKcat_Results"):
        env["DLKCAT_RESULTS_PATH"] = DATA_PATHS["DLKcat_Results"]

    valid_indices: list[int] = []
    invalid_reasons: dict[int, str] = {}
    valid_smiles: list[str] = []
    valid_sequences: list[str] = []
    predictions: list = [None] * len(sequences)

    # ── Validate inputs molecule by molecule ──────────────────────────────────
    for idx, (seq, substrate) in enumerate(zip(sequences, substrates)):
        seq_valid = all(c in _AMINO_ACIDS for c in seq)
        mol = convert_to_mol(substrate) if seq_valid else None

        reason = ""
        if not seq_valid:
            reason = "Invalid protein sequence (unsupported amino acid characters)"
        elif mol is None:
            reason = "Invalid substrate (not a valid SMILES or InChI)"
        else:
            if len(Chem.GetMolFrags(mol)) > 1:
                reason = (
                    "Substrate contains multiple disconnected fragments and cannot be processed"
                )
            else:
                if canonicalize_substrates:
                    smiles = Chem.MolToSmiles(Chem.AddHs(mol))
                else:
                    smiles = substrate_as_smiles(
                        substrate,
                        canonicalize=False,
                        preserve_raw_smiles_when_possible=True,
                    )
                valid_smiles.append(smiles)
                valid_sequences.append(seq)
                valid_indices.append(idx)
                increment_stage_validation(
                    job_public_id=public_id,
                    target="kcat",
                    method_key="DLKcat",
                    processed_inc=1,
                    invalid_inc=0,
                )
                continue

        _log.debug(
            "Prediction row invalid",
            extra={
                "event": "prediction.row_invalid",
                "job_public_id": public_id,
                "method_key": "DLKcat",
                "target": "kcat",
                "row_index": idx,
                "reason": reason,
            },
        )
        invalid_reasons[idx] = reason
        increment_stage_validation(
            job_public_id=public_id,
            target="kcat",
            method_key="DLKcat",
            processed_inc=1,
            invalid_inc=1,
        )

    set_stage_prediction_total(
        job_public_id=public_id,
        target="kcat",
        method_key="DLKcat",
        total_predictions=len(valid_indices),
    )

    if not valid_indices:
        return predictions, invalid_reasons

    # ── Write TSV input file ──────────────────────────────────────────────────
    try:
        with open(input_file, "w") as f:
            f.write("Substrate Name\tSubstrate SMILES\tProtein Sequence\n")
            for smiles, seq in zip(valid_smiles, valid_sequences):
                f.write(f"noname\t{smiles}\t{seq}\n")
    except OSError as e:
        raise PredictionError(
            "DLKcat could not write its input file. Please contact support if this persists."
        ) from e

    # ── Run prediction subprocess ─────────────────────────────────────────────
    try:
        run_prediction_subprocess(
            command=[python_path, prediction_script, input_file, output_file],
            job=job,
            env=env,
            label="DLKcat",
            method_key="DLKcat",
            target="kcat",
            valid_sequences=valid_sequences,
        )
    except subprocess.CalledProcessError as e:
        _cleanup(input_file, output_file)
        if e.returncode in (-9, 137):
            raise PredictionError(
                "DLKcat ran out of memory. Try reducing the number of rows or the sequence lengths."
            ) from e
        raise PredictionError(
            "DLKcat encountered an internal error and could not complete. "
            "Please verify your input and try again."
        ) from e
    except Exception as e:
        _cleanup(input_file, output_file)
        if isinstance(e, PredictionError):
            raise
        raise PredictionError(
            "DLKcat encountered an unexpected error. Please verify your input and try again."
        ) from e

    # ── Read output TSV ───────────────────────────────────────────────────────
    try:
        with open(output_file, "r") as f:
            next(f)  # skip header
            raw_values: list[float | None] = []
            for line in f:
                parts = line.strip().split("\t")
                try:
                    raw_values.append(float(parts[3]))
                except (IndexError, ValueError):
                    raw_values.append(None)
    except (OSError, StopIteration) as e:
        _cleanup(input_file, output_file)
        raise PredictionError(
            "DLKcat completed but its output file could not be read. "
            "Please contact support if this persists."
        ) from e

    for local_idx, pred in enumerate(raw_values):
        global_idx = valid_indices[local_idx]
        predictions[global_idx] = None if pred in (None, np.nan) else pred

    _cleanup(input_file, output_file)
    return predictions, invalid_reasons


def _cleanup(*paths: str) -> None:
    for path in paths:
        try:
            if os.path.exists(path):
                os.remove(path)
        except OSError:
            pass
