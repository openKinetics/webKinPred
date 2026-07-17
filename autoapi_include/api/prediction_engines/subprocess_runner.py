# api/prediction_engines/subprocess_runner.py
#
# Shared helper for running prediction subprocesses and streaming their output.
#
# All prediction engines use this helper so that progress reporting and OOM
# detection are handled consistently in one place.

import logging
import os
import re
import subprocess
import time

from api.models import Job
from api.services.embedding_progress_service import (
    start_embedding_tracking,
    stop_embedding_tracking,
)
from api.services.job_progress_service import set_stage_prediction_progress

_ANSI_RE = re.compile(r"\x1b\[[0-9;?]*[ -/]*[@-~]")
_RATIO_RE = re.compile(r"(\d+)\s*/\s*(\d+)")
_WARNING_LINE_RE = re.compile(r"(error|exception|traceback|fatal|failed|warning|warn)", re.I)
_NOISY_LINE_RE = re.compile(
    r"(MoleculeModel\(|^\s*\(|^\s*\)|^\s*\w+\(|\d+%\||\d+/\d+\s*\[|^\s*$)"
)
_log = logging.getLogger(__name__)


def _strip_ansi(text: str) -> str:
    return _ANSI_RE.sub("", text)


class _CatPredProgressEstimator:
    """
    Estimate user-facing CatPred progress from nested tqdm output.

    CatPred prints nested tqdm bars (models × batches). Users should see
    progress in final predictions, so this estimator maps nested loops to
    [0..expected_predictions].
    """

    def __init__(self, expected_predictions: int):
        self.expected_predictions = max(int(expected_predictions), 0)
        self.model_total: int | None = None
        self.model_done_hint = 0
        self.inferred_completed_models = 0
        self.batch_total: int | None = None
        self.batch_done = 0
        self._last_batch_total: int | None = None
        self._last_batch_done = 0
        self._last_emitted_done = 0

    def _update_inner_batch(self, done: int, total: int) -> None:
        # If inner tqdm resets from near-complete to a lower value, one model pass finished.
        if (
            self._last_batch_total == total
            and done < self._last_batch_done
            and self._last_batch_done >= max(1, int(total * 0.95))
        ):
            self.inferred_completed_models += 1

        self.batch_total = total
        self.batch_done = done
        self._last_batch_total = total
        self._last_batch_done = done

    def _estimate_done(self) -> int | None:
        if self.expected_predictions <= 0:
            return None
        if not self.model_total or self.model_total <= 0:
            return None

        completed_models = max(self.inferred_completed_models, self.model_done_hint)
        if completed_models >= self.model_total:
            model_progress = float(self.model_total)
        else:
            batch_frac = 0.0
            if self.batch_total and self.batch_total > 0:
                batch_frac = max(0.0, min(1.0, self.batch_done / self.batch_total))
            model_progress = min(float(self.model_total), float(completed_models) + batch_frac)

        frac = model_progress / float(self.model_total)
        est_done = int(round(frac * self.expected_predictions))
        return max(0, min(self.expected_predictions, est_done))

    def ingest_line(self, line: str) -> int | None:
        cleaned = _strip_ansi(line)
        ratios = [(int(a), int(b)) for a, b in _RATIO_RE.findall(cleaned)]
        if not ratios:
            return None

        for done, total in ratios:
            if total <= 0 or done < 0 or done > total:
                continue

            # Outer model-loop tqdm is small (typically 10/10).
            if total <= 20:
                self.model_total = total
                self.model_done_hint = max(self.model_done_hint, done)
                self.inferred_completed_models = max(self.inferred_completed_models, done)
                continue

            # Inner batch-loop tqdm (e.g. 102/138).
            self._update_inner_batch(done, total)

        est_done = self._estimate_done()
        if est_done is None or est_done <= self._last_emitted_done:
            return None

        self._last_emitted_done = est_done
        return est_done


def run_prediction_subprocess(
    command: list[str],
    job: Job,
    env: dict | None = None,
    label: str = "subprocess",
    method_key: str | None = None,
    target: str | None = None,
    valid_sequences: list[str] | None = None,
) -> None:
    """
    Run a prediction subprocess, stream its stdout line-by-line, and update
    the job's progress fields whenever the script emits a progress line.

    Expected progress line format (written by prediction scripts)::

        Progress: <done>/<total>

    For example: ``Progress: 42/100``

    Parameters
    ----------
    command : list[str]
        Full command to execute, including the python interpreter path and
        all arguments.
    job : Job
        Django Job instance.  Its ``predictions_made`` and
        ``total_predictions`` fields are updated in real time.
    env : dict | None
        Environment variables passed to the subprocess.  Defaults to the
        current process environment if None.
    label : str
        Short label used in log output to identify which engine is running
        (e.g. ``"DLKcat"``).

    Raises
    ------
    subprocess.CalledProcessError
        If the subprocess exits with a non-zero return code.
    """
    tracking_started = False
    if method_key and target and valid_sequences:
        tracking_started = start_embedding_tracking(
            job_public_id=job.public_id,
            method_key=method_key,
            target=target,
            valid_sequences=valid_sequences,
            env=env,
        )

    process = None
    last_done_reported = -1
    last_total_reported: int | None = None
    output_lines = 0
    suppressed_lines = 0
    progress_events = 0
    last_progress_bucket = -1
    started_at = time.monotonic()
    catpred_estimator = (
        _CatPredProgressEstimator(expected_predictions=len(valid_sequences or []))
        if str(label).strip().lower() == "catpred"
        else None
    )

    def _command_summary() -> dict[str, str | int | None]:
        return {
            "executable": os.path.basename(command[0]) if command else None,
            "script": os.path.basename(command[1]) if len(command) > 1 else None,
            "arg_count": len(command),
        }

    def _base_extra() -> dict[str, object]:
        return {
            "job_public_id": job.public_id,
            "method_key": method_key or label,
            "target": target,
            "subprocess_label": label,
            "source": "method_subprocess",
        }

    def _log_progress(done_i: int, total_i: int | None, *, estimated: bool = False) -> None:
        nonlocal progress_events, last_progress_bucket
        progress_events += 1
        if total_i is None or total_i <= 0:
            return

        bucket = int((done_i / total_i) * 10) if total_i else 0
        should_log = total_i <= 10 or done_i in {0, 1, total_i} or bucket > last_progress_bucket
        if not should_log:
            return

        last_progress_bucket = max(last_progress_bucket, bucket)
        _log.info(
            "Subprocess prediction progress",
            extra={
                "event": "subprocess.progress",
                **_base_extra(),
                "progress_done": done_i,
                "progress_total": total_i,
                "progress_percent": round((done_i / total_i) * 100, 2),
                "estimated": estimated,
            },
        )

    def _log_output_line(line: str) -> None:
        nonlocal suppressed_lines
        clean_line = _strip_ansi(line).strip()
        if _WARNING_LINE_RE.search(clean_line):
            _log.warning(
                "Subprocess emitted warning/error output",
                extra={
                    "event": "subprocess.stderr_line",
                    **_base_extra(),
                    "line": clean_line[:2000],
                    "stream": "stdout_stderr",
                },
            )
            return

        if str(label).strip().lower() == "catpred" or _NOISY_LINE_RE.search(clean_line):
            suppressed_lines += 1
            return

        _log.debug(
            "Subprocess output",
            extra={
                "event": "subprocess.output",
                **_base_extra(),
                "line": clean_line[:2000],
                "stream": "stdout_stderr",
            },
        )

    def _report_prediction_progress(
        done_i: int,
        total_i: int | None = None,
        *,
        estimated: bool = False,
    ) -> None:
        nonlocal last_done_reported, last_total_reported

        done_i = max(0, int(done_i))
        if total_i is not None:
            total_i = max(0, int(total_i))
            done_i = min(done_i, total_i)

        if total_i is None:
            total_i = last_total_reported

        # Avoid duplicate/no-op DB writes.
        if total_i == last_total_reported and done_i <= last_done_reported:
            return

        if method_key and target:
            set_stage_prediction_progress(
                job_public_id=job.public_id,
                target=target,
                method_key=method_key,
                done=done_i,
                total=total_i,
            )
        else:
            job.predictions_made = done_i
            if total_i is not None:
                job.total_predictions = total_i
                job.save(update_fields=["predictions_made", "total_predictions"])
            else:
                job.save(update_fields=["predictions_made"])

        last_done_reported = done_i
        if total_i is not None:
            last_total_reported = total_i
        _log_progress(done_i, total_i, estimated=estimated)

    try:
        _log.info(
            "Starting prediction subprocess",
            extra={
                "event": "subprocess.started",
                **_base_extra(),
                **_command_summary(),
                "valid_sequence_count": len(valid_sequences or []),
            },
        )
        process = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            env=env,
        )

        if process.stdout is not None:
            for line in iter(process.stdout.readline, ""):
                if not line:
                    break

                line = line.rstrip()
                output_lines += 1

                if line.startswith("Progress:"):
                    parts = line.split()
                    if len(parts) >= 2:
                        try:
                            done, total = parts[1].split("/")
                            _report_prediction_progress(int(done), int(total))
                        except (ValueError, AttributeError):
                            _log.debug(
                                "Ignoring malformed subprocess progress line",
                                extra={
                                    "event": "subprocess.progress_malformed",
                                    **_base_extra(),
                                    "line": _strip_ansi(line)[:2000],
                                },
                            )
                    continue

                # CatPred prints tqdm bars instead of "Progress: x/y" during inference.
                # Estimate prediction progress from those bars so the frontend does not
                # stay flat and then jump to 100% at the end.
                if catpred_estimator is not None:
                    estimated_done = catpred_estimator.ingest_line(line)
                    if estimated_done is not None:
                        _report_prediction_progress(
                            estimated_done,
                            catpred_estimator.expected_predictions,
                            estimated=True,
                        )
                        continue

                _log_output_line(line)

        process.wait()

        if process.returncode != 0:
            _log.error(
                "Prediction subprocess failed",
                extra={
                    "event": "subprocess.failed",
                    **_base_extra(),
                    "return_code": process.returncode,
                    "duration_ms": int((time.monotonic() - started_at) * 1000),
                    "output_lines": output_lines,
                    "suppressed_lines": suppressed_lines,
                    "progress_events": progress_events,
                },
            )
            raise subprocess.CalledProcessError(process.returncode, process.args)
        _log.info(
            "Prediction subprocess completed",
            extra={
                "event": "subprocess.completed",
                **_base_extra(),
                "return_code": process.returncode,
                "duration_ms": int((time.monotonic() - started_at) * 1000),
                "output_lines": output_lines,
                "suppressed_lines": suppressed_lines,
                "progress_events": progress_events,
            },
        )
    finally:
        if tracking_started:
            final_state = "done" if (process is not None and process.returncode == 0) else "error"
            stop_embedding_tracking(job.public_id, final_state=final_state)
