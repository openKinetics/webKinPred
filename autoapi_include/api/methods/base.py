# api/methods/base.py
#
# Core types shared across the method registry system.

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Literal


# ---------------------------------------------------------------------------
# Prediction target type aliases
# ---------------------------------------------------------------------------

PredictionTarget = Literal["kcat", "Km", "kcat/Km"]
"""A kinetic parameter that a method can predict."""

InputBehavior = Literal["expanded_pair", "native_multi", "native_full_reaction"]
"""How a method target consumes a user-facing ``Substrates`` input."""

InputFormat = Literal["single", "multi", "full reaction"]
"""
The user-facing CSV input format.

- "single": requires a single "Substrate" column (one SMILES/InChI per row).
- "multi": requires a "Substrates" column with semicolon-separated values
           (for example, "A;B").
- "full reaction": requires "Substrates" and "Products" columns
                   (semicolon-separated SMILES/InChI per row).

The Multi-Substrate schema supplies ``Substrates`` without ``Products``. The
orchestration layer expands that schema for pair-based targets and passes it
through as one ordered collection for native multi-substrate targets.

Backend descriptors use ``DescriptorInputFormat`` below. In descriptors and
validation logic, user-facing "single" uses the "single" column contract.
User-facing "full reaction" is represented by the backend "multi" contract
with both "Substrates" and "Products" columns.
"""

DescriptorInputFormat = Literal["single", "multi"]
"""
The backend CSV column contract stored on method descriptors.

- "single": reads the "Substrate" column.
- "multi": reads the "Substrates" and "Products" columns, corresponding to
           the user-facing "full reaction" input format.

The orchestration layer can also adapt a user-facing, semicolon-separated
``Substrates`` column to any ``"single"`` descriptor by expanding it into
individual protein/substrate pairs. Descriptors continue to declare only their
native engine contract here.
"""


# ---------------------------------------------------------------------------
# PredictionError — the canonical exception for prediction engine failures
# ---------------------------------------------------------------------------


class PredictionError(Exception):
    """
    Raised by prediction engines when an unrecoverable error occurs during
    prediction.

    The exception message is shown directly to the user as the job's error
    message, so it must be written in plain English without technical jargon,
    internal file paths, or stack traces.

    Prediction engines should catch all subprocess and I/O errors and re-raise
    them as PredictionError with a descriptive message.  Examples::

        raise PredictionError(
            "DLKcat ran out of memory. Try reducing the number of rows or the sequence lengths."
        )

        raise PredictionError(
            "EITLEM encountered an internal error and could not complete. "
            "Please verify your input and try again."
        )
    """


# ---------------------------------------------------------------------------
# SubprocessEngineConfig — configuration for the generic subprocess engine
# ---------------------------------------------------------------------------


@dataclass
class SubprocessEngineConfig:
    """
    Configuration for the built-in generic subprocess prediction engine.

    When a method descriptor provides this config (and does not provide
    ``pred_func``), the framework handles:

    1. Row-level validation (sequence + molecule format checks),
    2. Input file creation,
    3. Subprocess execution (with progress streaming),
    4. Output file parsing and index remapping.

    This allows a new method to be integrated without writing a custom
    ``api/prediction_engines/<method>.py`` module.
    """

    # Python executable resolution:
    # - If `python_path` is set, it is used directly.
    # - Otherwise `python_path_key` is looked up in PYTHON_PATHS config.
    python_path_key: str | None = None
    python_path: str | None = None

    # Prediction script resolution:
    # - If `script_path` is set, it is used directly.
    # - Otherwise `script_key` is looked up in PREDICTION_SCRIPTS config.
    script_key: str | None = None
    script_path: str | None = None

    # CLI contract used by the generic engine:
    #   python <script> <extra_args...> --input <input.json> --output <output.json>
    input_flag: str = "--input"
    output_flag: str = "--output"
    extra_args: list[str] = field(default_factory=list)

    # Environment injection:
    #   data_path_env maps ENV_VAR -> DATA_PATHS key.
    #   extra_env adds static ENV_VAR -> literal value.
    data_path_env: dict[str, str] = field(default_factory=dict)
    extra_env: dict[str, str] = field(default_factory=dict)

    # Validation policy used before launching the subprocess.
    validate_sequence: bool = True
    validate_chemistry: bool = True
    allowed_amino_acids: str = "ACDEFGHIKLMNPQRSTVWY"

    # When True, any incomplete GPU precompute result (offline, timeout,
    # request failure, incomplete remote outputs) is treated as a hard failure
    # for this method stage instead of falling back to local CPU execution.
    fail_on_gpu_precompute_failure: bool = False


# ---------------------------------------------------------------------------
# MethodDescriptor — full specification of one prediction method
# ---------------------------------------------------------------------------


@dataclass
class MethodDescriptor:
    """
    Complete specification for a kinetic-parameter prediction method.

    One instance of this class is defined in each ``api/methods/<method>.py``
    module as a module-level variable named ``descriptor``.  The registry
    loads all descriptors automatically — no manual registration is required.

    Parameters
    ----------
    key : str
        Unique, stable identifier for this method.  Used as the method ID in
        API requests (e.g. ``"DLKcat"``, ``"KinForm-H"``).  Must not contain
        spaces.

    display_name : str
        Human-readable name shown in the UI and included in result CSVs
        (e.g. ``"KinForm-H"``).

    authors : str
        Author list for the underlying publication.

    publication_title : str
        Full title of the paper describing this method.

    citation_url : str
        URL of the paper (DOI link or journal page).

    repo_url : str
        URL of the method's source-code repository.

    more_info : str
        Optional extra guidance shown to users (e.g. "Recommended for …").

    supports : list[PredictionTarget]
        Which kinetic parameters this method can predict.
        E.g. ``["kcat"]``, ``["kcat", "Km"]``.

    input_format : DescriptorInputFormat
        Backend CSV column contract expected by the method: ``"single"``
        reads the ``Substrate`` column, while ``"multi"`` reads the
        ``Substrates`` and ``Products`` columns. In user-facing text, the
        visible CSV formats are ``"single"``, ``"multi"``, and
        ``"full reaction"`` (the ``Substrates``/``Products`` backend
        ``"multi"`` contract).

    input_behavior_by_target : dict[PredictionTarget, InputBehavior]
        Optional target-level override for user-facing ``Substrates`` input.
        Single-contract descriptors default to ``expanded_pair`` and
        full-reaction descriptors default to ``native_full_reaction``.
        CatPred uses ``native_multi`` for kcat while retaining
        ``expanded_pair`` for Km.

    output_cols : dict[str, str]
        Maps each supported prediction target to the output CSV column name.
        E.g. ``{"kcat": "kcat (1/s)", "Km": "KM (mM)"}``.

    max_seq_len : int | float
        Maximum protein sequence length accepted by the method.
        Use ``float("inf")`` if the method has no length limit.

    col_to_kwarg : dict[str, str]
        Maps CSV column names to the keyword argument names of ``pred_func``.
        The framework reads each listed column from the input DataFrame and
        passes it to ``pred_func`` under the mapped name.

        Examples::

            {"Substrate": "substrates"}  # single-substrate methods
            {
                "Substrates": "substrates",  # TurNup
                "Products": "products",
            }

    target_kwargs : dict[str, dict]
        Extra static keyword arguments passed to ``pred_func`` depending on
        the prediction target.  Use this for methods that handle both kcat
        and Km through the same function but need different flags::

            {
                "kcat": {"kinetics_type": "KCAT"},
                "Km": {"kinetics_type": "KM"},
            }

        For methods that only support one target, this can be empty or omitted.

    pred_func : Callable | None
        Optional custom prediction function.  Must follow this exact
        signature if provided::

            def pred_func(
                sequences: list[str],
                public_id: str,
                **kwargs,
            ) -> tuple[list, list[int]]:
                \"\"\"
                Parameters
                ----------
                sequences
                    Pre-filtered, possibly truncated protein sequences.
                    Length matches the number of valid rows passed to this call.
                public_id
                    Job identifier used to update progress in the database.
                **kwargs
                    Assembled by the framework from col_to_kwarg (CSV-derived
                    inputs) and target_kwargs (target-specific flags).

                Returns
                -------
                predictions
                    One value per input sequence, in the same order.
                    Use ``None`` or empty string for rows where a prediction
                    could not be made.
                invalid_indices
                    Zero-based indices (into the ``sequences`` list) of rows
                    rejected due to invalid substrate or sequence format.

                Raises
                ------
                PredictionError
                    When the prediction cannot complete due to a model-level
                    error (subprocess failure, out of memory, timeout, …).
                    The message is shown verbatim to the user.
                \"\"\"

        The function usually lives in
        ``api/prediction_engines/<method>.py``.

        If omitted, set ``subprocess`` (below) to use the built-in generic
        subprocess engine instead.

    subprocess : SubprocessEngineConfig | None
        Configuration for the built-in generic subprocess engine.
        Recommended for most new methods because it avoids writing a custom
        engine module.

    embeddings_used : list[str]
        Informational list of embedding model keys used by this method.
        Check ``api/embeddings/registry.py`` for which of these are already
        implemented in the shared infrastructure.  Embeddings that are already
        implemented can be reused directly by referencing the conda python path
        from config (as KinForm does) rather than bundling a separate copy.
    """

    # ── Identity ──────────────────────────────────────────────────────────────
    key: str
    display_name: str
    authors: str
    publication_title: str
    citation_url: str
    repo_url: str
    more_info: str = ""

    # ── Model/weights version ─────────────────────────────────────────────────
    # Stable identifier for the model weights/algorithm version. Bumping this
    # repartitions the ReconXKG prediction cache (new keys => recompute), so it
    # MUST be incremented whenever a change would alter predicted values for the
    # same inputs (new checkpoint, preprocessing change, etc.).
    model_version: str = "1"

    # ── Capabilities ──────────────────────────────────────────────────────────
    supports: list[PredictionTarget] = field(default_factory=list)
    input_format: DescriptorInputFormat = "single"
    input_behavior_by_target: dict[PredictionTarget, InputBehavior] = field(
        default_factory=dict
    )
    output_cols: dict[str, str] = field(default_factory=dict)

    # ── Sequence length limit ─────────────────────────────────────────────────
    max_seq_len: int | float = float("inf")

    # ── Input column → pred_func kwarg mapping ────────────────────────────────
    col_to_kwarg: dict[str, str] = field(default_factory=dict)

    # ── Per-target extra kwargs for pred_func ─────────────────────────────────
    target_kwargs: dict[str, dict] = field(default_factory=dict)

    # ── The prediction callable ───────────────────────────────────────────────
    pred_func: Callable | None = None
    subprocess: SubprocessEngineConfig | None = None

    # ── Embedding models used (informational) ─────────────────────────────────
    embeddings_used: list[str] = field(default_factory=list)

    def input_behavior(self, target: PredictionTarget) -> InputBehavior:
        """Return the target-specific orchestration behavior."""
        configured = self.input_behavior_by_target.get(target)
        if configured:
            return configured
        if self.input_format == "multi":
            return "native_full_reaction"
        return "expanded_pair"

    def accepted_csv_types_for_target(self, target: PredictionTarget) -> list[str]:
        """Return user-facing CSV schemas accepted for one prediction target."""
        behavior = self.input_behavior(target)
        if behavior == "native_full_reaction":
            return ["full_reaction"]
        if behavior == "native_multi":
            return ["multi", "full_reaction"]
        return ["single", "multi", "full_reaction"]
