"""
Similarity analysis service that orchestrates the similarity workflow.
"""

import json
import logging
import os
import subprocess
import tempfile
from typing import Any, Dict, List, Optional

import pandas as pd
from api.methods.base import PredictionError
from api.services.progress_service import push_line
from api.utils.similarity_config import SIMILARITY_DATASETS, TARGET_DBS
from api.utils.similarity_utils import (
    TMP_DIR,
    _mmseqs_cmd,
    calculate_average_similarity,
    calculate_identity_histogram,
    cleanup_temporary_files,
    create_fasta_file,
    create_mmseqs_database,
    create_unique_sequence_mapping,
    extract_protein_sequences_from_csv,
    map_results_to_original_sequences,
    parse_mmseqs_results,
    parse_mmseqs_results_raw,
    run_mmseqs_search,
)
from api.utils.sequence_expansion import split_sequence_list

_log = logging.getLogger(__name__)


class SimilarityCacheOnlyMiss(PredictionError):
    """A strict synchronous run could not use its preflight snapshot."""


_SIMILARITY_SEQUENCE_ALPHABET = set("ACDEFGHIKLMNPQRSTVWY")


def _find_merged_db() -> tuple[str, str] | tuple[None, None]:
    """Return (merged_db_path, membership_json_path) if both exist, else (None, None)."""
    datasets = SIMILARITY_DATASETS or {}
    any_dataset = next(iter(datasets.values()), {})
    any_target_db = any_dataset.get("target_db", "")
    if not any_target_db:
        return None, None
    dbs_dir = os.path.dirname(any_target_db)
    merged_db = os.path.join(dbs_dir, "targetdb_merged")
    membership_path = f"{merged_db}_membership.json"
    if (os.path.exists(merged_db) or os.path.exists(f"{merged_db}.dbtype")) and os.path.exists(
        membership_path
    ):
        return merged_db, membership_path
    return None, None


def _load_membership(membership_path: str) -> Dict[str, List[str]]:
    with open(membership_path) as f:
        return json.load(f)


def analyze_sequence_similarity(csv_file, session_id: str = "default") -> Dict[str, Any]:
    """
    Analyze sequence similarity against target databases.

    Args:
        csv_file: Uploaded CSV file containing protein sequences
        session_id: Session ID for logging

    Returns:
        Dictionary containing similarity analysis results

    Raises:
        ValueError: If CSV is invalid or contains no sequences
        Exception: If analysis fails
    """
    input_sequences = extract_protein_sequences_from_csv(csv_file)
    unique_sequences, seq_to_unique_id = create_unique_sequence_mapping(input_sequences)
    query_file_path = create_fasta_file(unique_sequences, seq_to_unique_id)
    temp_files_to_cleanup = [query_file_path]

    try:
        query_db, temp_query_dir = create_mmseqs_database(query_file_path, session_id)
        temp_files_to_cleanup.append(temp_query_dir)

        method_histograms = {}

        datasets = SIMILARITY_DATASETS or {
            label: {"label": label, "target_db": path} for label, path in TARGET_DBS.items()
        }

        # Filter to datasets whose DBs are present on disk
        active_datasets: Dict[str, Any] = {}
        for _dataset_key, dataset in datasets.items():
            label = dataset.get("label") or _dataset_key
            target_db = dataset.get("target_db")
            if not target_db:
                push_line(session_id, f"[WARN] Skipping dataset '{label}' (missing target_db path)")
                continue
            if not (os.path.exists(target_db) or os.path.exists(f"{target_db}.dbtype")):
                push_line(session_id, f"[WARN] Skipping dataset '{label}' (DB files not found)")
                continue
            active_datasets[label] = dataset

        merged_db, membership_path = _find_merged_db()

        if merged_db and active_datasets:
            membership = _load_membership(membership_path)

            # Build reverse index: label -> set of mseq IDs
            label_to_targets: Dict[str, set] = {}
            for mseq_id, labels in membership.items():
                for lbl in labels:
                    label_to_targets.setdefault(lbl, set()).add(mseq_id)

            covered = [lbl for lbl in active_datasets if lbl in label_to_targets]
            uncovered = [lbl for lbl in active_datasets if lbl not in label_to_targets]

            if uncovered:
                push_line(
                    session_id,
                    f"[WARN] Not in merged DB, searching individually: {', '.join(uncovered)}",
                )

            if covered:
                # Single search against the merged DB; no --max-seqs truncation so
                # every hit that would be found in any individual search is present.
                push_line(session_id, "==> Running single merged search")
                result_file = run_mmseqs_search(
                    query_db, merged_db, "merged", session_id, max_seqs=len(membership)
                )
                temp_files_to_cleanup.append(os.path.dirname(result_file))

                raw_hits = parse_mmseqs_results_raw(result_file)

                for label in covered:
                    push_line(session_id, f"==> Processing DB: {label}")
                    target_ids = label_to_targets[label]

                    # Filter hits to this DB's sequences; 0.0 for queries with no hits
                    identity_lists: Dict[str, List[float]] = {}
                    for query_id, hits in raw_hits.items():
                        filtered = [p for t, p in hits if t in target_ids]
                        if filtered:
                            identity_lists[query_id] = filtered
                    for unique_id in seq_to_unique_id.values():
                        if unique_id not in identity_lists:
                            identity_lists[unique_id] = [0.0]

                    unique_max = {k: max(v) for k, v in identity_lists.items()}
                    unique_mean = {k: sum(v) / len(v) for k, v in identity_lists.items()}

                    query_to_max, query_to_mean = map_results_to_original_sequences(
                        unique_max, unique_mean, input_sequences, seq_to_unique_id
                    )
                    histogram_max_counts, histogram_max_perc = calculate_identity_histogram(
                        query_to_max
                    )
                    histogram_mean_counts, histogram_mean_perc = calculate_identity_histogram(
                        query_to_mean
                    )
                    method_histograms[label] = {
                        "histogram_max": histogram_max_perc,
                        "histogram_mean": histogram_mean_perc,
                        "average_max_similarity": calculate_average_similarity(query_to_max),
                        "average_mean_similarity": calculate_average_similarity(query_to_mean),
                        "count_max": histogram_max_counts,
                        "count_mean": histogram_mean_counts,
                    }
                    push_line(session_id, f"--> [{label}] Aggregated {len(query_to_max)} sequences")

            # Individual fallback for any dataset not covered by the merged DB
            for label in uncovered:
                target_db = active_datasets[label].get("target_db")
                push_line(session_id, f"==> Processing DB: {label}")
                method_histograms[label] = analyze_similarity_for_method(
                    query_db, target_db, query_file_path, label,
                    input_sequences, seq_to_unique_id, session_id,
                )

        else:
            # No merged DB available — original per-DB loop
            for label, dataset in active_datasets.items():
                target_db = dataset.get("target_db")
                push_line(session_id, f"==> Processing DB: {label}")
                method_histograms[label] = analyze_similarity_for_method(
                    query_db, target_db, query_file_path, label,
                    input_sequences, seq_to_unique_id, session_id,
                )

        if not method_histograms:
            raise ValueError(
                "No similarity datasets are available. "
                "Add a dataset in similarity config and build its MMseqs DB."
            )

        return method_histograms

    finally:
        cleanup_temporary_files(*temp_files_to_cleanup)


def analyze_similarity_for_method(
    query_db: str,
    target_db: str,
    query_file_path: str,
    method_name: str,
    original_sequences: List[str],
    seq_to_unique_id: Dict[str, str],
    session_id: str,
) -> Dict[str, Any]:
    """
    Analyze similarity for a specific method/database.

    Args:
        query_db: Path to query database
        target_db: Path to target database
        query_file_path: Path to original FASTA file
        method_name: Name of the method
        original_sequences: Original sequence list
        seq_to_unique_id: Sequence to unique ID mapping
        session_id: Session ID for logging

    Returns:
        Dictionary containing method-specific results
    """
    result_file = None

    try:
        # Run MMseqs2 search
        result_file = run_mmseqs_search(query_db, target_db, method_name, session_id)

        # Parse results to get identity scores
        unique_max_identity, unique_mean_identity = parse_mmseqs_results(
            result_file, query_file_path
        )

        # Map results back to original sequences
        query_to_max, query_to_mean = map_results_to_original_sequences(
            unique_max_identity, unique_mean_identity, original_sequences, seq_to_unique_id
        )

        # Calculate histograms
        histogram_max_counts, histogram_max_perc = calculate_identity_histogram(query_to_max)
        histogram_mean_counts, histogram_mean_perc = calculate_identity_histogram(query_to_mean)

        # Calculate averages
        average_max_similarity = calculate_average_similarity(query_to_max)
        average_mean_similarity = calculate_average_similarity(query_to_mean)

        push_line(session_id, f"--> [{method_name}] Aggregated {len(query_to_max)} sequences")

        return {
            "histogram_max": histogram_max_perc,
            "histogram_mean": histogram_mean_perc,
            "average_max_similarity": average_max_similarity,
            "average_mean_similarity": average_mean_similarity,
            "count_max": histogram_max_counts,
            "count_mean": histogram_mean_counts,
        }

    finally:
        # Clean up result file and its parent directory
        if result_file and os.path.exists(result_file):
            result_dir = os.path.dirname(result_file)
            cleanup_temporary_files(result_dir)


def _similarity_column_names(method_key: str) -> tuple[str, str]:
    return (
        f"mean similarity to {method_key} training data",
        f"max similarity to {method_key} training data",
    )


def _resolve_similarity_dataset_for_method(method_key: str) -> tuple[Optional[str], Optional[str]]:
    datasets = SIMILARITY_DATASETS or {
        label: {"label": label, "target_db": path} for label, path in TARGET_DBS.items()
    }

    for dataset_key, dataset in datasets.items():
        dataset_methods = set(dataset.get("method_keys") or [])
        if method_key in dataset_methods:
            label = dataset.get("label") or dataset_key
            return label, dataset.get("target_db")

    return None, None


def similarity_cache_label_for_method(method_key: str) -> str:
    """Return the persistent-cache dataset label used by output enrichment."""
    label, _target_db = _resolve_similarity_dataset_for_method(method_key)
    return label or method_key


def _run_mmseqs_command(cmd: list[str]) -> None:
    proc = subprocess.run(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    if proc.returncode != 0:
        output = (proc.stdout or "").strip()
        if output:
            output = output.splitlines()[-1]
            raise RuntimeError(f"MMseqs command failed: {' '.join(cmd)} :: {output}")
        raise RuntimeError(f"MMseqs command failed: {' '.join(cmd)}")


def _write_blank_similarity_columns(
    output_csv_path: str,
    mean_col: str,
    max_col: str,
) -> None:
    try:
        df = pd.read_csv(output_csv_path)
    except Exception as exc:
        _log.warning(
            "Could not read output CSV to add blank similarity columns",
            extra={
                "event": "similarity.output_blank_columns_read_failed",
                "output_csv_path": output_csv_path,
                "exception_type": type(exc).__name__,
            },
            exc_info=True,
        )
        return

    df[mean_col] = ""
    df[max_col] = ""
    try:
        df.to_csv(output_csv_path, index=False)
    except Exception as exc:
        _log.warning(
            "Could not write blank similarity columns to output CSV",
            extra={
                "event": "similarity.output_blank_columns_write_failed",
                "output_csv_path": output_csv_path,
                "exception_type": type(exc).__name__,
            },
            exc_info=True,
        )


def _compute_mmseqs_similarity(
    sequences: list[str],
    target_db: str,
    temp_files_to_cleanup: list[str],
) -> tuple[dict[str, float], dict[str, float]]:
    """
    Run MMseqs2 for ``sequences`` against ``target_db``.

    Returns ``(sequence_to_max, sequence_to_mean)`` percent-identity dicts. The
    query id map is built internally so this can be called with any subset of
    sequences (e.g. only ReconXKG cache misses).
    """
    seq_to_unique_id: dict[str, str] = {}
    for seq in sequences:
        if seq and seq not in seq_to_unique_id:
            seq_to_unique_id[seq] = f"useq{len(seq_to_unique_id)}"

    if not seq_to_unique_id:
        return {}, {}

    query_fasta_path = create_fasta_file(list(seq_to_unique_id.keys()), seq_to_unique_id)
    temp_files_to_cleanup.append(query_fasta_path)

    query_dir = tempfile.mkdtemp(dir=TMP_DIR)
    temp_files_to_cleanup.append(query_dir)
    query_db = os.path.join(query_dir, "queryDB")
    _run_mmseqs_command(_mmseqs_cmd("createdb", query_fasta_path, query_db))

    result_dir = tempfile.mkdtemp(dir=TMP_DIR)
    temp_files_to_cleanup.append(result_dir)
    result_db = os.path.join(result_dir, "resultDB")
    result_file = os.path.join(result_dir, "result.m8")

    _run_mmseqs_command(
        _mmseqs_cmd(
            "search",
            query_db,
            target_db,
            result_db,
            result_dir,
            "--max-seqs",
            "1000",
            "-s",
            "7.5",
            "-e",
            "0.001",
            "-v",
            "0",
        )
    )
    _run_mmseqs_command(
        _mmseqs_cmd(
            "convertalis",
            query_db,
            target_db,
            result_db,
            result_file,
            "--format-output",
            "query,target,pident",
        )
    )

    unique_max_identity, unique_mean_identity = parse_mmseqs_results(
        result_file,
        query_fasta_path,
    )

    sequence_to_max = {
        seq: round(float(unique_max_identity.get(unique_id, 0.0)), 2)
        for seq, unique_id in seq_to_unique_id.items()
    }
    sequence_to_mean = {
        seq: round(float(unique_mean_identity.get(unique_id, 0.0)), 2)
        for seq, unique_id in seq_to_unique_id.items()
    }
    return sequence_to_max, sequence_to_mean


def append_kcat_similarity_columns_to_output_csv(
    output_csv_path: str,
    kcat_method_key: str,
    recon_xkg: bool = False,
    cached_similarity_snapshot: dict[
        str, tuple[float | None, float | None]
    ] | None = None,
    cache_only: bool = False,
    selected_sequences_by_row: list[str] | None = None,
) -> None:
    """
    Best-effort enrichment for completed kcat jobs.

    Adds two per-row columns to output.csv:
      - mean similarity to {method} training data
      - max similarity to {method} training data

    When ``recon_xkg`` is set, per-sequence similarity is served from the
    persistent SimilarityStore and MMseqs2 runs only for sequences not yet
    cached. A cached ``NULL``/``NULL`` pair is a negative cache hit and renders
    as blank cells, so repeated deterministic failures do not retry MMseqs2.
    On any error, both columns are still created with blank values.
    """
    mean_col, max_col = _similarity_column_names(kcat_method_key)
    temp_files_to_cleanup: list[str] = []
    cache_label = kcat_method_key
    seq_sha_by_seq: dict[str, str] = {}
    store = None
    unique_sequences: list[str] = []

    try:
        df = pd.read_csv(output_csv_path)
        if "Protein Sequence" not in df.columns:
            raise ValueError('Output CSV is missing required "Protein Sequence" column')

        dataset_label, target_db = _resolve_similarity_dataset_for_method(kcat_method_key)
        raw_sequences = _row_similarity_sequences(df, selected_sequences_by_row)
        unique_sequences = _unique_nonempty_sequences(raw_sequences)

        if not unique_sequences and cache_only:
            _write_blank_similarity_columns(output_csv_path, mean_col, max_col)
            return
        if not unique_sequences:
            raise ValueError("No non-empty protein sequences available for similarity analysis")

        sequence_to_max: dict[str, float | None] = {}
        sequence_to_mean: dict[str, float | None] = {}
        sequences_to_compute = unique_sequences
        cache_label = dataset_label or kcat_method_key

        if cache_only:
            cached = cached_similarity_snapshot or {}
            missing = [seq for seq in unique_sequences if seq not in cached]
            if missing:
                raise SimilarityCacheOnlyMiss(
                    "A similarity value was absent from the ReconXKG preflight snapshot."
                )
            for seq in unique_sequences:
                mean_sim, max_sim = cached[seq]
                sequence_to_mean[seq] = mean_sim
                sequence_to_max[seq] = max_sim
            sequences_to_compute = []
        elif recon_xkg:
            from api.services import prediction_store as store  # noqa: PLC0415

            seq_sha_by_seq = {seq: store.sha256_text(seq) for seq in unique_sequences}
            cached = store.get_similarity_many(seq_sha_by_seq, cache_label)
            for seq, (mean_sim, max_sim) in cached.items():
                sequence_to_mean[seq] = mean_sim
                sequence_to_max[seq] = max_sim
            sequences_to_compute = [seq for seq in unique_sequences if seq not in cached]

        invalid_sequences = [
            seq for seq in sequences_to_compute if not _valid_similarity_sequence(seq)
        ]
        if invalid_sequences:
            _mark_blank_similarity_values(
                invalid_sequences,
                sequence_to_mean,
                sequence_to_max,
            )
            if recon_xkg and store is not None:
                _cache_blank_similarity_entries(
                    store,
                    invalid_sequences,
                    seq_sha_by_seq,
                    cache_label,
                )
            invalid_set = set(invalid_sequences)
            sequences_to_compute = [
                seq for seq in sequences_to_compute if seq not in invalid_set
            ]

        if sequences_to_compute:
            db_available = bool(
                target_db
                and (os.path.exists(target_db) or os.path.exists(f"{target_db}.dbtype"))
            )
            if not db_available:
                _log.warning(
                    "Similarity DB unavailable; using blank similarity values",
                    extra={
                        "event": "similarity.output_db_unavailable",
                        "method_key": kcat_method_key,
                        "target_db": target_db,
                        "sequence_count": len(sequences_to_compute),
                    },
                )
                _mark_blank_similarity_values(
                    sequences_to_compute,
                    sequence_to_mean,
                    sequence_to_max,
                )
                if recon_xkg and store is not None:
                    _cache_blank_similarity_entries(
                        store,
                        sequences_to_compute,
                        seq_sha_by_seq,
                        cache_label,
                    )
            else:
                try:
                    computed_max, computed_mean = _compute_mmseqs_similarity(
                        sequences_to_compute, target_db, temp_files_to_cleanup
                    )
                except Exception as exc:
                    _log.warning(
                        "MMseqs similarity computation failed; using blank similarity values",
                        extra={
                            "event": "similarity.output_compute_failed",
                            "method_key": kcat_method_key,
                            "target_db": target_db,
                            "sequence_count": len(sequences_to_compute),
                            "exception_type": type(exc).__name__,
                        },
                        exc_info=True,
                    )
                    _mark_blank_similarity_values(
                        sequences_to_compute,
                        sequence_to_mean,
                        sequence_to_max,
                    )
                    if recon_xkg and store is not None:
                        _cache_blank_similarity_entries(
                            store,
                            sequences_to_compute,
                            seq_sha_by_seq,
                            cache_label,
                        )
                else:
                    sequence_to_max.update(computed_max)
                    sequence_to_mean.update(computed_mean)

                    if recon_xkg and store is not None:
                        store.upsert_similarity_many(
                            [
                                (
                                    seq,
                                    seq_sha_by_seq[seq],
                                    computed_mean.get(seq, 0.0),
                                    computed_max.get(seq, 0.0),
                                )
                                for seq in sequences_to_compute
                            ],
                            cache_label,
                        )

        mean_values: list[float | str] = []
        max_values: list[float | str] = []
        for seq in raw_sequences:
            if not seq:
                mean_values.append("")
                max_values.append("")
                continue
            mean_values.append(_similarity_output_cell(sequence_to_mean.get(seq, 0.0)))
            max_values.append(_similarity_output_cell(sequence_to_max.get(seq, 0.0)))

        df[mean_col] = mean_values
        df[max_col] = max_values
        df.to_csv(output_csv_path, index=False)

    except Exception as exc:
        if cache_only:
            raise
        _log.warning(
            "Could not enrich output CSV with kcat similarity columns",
            extra={
                "event": "similarity.output_enrichment_failed",
                "output_csv_path": output_csv_path,
                "method_key": kcat_method_key,
                "exception_type": type(exc).__name__,
            },
            exc_info=True,
        )
        _write_blank_similarity_columns(output_csv_path, mean_col, max_col)
    finally:
        cleanup_temporary_files(*temp_files_to_cleanup)


def _row_similarity_sequences(
    df: pd.DataFrame,
    selected_sequences_by_row: list[str] | None,
) -> list[str]:
    raw_values = df["Protein Sequence"].fillna("").tolist()
    selected = selected_sequences_by_row or []
    out: list[str] = []
    for index, raw_value in enumerate(raw_values):
        explicit = ""
        if index < len(selected):
            explicit = str(selected[index] or "").strip()
        if explicit:
            out.append(explicit)
            continue

        from_extra = _selected_sequence_from_extra_info(
            df["Extra Info kcat"].iloc[index] if "Extra Info kcat" in df.columns else ""
        )
        if from_extra:
            out.append(from_extra)
            continue

        tokens = split_sequence_list(raw_value)
        out.append(tokens[0] if len(tokens) == 1 else "")
    return out


def kcat_similarity_sequences_for_output_rows(
    df: pd.DataFrame,
    selected_sequences_by_row: list[str] | None = None,
) -> list[str]:
    """Return the unique sequence keys kcat output enrichment will require."""
    return _unique_nonempty_sequences(
        _row_similarity_sequences(df, selected_sequences_by_row)
    )


def _unique_nonempty_sequences(sequences: list[str]) -> list[str]:
    unique_sequences: list[str] = []
    seen: set[str] = set()
    for seq in sequences:
        if seq and seq not in seen:
            seen.add(seq)
            unique_sequences.append(seq)
    return unique_sequences


def _valid_similarity_sequence(sequence: str) -> bool:
    return bool(sequence) and all(
        char in _SIMILARITY_SEQUENCE_ALPHABET for char in sequence
    )


def _mark_blank_similarity_values(
    sequences: list[str],
    sequence_to_mean: dict[str, float | None],
    sequence_to_max: dict[str, float | None],
) -> None:
    for seq in sequences:
        sequence_to_mean[seq] = None
        sequence_to_max[seq] = None


def _cache_blank_similarity_entries(
    store: Any,
    sequences: list[str],
    seq_sha_by_seq: dict[str, str],
    cache_label: str,
) -> int:
    entries = [
        (
            seq,
            seq_sha_by_seq.get(seq) or store.sha256_text(seq),
            None,
            None,
        )
        for seq in sequences
    ]
    return store.upsert_similarity_many(entries, cache_label)


def _similarity_output_cell(value: float | None) -> float | str:
    return "" if value is None else value


def _selected_sequence_from_extra_info(value: object) -> str:
    if value is None:
        return ""
    try:
        if pd.isna(value):
            return ""
    except (TypeError, ValueError):
        pass
    text = str(value).strip()
    if not text:
        return ""
    try:
        payload = json.loads(text)
    except (TypeError, ValueError, json.JSONDecodeError):
        return ""
    if not isinstance(payload, list):
        return ""
    for sequence_item in payload:
        if not isinstance(sequence_item, dict):
            continue
        if sequence_item.get("selected"):
            return str(sequence_item.get("sequence") or "").strip()
        substrates = sequence_item.get("substrates")
        if isinstance(substrates, list) and any(
            isinstance(item, dict) and item.get("selected") for item in substrates
        ):
            return str(sequence_item.get("sequence") or "").strip()
    return ""
