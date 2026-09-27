"""Render CatRange classes before their optional, derived numeric summaries.

Inference and reduction continue to use numeric values. This module only builds
the final CSV columns (also consumed by the JSON result endpoint). It recovers a
class only from an exact known midpoint and explicit CatRange provenance; it
never assigns an arbitrary measured value to a class.
"""

from __future__ import annotations

import json
import math
from typing import Any

import pandas as pd


# Keep synchronized with models/CatRange/inference/catrange_inference.py.
# Avoid importing the model module and its runtime dependencies into the server.
BIN_EDGES = {
    "kcat": (0.0, 1e-8, 1e-2, 1e-1, 1.0, 10.0, 100.0, 1000.0, 1e8),
    "Km": (1e-14, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1e4),
}
_MODEL_SOURCE = "Prediction from CatRange"
_EXPERIMENTAL_SOURCES = {"BRENDA", "Sabio-RK", "UniProt", "Experimental"}


def _base_source(source: Any) -> str:
    text = str(source or "")
    suffix = " (per substrate)"
    return text[:-len(suffix)] if text.endswith(suffix) else text


def _number(value: Any) -> float | None:
    if value is None or (isinstance(value, str) and value.strip().lower() in {"", "nan", "none"}):
        return None
    if isinstance(value, bool):
        raise ValueError("A CatRange reporting value cannot be a boolean")
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Invalid CatRange reporting value: {value!r}") from exc
    if math.isnan(number):
        return None
    if not math.isfinite(number):
        raise ValueError("A CatRange reporting value must be finite")
    return number


def _close(first: float, second: float) -> bool:
    return math.isclose(first, second, rel_tol=1e-12, abs_tol=0.0)


def _range_for_midpoint(target: str, value: float) -> str:
    unit = "s^-1" if target == "kcat" else "M"
    edges = BIN_EDGES[target]
    for low, high in zip(edges, edges[1:]):
        # Match the subprocess adapter, including its zero-lower-bound rule.
        midpoint = math.sqrt(low * high) if low > 0 else (low + high) / 2.0
        if _close(value, midpoint):
            return f"{low:g} to {high:g} {unit}"
    raise ValueError(
        f"CatRange {target} value {value!r} is not a known class midpoint; "
        "cannot report a predicted range"
    )


def _report_value(target: str, value: Any, source: Any) -> tuple[Any, Any, Any]:
    numeric = _number(value)
    if numeric is None:
        return None, None, None
    provenance = _base_source(source)
    if provenance == _MODEL_SOURCE:
        return _range_for_midpoint(target, numeric), numeric, None
    if provenance in _EXPERIMENTAL_SOURCES:
        return None, None, numeric
    raise ValueError(f"Cannot identify CatRange result provenance: {source!r}")


def _substrate_items(extra: Any) -> list[dict[str, Any]]:
    try:
        parsed = json.loads(extra) if isinstance(extra, str) else extra
    except (TypeError, ValueError):
        return []
    found: list[dict[str, Any]] = []

    def visit(item: Any) -> None:
        if isinstance(item, list):
            for child in item:
                visit(child)
        elif isinstance(item, dict):
            if "substrateIndex" in item:
                found.append(item)
            if "substrates" in item:
                visit(item["substrates"])

    visit(parsed)
    return found


def _array_sources(values: list[Any], source: Any, extra: Any) -> list[Any]:
    if _base_source(source) in {_MODEL_SOURCE, *_EXPERIMENTAL_SOURCES}:
        return [source] * len(values)

    items = _substrate_items(extra)
    sources: list[Any] = []
    for index, value in enumerate(values, start=1):
        numeric = _number(value)
        if numeric is None:
            sources.append("")
            continue
        candidates = [item for item in items if item.get("substrateIndex") == index]
        # Sequence expansion marks the winning sequence separately for each slot.
        if any("selected" in item for item in candidates):
            candidates = [item for item in candidates if item.get("selected") is True]
        candidates = [
            item for item in candidates
            if not item.get("error")
            and _number(item.get("prediction")) is not None
            and _close(_number(item["prediction"]), numeric)
        ]
        matched_sources = {_base_source(item.get("source")) for item in candidates}
        if len(matched_sources) != 1:
            raise ValueError(
                f"Cannot identify CatRange result provenance for substrate {index}"
            )
        sources.append(matched_sources.pop())
    return sources


def build_catrange_report_columns(target: str, result: dict[str, Any]) -> dict[str, list[Any]]:
    """Return ordered, presentation-only columns without mutating ``result``.

    The primary column contains class ranges. Measured values have blank ranges
    and appear only in the experimental column. Ordered Km arrays preserve null
    slots, including mixed model/experimental sources. Source and Extra Info are
    copied unchanged. The midpoint uses a geometric mean except the zero-lower
    kcat class, which follows the adapter's arithmetic midpoint convention.
    Experimental Km units must be confirmed against the source dataset; no
    implicit unit conversion is applied here.

    ``result`` must provide aligned ``preds``, ``sources``, and ``extra`` lists
    plus ``output_col``. Unsupported values/provenance raise ``ValueError``.
    """
    if target not in BIN_EDGES:
        raise ValueError(f"CatRange does not report target {target!r}")
    predictions, sources, extra = (result[key] for key in ("preds", "sources", "extra"))
    if not (len(predictions) == len(sources) == len(extra)):
        raise ValueError("CatRange reporting values, sources, and details must be aligned")

    ranges, midpoints, measured = [], [], []
    for value, source, details in zip(predictions, sources, extra):
        array = value if isinstance(value, (list, tuple)) else None
        if isinstance(value, str) and value.lstrip().startswith("["):
            try:
                array = json.loads(value)
            except ValueError as exc:
                raise ValueError("Invalid CatRange result array") from exc
        if array is not None:
            if target != "Km" or not isinstance(array, (list, tuple)):
                raise ValueError("Only CatRange Km results can contain per-substrate arrays")
            values = list(array)
            slot_sources = _array_sources(values, source, details)
            slots = [_report_value(target, item, origin) for item, origin in zip(values, slot_sources)]
            for column, position in ((ranges, 0), (midpoints, 1), (measured, 2)):
                column.append(json.dumps([slot[position] for slot in slots], allow_nan=False, separators=(",", ":")))
        else:
            reported = _report_value(target, value, source)
            for column, item in zip((ranges, midpoints, measured), reported):
                column.append("" if item is None else item)

    parameter, unit = ("kcat", "1/s") if target == "kcat" else ("KM", "M")
    experimental_unit = "1/s" if target == "kcat" else "source units"
    return {
        result["output_col"]: ranges,
        f"Source {target}": list(sources),
        f"Extra Info {target}": list(extra),
        f"Derived {parameter} range midpoint ({unit})": midpoints,
        f"Experimental {parameter} ({experimental_unit})": measured,
    }


def build_prediction_result_frame(
    dataframe: Any,
    targets: list[str],
    target_results: dict[str, dict[str, Any]],
    method_keys: dict[str, str],
) -> Any:
    """Assemble final output columns, preserving other methods and input order."""
    results = dataframe.copy()
    preferred: list[str] = []
    for target in targets:
        result = target_results[target]
        if method_keys[target] == "CatRange":
            columns = build_catrange_report_columns(target, result)
        else:
            columns = {
                result["output_col"]: result["preds"],
                f"Source {target}": result["sources"],
                f"Extra Info {target}": result["extra"],
            }
        for column, values in columns.items():
            results[column] = values
        preferred.extend(columns)
    return results[preferred + [column for column in results.columns if column not in preferred]]


def completed_reaction_count(
    targets: list[str], target_results: dict[str, dict[str, Any]]
) -> int:
    """Count rows with every requested numeric result, before display formatting.

    Experimental overrides count as completed even though their range cells are
    blank. This retains the worker's existing nonempty/non-null result rule and
    aligns all targets by row position, independent of input DataFrame indexes.
    """
    if not targets:
        return 0
    count = len(target_results[targets[0]]["preds"])
    completed = pd.Series(True, index=range(count))
    for target in targets:
        values = pd.Series(list(target_results[target]["preds"]), dtype=object)
        if len(values) != count:
            raise ValueError("Prediction targets must contain the same number of rows")
        completed &= values.ne("") & values.notna()
    return int(completed.sum())
