"""Assemble the final prediction CSV/JSON columns and count completed rows.

Column layout matches every other method: one prediction column per target
(named by ``output_col``), plus ``Source {target}`` and ``Extra Info {target}``.
CatRange values are rendered as class ranges at this final output boundary.
Inference, caching, reduction and completion counting retain numeric values.
"""

from __future__ import annotations

import json
import math
from typing import Any

import pandas as pd

from api.methods.catrange import descriptor as catrange_descriptor


# Match the inference bin edges without importing the model runtime.
BIN_EDGES = {
    "kcat": (0, 1e-8, 1e-2, 1e-1, 1, 10, 100, 1000, 1e8),
    "Km": (1e-14, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1e4),
}
_MODEL_SOURCE = "Prediction from CatRange"
_MIXED_SOURCE = "Mixed per-substrate sources; see Extra Info"


def catrange_range_label(target: str, value: Any) -> Any:
    """Recover a class only from its known adapter midpoint; preserve blanks."""
    if value is None or value == "" or pd.isna(value):
        return value
    edges = BIN_EDGES[target]
    unit = "s^-1" if target == "kcat" else "M"
    for low, high in zip(edges, edges[1:]):
        midpoint = math.sqrt(low * high) if low > 0 else (low + high) / 2
        if not isinstance(value, bool) and math.isclose(
            float(value), midpoint, rel_tol=1e-12, abs_tol=0
        ):
            return f"{low:g} to {high:g} {unit}"
    raise ValueError(f"Cannot report CatRange {target} range: {value!r} is not a known class midpoint")


def _report_value(target: str, value: Any, source: str) -> Any:
    if source.removesuffix(" (per substrate)") == _MODEL_SOURCE:
        return catrange_range_label(target, value)
    return value


def _prediction_items(node: Any):
    """Visit both substrate-only and nested sequence/substrate metadata."""
    if isinstance(node, list):
        for child in node:
            yield from _prediction_items(child)
    elif isinstance(node, dict):
        if "prediction" in node:
            yield node
        for child in node.values():
            if isinstance(child, (list, dict)):
                yield from _prediction_items(child)


def _slot_source(index: int, value: Any, source: str, items: list[dict]) -> str:
    if value is None or source != _MIXED_SOURCE:
        return source
    candidates = [item for item in items if item.get("substrateIndex") == index]
    if any("selected" in item for item in candidates):
        candidates = [item for item in candidates if item.get("selected") is True]
    origins = {
        item.get("source", "") for item in candidates
        if not item.get("error") and item.get("prediction") is not None
        and math.isclose(float(item["prediction"]), float(value), rel_tol=1e-12, abs_tol=0)
    }
    if len(origins) != 1 or "" in origins:
        raise ValueError(f"Cannot identify CatRange result source for substrate {index}")
    return origins.pop()


def _report_row(target: str, value: Any, source: str, extra: str) -> tuple[Any, str]:
    try:
        metadata = json.loads(extra)
    except (TypeError, ValueError):
        metadata = None
    items = list(_prediction_items(metadata))
    if isinstance(value, str) and value.lstrip().startswith("["):
        values = json.loads(value)
        value = json.dumps([
            _report_value(target, item, _slot_source(index, item, source, items))
            for index, item in enumerate(values, start=1)
        ], separators=(",", ":"), allow_nan=False)
    else:
        value = _report_value(target, value, source)
    changed = False
    for item in items:
        if item.get("source") == _MODEL_SOURCE:
            item["prediction"] = catrange_range_label(target, item["prediction"])
            changed = True
    if changed:
        extra = json.dumps(metadata, separators=(",", ":"), allow_nan=False)
    return value, extra


def build_prediction_result_frame(
    dataframe: Any,
    targets: list[str],
    target_results: dict[str, dict[str, Any]],
) -> Any:
    """Assemble final output columns, preserving other methods and input order."""
    results = dataframe.copy()
    preferred: list[str] = []
    for target in targets:
        result = target_results[target]
        pred_col = result["output_col"]
        source_col = f"Source {target}"
        extra_col = f"Extra Info {target}"
        predictions, extra = result["preds"], result["extra"]
        if pred_col == catrange_descriptor.output_cols.get(target):
            if not (len(predictions) == len(result["sources"]) == len(extra)):
                raise ValueError("CatRange results, sources and details must have matching lengths")
            rows = [
                _report_row(target, value, source, details)
                for value, source, details in zip(predictions, result["sources"], extra)
            ]
            predictions = [row[0] for row in rows]
            extra = [row[1] for row in rows]
        results[pred_col] = predictions
        results[source_col] = result["sources"]
        results[extra_col] = extra
        preferred.extend([pred_col, source_col, extra_col])
    return results[preferred + [column for column in results.columns if column not in preferred]]


def completed_reaction_count(
    targets: list[str], target_results: dict[str, dict[str, Any]]
) -> int:
    """Count rows with every requested numeric result, before display formatting.

    Aligns all targets by row position, independent of input DataFrame indexes.
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
