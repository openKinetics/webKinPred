"""Assemble the final prediction CSV/JSON columns and count completed rows.

Column layout matches every other method: one prediction column per target
(named by ``output_col``), plus ``Source {target}`` and ``Extra Info {target}``.
CatRange's predicted-range text already arrives in ``Extra Info`` from the
model subprocess, so no CatRange-specific column handling is needed here.
"""

from __future__ import annotations

from typing import Any

import pandas as pd


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
        results[pred_col] = result["preds"]
        results[source_col] = result["sources"]
        results[extra_col] = result["extra"]
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
