"""Shared JSON serialization for completed prediction CSV files."""

from __future__ import annotations

from typing import Any

import pandas as pd


def serialize_result_csv(output_path: str) -> dict[str, Any]:
    """Read a result CSV and represent every blank cell as JSON ``null``."""
    dataframe = pd.read_csv(output_path)
    records = dataframe.astype(object).where(pd.notna(dataframe), None)
    return {
        "columns": list(dataframe.columns),
        "rowCount": len(dataframe),
        "data": records.to_dict(orient="records"),
    }
