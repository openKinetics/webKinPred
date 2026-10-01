"""Shared JSON serialization for completed prediction CSV files."""

from __future__ import annotations

from typing import Any

import pandas as pd

from api.methods.catrange import descriptor as catrange_descriptor


def serialize_result_csv(output_path: str) -> dict[str, Any]:
    """Read a result CSV and represent every blank cell as JSON ``null``."""
    dataframe = pd.read_csv(output_path)
    records = dataframe.astype(object).where(pd.notna(dataframe), None)
    # Range strings make pandas read the entire CatRange column as text.
    # Keep measured scalar values numeric; leave ranges and encoded arrays alone.
    for column in catrange_descriptor.output_cols.values():
        if column in records:
            numeric = pd.to_numeric(records[column], errors="coerce")
            mask = numeric.notna()
            records.loc[mask, column] = numeric[mask]
    return {
        "columns": list(dataframe.columns),
        "rowCount": len(dataframe),
        "data": records.to_dict(orient="records"),
    }
