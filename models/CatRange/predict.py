#!/usr/bin/env python3
"""Simple CatRange subprocess adapter for webKinPred.

This wrapper is intentionally lightweight: it loads the local CatRange
inference implementation, accepts the webKinPred JSON payload, and writes
predictions back in the expected schema.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import sys
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
CATRANGE_ROOT = Path(__file__).resolve().parent.resolve()
if str(CATRANGE_ROOT) not in sys.path:
    sys.path.insert(0, str(CATRANGE_ROOT))

try:
    from inference.catrange_inference import CatRangeInference
except Exception as exc:  # pragma: no cover - runtime setup
    raise SystemExit(f"CatRange import failed: {exc}") from exc

def _load_inference(models_dir: str | None = None):
    explicit_models_dir = models_dir or os.environ.get("CATRANGE_MODELS_DIR")
    if explicit_models_dir:
        base_dir = Path(explicit_models_dir).expanduser().resolve()
    else:
        base_dir = (Path(__file__).resolve().parent / "inference" / "models").resolve()
    return CatRangeInference(models_dir=base_dir, device="auto", verbose=False)


def _parse_range_label(range_label: str | None) -> tuple[float | None, str | None]:
    if not isinstance(range_label, str):
        return None, None

    text = range_label.strip()
    if not text:
        return None, None

    match = re.match(r"^([0-9eE.+-]+)\s+to\s+([0-9eE.+-]+)\s*(.*)$", text)
    if not match:
        return None, text

    low_text, high_text, suffix = match.groups()
    try:
        low = float(low_text)
        high = float(high_text)
    except ValueError:
        return None, text

    midpoint = math.sqrt(low * high) if low > 0 and high > 0 else (low + high) / 2.0
    if not math.isfinite(midpoint):
        midpoint = (low + high) / 2.0
    return midpoint, f"Predicted range: {text}"


def _build_prediction_payload(frame: Any, target: str) -> dict[str, Any]:
    parameter = "kcat" if target == "kcat" else "km"
    column = f"{parameter}_pred_range"
    predictions: list[Any] = []
    extra_info: list[str] = []
    for _, row in frame.iterrows():
        range_label = row.get(column)
        numeric_value, extra_text = _parse_range_label(range_label)
        predictions.append(numeric_value)
        extra_info.append(extra_text or "")
    return {"predictions": predictions, "extra_info": extra_info, "invalid_indices": []}


def predict_rows(rows: list[dict[str, Any]], target: str) -> tuple[list[Any], list[int], list[str]]:
    if not rows:
        return [], [], []

    parameter = "kcat" if target == "kcat" else "km"
    inference = _load_inference()
    pairs = []
    for row in rows:
        sequence = str(row.get("sequence", "")).strip()
        substrate = str(row.get("substrates") or row.get("Substrate") or "").strip()
        if not sequence or not substrate:
            raise ValueError("Each row requires a non-empty sequence and substrate")
        pairs.append((sequence, substrate))

    out = inference.predict(pairs=pairs, parameter=parameter)
    payload = _build_prediction_payload(out, target)
    return payload["predictions"], [], payload["extra_info"]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    with open(args.input, "r", encoding="utf-8") as fh:
        payload = json.load(fh)

    rows = payload.get("rows", [])
    target = payload.get("target", "kcat")

    try:
        predictions, invalid_indices, extra_info = predict_rows(rows, target)
    except Exception as exc:  # pragma: no cover - runtime fallback
        predictions = [None] * len(rows)
        invalid_indices = list(range(len(rows)))
        extra_info = [""] * len(rows)
        print(f"CatRange failed: {exc}", file=sys.stderr)

    with open(args.output, "w", encoding="utf-8") as fh:
        json.dump(
            {
                "predictions": predictions,
                "invalid_indices": invalid_indices,
                "extra_info": extra_info,
            },
            fh,
        )


if __name__ == "__main__":
    main()
