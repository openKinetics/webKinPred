#!/usr/bin/env python3
"""Simple CatRange subprocess adapter for webKinPred.

This wrapper is intentionally lightweight: it loads the local CatRange
inference implementation, accepts the webKinPred JSON payload, and writes
predictions back in the expected schema.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
CATRANGE_ROOT = Path(__file__).resolve().parent.resolve()
if str(CATRANGE_ROOT) not in sys.path:
    sys.path.insert(0, str(CATRANGE_ROOT))

try:
    from inference.catrange_inference import CatRangeInference
except Exception as exc:  # pragma: no cover - import guard for runtime setup
    raise SystemExit(f"CatRange import failed: {exc}") from exc


def _load_inference(models_dir: str | None = None):
    explicit_models_dir = models_dir or os.environ.get("CATRANGE_MODELS_DIR")
    if explicit_models_dir:
        base_dir = Path(explicit_models_dir).expanduser().resolve()
    else:
        base_dir = (Path(__file__).resolve().parent / "inference" / "models").resolve()
    return CatRangeInference(models_dir=base_dir, device="auto", verbose=False)


def predict_rows(rows: list[dict[str, Any]], target: str) -> tuple[list[Any], list[int]]:
    if not rows:
        return [], []

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
    preds: list[Any] = []
    for row in out.itertuples(index=False):
        if parameter == "kcat":
            range_label = getattr(row, "kcat_pred_range", None)
            preds.append(range_label)
        else:
            range_label = getattr(row, "km_pred_range", None)
            preds.append(range_label)
    return preds, []


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
        predictions, invalid_indices = predict_rows(rows, target)
    except Exception as exc:  # pragma: no cover - runtime fallback
        predictions = [None] * len(rows)
        invalid_indices = list(range(len(rows)))
        print(f"CatRange failed: {exc}", file=sys.stderr)

    with open(args.output, "w", encoding="utf-8") as fh:
        json.dump({"predictions": predictions, "invalid_indices": invalid_indices}, fh)


if __name__ == "__main__":
    main()
