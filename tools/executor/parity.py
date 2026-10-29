"""Compare exported scientific artifacts without touching jobs or caches."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
from api.execution.artifacts import atomic_json, safe_relative, sha256_file


def compare(baseline, staged, configuration):
    evidence = []
    for entry in configuration["files"]:
        relative = safe_relative(entry["path"])
        left, right = baseline / relative, staged / relative
        if entry["kind"] == "embedding":
            a, b = np.load(left, allow_pickle=False), np.load(right, allow_pickle=False)
            if a.shape != b.shape or a.dtype != b.dtype:
                raise ValueError(f"Embedding shape/dtype changed: {relative}")
            np.testing.assert_allclose(a, b, rtol=1e-5, atol=1e-6, equal_nan=False)
        elif entry["kind"] == "csv":
            with left.open(newline="") as first, right.open(newline="") as second:
                a, b = csv.reader(first), csv.reader(second)
                if next(a) != next(b):
                    raise ValueError("Result columns changed")
                from itertools import zip_longest

                numeric = entry.get("numeric_columns", {})
                for position, (x, y) in enumerate(zip_longest(a, b)):
                    if x is None or y is None or len(x) != len(y):
                        raise ValueError("Result row count or width changed")
                    for column, (old, new) in enumerate(zip(x, y, strict=False)):
                        tolerance = numeric.get(str(column))
                        if tolerance and old and new:
                            np.testing.assert_allclose(
                                float(old),
                                float(new),
                                rtol=tolerance["rtol"],
                                atol=tolerance["atol"],
                                equal_nan=False,
                            )
                        elif old != new:
                            raise ValueError(f"Result changed at row {position}, column {column}")
        else:
            if sha256_file(left) != sha256_file(right):
                raise ValueError("Artifact bytes changed: " + str(relative))
        evidence.append(
            {"path": str(relative), "baseline": sha256_file(left), "staged": sha256_file(right)}
        )
    if not evidence:
        raise ValueError("No scientific artifacts compared")
    return {"passed": True, "covered_limits": configuration["covered_limits"], "files": evidence}


def main():
    parser = argparse.ArgumentParser()
    for name in ("baseline", "staged", "configuration", "output"):
        parser.add_argument(name, type=Path)
    args = parser.parse_args()
    atomic_json(
        args.output, compare(args.baseline, args.staged, json.loads(args.configuration.read_text()))
    )


if __name__ == "__main__":
    main()
