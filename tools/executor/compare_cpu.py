"""Compare bounded CPU execution with one serial adapter call on copied inputs."""

from __future__ import annotations

import argparse
import copy
import json
import math
import shutil
from pathlib import Path

from api.execution.artifacts import atomic_json, sha256_file

from tools.executor.profile import run_probe
from tools.executor.profiling_suite import dataset_manifest


def compare_results(serial, staged, *, rtol=1e-5, atol=1e-6):
    """Preserve categorical outcomes, row order, reductions, and invalid inputs."""

    def check(before, after, path):
        if isinstance(before, float) and isinstance(after, (int, float)):
            if not (
                math.isfinite(before)
                and math.isfinite(after)
                and math.isclose(before, after, rel_tol=rtol, abs_tol=atol)
            ):
                raise ValueError("CPU numerical result changed: " + path)
        elif isinstance(before, dict) and isinstance(after, dict):
            if before.keys() != after.keys():
                raise ValueError("CPU result fields changed: " + path)
            for key in before:
                check(before[key], after[key], path + "." + key)
        elif isinstance(before, list) and isinstance(after, list):
            if len(before) != len(after):
                raise ValueError("CPU result row count changed: " + path)
            for index, (a, b) in enumerate(zip(before, after, strict=True)):
                check(a, b, path + f"[{index}]")
        elif isinstance(before, str) and isinstance(after, str) and before != after:
            # Expanded predictions and their annotations contain JSON arrays.
            # Compare numeric leaves at the declared tolerance, while retaining
            # exact sequence indices, labels, list order, and source annotations.
            try:
                a, b = json.loads(before), json.loads(after)
            except (ValueError, TypeError):
                raise ValueError("CPU categorical result metadata changed: " + path) from None
            check(a, b, path)
        elif type(before) is not type(after) or before != after:
            raise ValueError("CPU categorical result metadata changed: " + path)

    check(serial, staged, "result")


def compare(reference_path, dataset, destination):
    reference = json.loads(reference_path.read_text())
    runtime = reference["configuration"]
    if runtime.get("gpu_uuid") or reference["exit_code"] or reference["oom_killed"]:
        raise ValueError("A successful CPU reference is required")
    if dataset_manifest(dataset)["checksum"] != runtime["dataset_checksum"]:
        raise ValueError("Reference dataset changed")
    destination.mkdir(parents=True, exist_ok=False, mode=0o700)
    work = destination / "serial"
    shutil.copytree(dataset, work)
    request = json.loads((work / "request.json").read_text())
    if request["kind"] != "predict" or request["payload"].get("step"):
        raise ValueError("Only CPU prediction adapters have a serial unit comparison")
    units = request["payload"]["shape"]["units"]
    request["settings"]["unit_batch_size"] = max(1, units)
    atomic_json(work / "request.json", request)
    serial_runtime = copy.deepcopy(runtime)
    serial_runtime.update(
        scratch=str(work),
        dataset_checksum=dataset_manifest(work)["checksum"],
        run_mode="serial_compatibility",
    )
    report = run_probe(serial_runtime, destination / "serial.report.json")
    if report["exit_code"] or report["oom_killed"]:
        raise RuntimeError("Serial CPU comparison failed; inspect retained probe evidence")
    return validate_existing(reference_path, destination)


def validate_existing(reference_path, destination):
    reference = json.loads(reference_path.read_text())
    runtime = reference["configuration"]
    report = json.loads((destination / "serial.report.json").read_text())
    if any(row["exit_code"] or row["oom_killed"] for row in [reference, report]):
        raise ValueError("Failed probes cannot establish CPU compatibility")
    work = destination / "serial"
    request = json.loads((work / "request.json").read_text())
    output = request["payload"]["outputs"][0]["path"]
    serial_path = work / "media" / output
    staged_path = Path(runtime["scratch"]) / "media" / output
    compare_results(
        json.loads(serial_path.read_text())["result"],
        json.loads(staged_path.read_text())["result"],
    )
    evidence = {
        "passed": True,
        "scope": "Serial adapter versus bounded unit batches using identical cached features",
        "covered_limits": request["payload"]["shape"],
        "rtol": 1e-5,
        "atol": 1e-6,
        "reference_report": sha256_file(reference_path),
        "serial_report": sha256_file(destination / "serial.report.json"),
        "serial_result": sha256_file(serial_path),
        "staged_result": sha256_file(staged_path),
    }
    atomic_json(destination / "comparison.json", evidence)
    return evidence


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    for name in ("reference", "dataset", "destination"):
        parser.add_argument(name, type=Path)
    args = parser.parse_args()
    print(json.dumps(compare(args.reference, args.dataset, args.destination)))
