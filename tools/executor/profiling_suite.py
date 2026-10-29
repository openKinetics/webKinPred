"""Reproducible probe matrix and conservative registry compiler.

The compiler deliberately requires external scientific parity and shape-coverage
evidence. Successful model execution alone cannot activate a resource profile.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path

from api.execution.artifacts import atomic_json, safe_relative, sha256_file
from api.execution.resources import reservation
from api.execution.validation import gpu_numerical_waiver

from tools.executor.profile import run_probe


def digest(value):
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()


def dataset_manifest(directory):
    root = directory.resolve()
    files = {}
    for path in sorted(root.rglob("*")):
        if path.is_symlink():
            raise ValueError("Profiling inputs must be copied regular files")
        if path.is_file():
            files[str(path.relative_to(root))] = sha256_file(path)
    if not files:
        raise ValueError("Empty profiling dataset")
    return {"files": files, "checksum": digest(files)}


def measurement_peaks(report):
    peaks = dict(report["peaks"])
    # File-backed pages can be charged to an older container until eviction.
    # A fresh process alone therefore does not establish a cold cgroup peak.
    # Reserve at least the whole process tree RSS, including shared libraries.
    peaks["ram"] = max(
        peaks.get("ram", 0),
        max(
            (sample.get("process_tree", {}).get("rss", 0) for sample in report.get("samples", [])),
            default=0,
        ),
    )
    peaks["vram"] = max(
        peaks.get("vram", 0),
        sum(value.get("reserved_peak", 0) for value in report.get("framework_metrics", [])),
    )
    return peaks


def run_suite(configuration, destination):
    # An existing directory may contain production media: never reuse it implicitly.
    destination.mkdir(parents=True, exist_ok=False, mode=0o700)
    inputs = Path(configuration["dataset"]).resolve()
    manifest = dataset_manifest(inputs)
    atomic_json(destination / "dataset.json", manifest)
    reports = []
    for probe in configuration["probes"]:
        ident = safe_relative(probe["id"])
        if len(ident.parts) != 1:
            raise ValueError("Probe IDs must be single path components")
        for mode, count in (("fresh", 3), ("warm", 5)):
            for index in range(count):
                root = destination / str(ident) / f"{mode}-{index}"
                shutil.copytree(inputs, root)
                settings = {
                    **configuration["runtime"],
                    **probe["runtime"],
                    "isolated": True,
                    "dataset_checksum": manifest["checksum"],
                    "scratch": str(root),
                    "probe_id": str(ident),
                    "run_mode": mode,
                }
                warm_scope = probe.get("warm_scope", "process")
                if warm_scope not in {"process", "filesystem"}:
                    raise ValueError("Unknown warm measurement scope")
                settings["warm_scope"] = warm_scope if mode == "warm" else "fresh_process"
                if mode == "warm" and warm_scope == "process" and not probe.get("warm_command"):
                    raise ValueError(
                        "Warm probes require an adapter command that reuses loaded models"
                    )
                if mode == "warm" and probe.get("warm_command"):
                    settings["command"] = probe["warm_command"]
                report_path = root.parent / (root.name + ".report.json")
                report = run_probe(settings, report_path)
                reports.append({"probe": str(ident), "mode": mode, "report": str(report_path)})
                atomic_json(destination / "runs.json", reports)
                if report["exit_code"] or report["oom_killed"]:
                    raise RuntimeError("Probe failed; stop escalation and review isolated evidence")


def compile_registry(configuration, runs, parity, destination):
    registry = []
    rows = json.loads(runs.read_text())
    for probe in configuration["probes"]:
        selected = [row for row in rows if row["probe"] == probe["id"]]
        if (
            sum(row["mode"] == "fresh" for row in selected) < 3
            or sum(row["mode"] == "warm" for row in selected) < 5
        ):
            raise ValueError("Incomplete fresh/warm measurements")
        reports = [json.loads(Path(row["report"]).read_text()) for row in selected]
        if any(report["exit_code"] or report["oom_killed"] for report in reports):
            raise ValueError("Failed probes cannot validate an envelope")
        if any(
            report["peaks"].get("ram", 0) <= 0
            or report["peaks"].get("pids", 0) <= 0
            or (probe["kind"] == "embedding" and report["peaks"].get("vram", 0) <= 0)
            for report in reports
        ):
            raise ValueError("Incomplete resource measurements; check host PID/cgroup visibility")
        check = parity.get(probe["id"], {})
        waived = probe["kind"] == "embedding" and gpu_numerical_waiver(check)
        if (check.get("passed") is not True and not waived) or check.get("covered_limits") != probe[
            "limits"
        ]:
            raise ValueError("Scientific parity and exact shape coverage are required")
        checksums = {r["configuration"]["dataset_checksum"] for r in reports}
        if len(checksums) != 1:
            raise ValueError("Mixed dataset identities")
        measured = [measurement_peaks(report) for report in reports]
        resources = reservation(
            {
                key: max(peaks.get(key, 0) for peaks in measured)
                for key in ("ram", "vram", "disk", "cpu", "pids")
            }
        )
        evidence = {
            "fresh_runs": 3,
            "warm_runs": 5,
            "parity_passed": check.get("passed") is True,
            "gpu_numerical_waiver": check.get("gpu_numerical_waiver", {}),
            "covered_limits": probe["limits"],
            "dataset_checksum": checksums.pop(),
            "parity": check,
            "runtime": configuration["identity"],
            "reports": [sha256_file(Path(row["report"])) for row in selected],
        }
        profile = {
            "id": probe["id"],
            "stage_kind": probe["kind"],
            "method": probe["method"],
            "image": probe["runtime"].get("image", configuration["runtime"]["image"]),
            "settings": probe["settings"],
            "resources": resources,
            "limits": probe["limits"],
            "evidence": evidence,
        }
        profile["fingerprint"] = digest({k: v for k, v in profile.items() if k != "id"})
        registry.append(profile)
    atomic_json(destination, registry)
    return registry


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=["run", "compile", "manifest"])
    parser.add_argument("source", type=Path)
    parser.add_argument("destination", type=Path)
    parser.add_argument("--runs", type=Path)
    parser.add_argument("--parity", type=Path)
    args = parser.parse_args()
    if args.action == "manifest":
        atomic_json(args.destination, dataset_manifest(args.source))
    elif args.action == "run":
        run_suite(json.loads(args.source.read_text()), args.destination)
    else:
        compile_registry(
            json.loads(args.source.read_text()),
            args.runs,
            json.loads(args.parity.read_text()),
            args.destination,
        )


if __name__ == "__main__":
    main()
