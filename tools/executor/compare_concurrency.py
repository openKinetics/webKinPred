"""Measure serial versus simultaneous isolated probes on identical copied inputs."""

import argparse
import json
import shutil
import statistics
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from api.execution.artifacts import atomic_json
from api.execution.resources import admit, reservation
from api.execution.telemetry import snapshot

from tools.executor.profile import check_availability, run_probe
from tools.executor.profiling_suite import dataset_manifest, measurement_peaks


def compare(configuration, destination):
    destination.mkdir(parents=True, exist_ok=False, mode=0o700)
    probes = configuration["probes"]
    if len(probes) < 2:
        raise ValueError("At least two probes are required for a concurrency comparison")
    envelopes = []
    for probe in probes:
        paths = probe.get("reference_reports") or [probe["reference_report"]]
        references = [json.loads(Path(path).read_text()) for path in paths]
        reference = references[0]
        manifest = dataset_manifest(Path(probe["dataset"]))
        if not references or any(
            row["exit_code"]
            or row["oom_killed"]
            or row["peaks"].get("ram", 0) <= 0
            or row["peaks"].get("pids", 0) <= 0
            or manifest["checksum"] != row["configuration"]["dataset_checksum"]
            or row["configuration"]["image"] != reference["configuration"]["image"]
            for row in references
        ):
            raise ValueError("Every simultaneous probe needs successful matching isolated evidence")
        measured = [measurement_peaks(row) for row in references]
        envelopes.append(
            reservation(
                {
                    key: max(row.get(key, 0) for row in measured)
                    for key in ("ram", "vram", "disk", "cpu", "pids")
                }
            )
        )
        probe["runtime"] = reference["configuration"]
    gpu_uuids = {p["runtime"]["gpu_uuid"] for p in probes if p["runtime"].get("gpu_uuid")}
    if len(gpu_uuids) > 1:
        raise ValueError("This comparison measures one GPU ledger at a time")
    gpu_uuid = next(iter(gpu_uuids), "")
    results = []

    def ensure_capacity(indices):
        # Driver memory accounting can settle after a container exits. Observe
        # fresh availability rather than treating that transient as a failed
        # comparison or launching into a still-resident CUDA context.
        deadline = time.monotonic() + configuration.get("capacity_wait_seconds", 30)
        while True:
            telemetry = snapshot(destination, gpu_uuid)
            telemetry["cpu_external"] = telemetry["cpu_used"]
            active = []
            decision = None
            for index in indices:
                check_availability(probes[index]["runtime"])
                decision = admit(
                    envelopes[index],
                    active,
                    configuration["budgets"],
                    telemetry,
                    gpu=bool(probes[index]["runtime"].get("gpu_uuid")),
                )
                if not decision.fits:
                    break
                active.append({"resources": envelopes[index], "observed": {}})
            if decision and decision.fits:
                return telemetry
            if time.monotonic() >= deadline:
                raise RuntimeError("Concurrency probe is waiting: " + decision.reason)
            time.sleep(1)

    def run(index, directory):
        probe = probes[index]
        work = directory / str(index)
        shutil.copytree(probe["dataset"], work)
        runtime = {**probe["runtime"], "scratch": str(work)}
        # Use the measured padded envelope as the containment boundary.
        runtime["memory_limit"] = envelopes[index]["ram"]
        runtime["pids_limit"] = envelopes[index]["pids"]
        report_path = directory / (str(index) + ".report.json")
        report = run_probe(runtime, report_path)
        if report["exit_code"] or report["oom_killed"]:
            raise RuntimeError("Concurrency probe failed; retain evidence and stop escalation")
        return str(report_path)

    for iteration in range(configuration.get("repetitions", 3)):
        serial_start = time.monotonic()
        serial = []
        for index in range(len(probes)):
            ensure_capacity([index])
            serial.append(run(index, destination / str(iteration) / "serial"))
        serial_seconds = time.monotonic() - serial_start
        baseline = ensure_capacity(range(len(probes)))
        concurrent_start = time.monotonic()
        with ThreadPoolExecutor(max_workers=len(probes)) as workers:
            futures = [
                workers.submit(run, index, destination / str(iteration) / "concurrent")
                for index in range(len(probes))
            ]
            concurrent = [future.result() for future in futures]
        concurrent_seconds = time.monotonic() - concurrent_start
        results.append(
            {
                "serial_seconds": serial_seconds,
                "concurrent_seconds": concurrent_seconds,
                "serial_reports": serial,
                "concurrent_reports": concurrent,
                "external_baseline": baseline,
            }
        )
        atomic_json(destination / "measurements.json", results)
    serial_seconds = statistics.median(row["serial_seconds"] for row in results)
    concurrent_seconds = statistics.median(row["concurrent_seconds"] for row in results)
    summary = {
        "repetitions": len(results),
        "probe_count": len(probes),
        "median_serial_seconds": serial_seconds,
        "median_concurrent_seconds": concurrent_seconds,
        "measured_speed_ratio": serial_seconds / concurrent_seconds,
        "recommended_concurrency": len(probes) if concurrent_seconds < serial_seconds * 0.95 else 1,
        "scope": "Identical stage probes, including startup/publication; not whole-job throughput",
    }
    atomic_json(destination / "comparison.json", summary)
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("configuration", type=Path)
    parser.add_argument("destination", type=Path)
    args = parser.parse_args()
    print(json.dumps(compare(json.loads(args.configuration.read_text()), args.destination)))
