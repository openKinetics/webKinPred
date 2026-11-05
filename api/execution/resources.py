"""Pure conservative admission rules. All sizes are bytes, CPU is in cores."""

from __future__ import annotations

import math
from dataclasses import dataclass

GIB = 1024**3
MIB = 1024**2
LIVE_STATES = ("leased", "running", "uncertain")


@dataclass(frozen=True)
class Admission:
    fits: bool
    reason: str = ""


def reservation(peak: dict) -> dict:
    def padded(key, floor):
        value = int(peak.get(key, 0))
        return value + max(math.ceil(value / 4), floor)

    return {
        "ram": padded("ram", 512 * MIB),
        "vram": padded("vram", 256 * MIB) if peak.get("vram", 0) else 0,
        "disk": padded("disk", GIB),
        "cpu": float(peak.get("cpu", 1)),
        "pids": max(128, int(peak.get("pids", 1)) * 2),
    }


def cpu_capacity(budgets: dict, telemetry: dict) -> float:
    """CPUs that attempt reservations may occupy now: the budget, bounded by spare CPUs."""
    capacity = budgets.get("cpu", 0)
    if "cpu_external" in telemetry:
        spare = (
            telemetry.get("cpu_count", 0)
            - budgets.get("cpu_headroom", 1)
            - telemetry["cpu_external"]
        )
        capacity = min(capacity, spare * budgets.get("cpu_overcommit", 1))
    return capacity


def admit(
    candidate: dict, active: list[dict], budgets: dict, telemetry: dict, *, gpu: bool
) -> Admission:
    # NaN and missing telemetry must never accidentally compare as available capacity.
    for key in ("ram", "cpu", "disk", "ram_headroom", "disk_headroom", "vram_headroom"):
        value = budgets.get(key, 0) if key.endswith("headroom") else budgets.get(key)
        if not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
            return Admission(False, "invalid_budget")
    for key in ("ram_available", "disk_free") + (("vram_total", "vram_free") if gpu else ()):
        value = telemetry.get(key)
        if not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
            return Admission(False, "telemetry_unavailable")
    for key in ("ram", "vram", "disk", "cpu", "pids"):
        value = candidate.get(key, 0)
        if not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
            return Admission(False, "invalid_profile")
    if candidate.get("ram", 0) <= 0 or candidate.get("cpu", 0) <= 0:
        return Admission(False, "invalid_profile")
    # An attempt whose container has exited is only publishing outputs and
    # committing: its processes, memory and GPU contexts are gone. It keeps
    # its disk reservation (outputs are being copied) but no compute share.
    computing = [x for x in active if not x.get("observed", {}).get("released")]
    # Reservations are peak envelopes, and attempts spend much of their life
    # starting processes, loading weights or publishing at a fraction of that
    # peak. A host may therefore reserve a bounded multiple of its spare CPUs;
    # each container keeps its own CPU limit, and memory/GPU checks are exact.
    overcommit = budgets.get("cpu_overcommit", 1)
    if (
        not isinstance(overcommit, (int, float))
        or isinstance(overcommit, bool)
        or not math.isfinite(overcommit)
        or not 1 <= overcommit <= 4
    ):
        return Admission(False, "invalid_budget")
    if "cpu_external" in telemetry:
        external = telemetry["cpu_external"]
        if not isinstance(external, (int, float)) or not math.isfinite(external) or external < 0:
            return Admission(False, "telemetry_unavailable")
        spare = telemetry.get("cpu_count", 0) - budgets.get("cpu_headroom", 1) - external
        if (
            candidate["cpu"] + sum(x["resources"].get("cpu", 0) for x in computing)
            > spare * overcommit
        ):
            return Admission(False, "waiting_cpu")
    for key in ("ram", "cpu", "disk"):
        holders = active if key == "disk" else computing
        if (
            key not in budgets
            or sum(x["resources"].get(key, 0) for x in holders) + candidate.get(key, 0)
            > budgets[key]
        ):
            return Admission(False, "waiting_" + key)
    if any(x.get("observed", {}).get("vram", 0) > x["resources"].get("vram", 0) for x in active):
        return Admission(False, "profile_vram_breach")
    if telemetry.get("memory_pressure", False) or telemetry.get("io_pressure", False):
        return Admission(False, "host_pressure")
    for resource, available, guard in (
        ("ram", "ram_available", "ram_headroom"),
        ("disk", "disk_free", "disk_headroom"),
    ):
        unspent = sum(
            max(0, x["resources"].get(resource, 0) - x.get("observed", {}).get(resource, 0))
            for x in (active if resource == "disk" else computing)
        )
        if (
            available not in telemetry
            or candidate.get(resource, 0) + unspent + budgets.get(guard, 0) > telemetry[available]
        ):
            return Admission(False, "waiting_" + resource)
    if not gpu and "cpu_slots" in budgets:
        slots = budgets["cpu_slots"]
        if not isinstance(slots, int) or isinstance(slots, bool) or slots < 1:
            return Admission(False, "invalid_budget")
        if sum(not x["resources"].get("vram", 0) for x in computing) >= slots:
            return Admission(False, "waiting_cpu")
    if gpu:
        if (
            not telemetry.get("gpu_uuid")
            or "vram_free" not in telemetry
            or "vram_total" not in telemetry
        ):
            return Admission(False, "gpu_unavailable")
        guard = max(
            2 * GIB, math.ceil(telemetry["vram_total"] / 10), budgets.get("vram_headroom", 0)
        )
        unspent = sum(
            max(0, x["resources"].get("vram", 0) - x.get("observed", {}).get("vram", 0))
            for x in computing
        )
        if candidate.get("vram", 0) + unspent + guard > telemetry["vram_free"]:
            return Admission(False, "waiting_vram")
        if sum(bool(x["resources"].get("vram")) for x in computing) >= budgets.get("gpu_slots", 1):
            return Admission(False, "waiting_gpu_slot")
    elif candidate.get("vram", 0):
        return Admission(False, "invalid_cpu_profile")
    return Admission(True)
