#!/usr/bin/env python3
"""Delete Loki chunks past retention at a pace production's rotating disk tolerates.

Until 26 November 2026, Loki's retention_period was never applied (no compactor retention),
so production kept every chunk since April 2026: 183,000 files, 179,000 of them past 7 days.
The compactor now applies retention, but its sweeper unlinks without pacing, and production's
supervisors refuse launches above 5 % full I/O pressure. This script removes chunk files
older than MAX_AGE_DAYS (beyond the 7-day retention plus the sweeper's 24 h delay) before the
sweeper reaches them; the sweeper treats missing chunks as already deleted.

Usage (as root on production): python3 cleanup_loki_chunks.py LOG.jsonl
"""
import json
import os
import subprocess
import sys
import time

ROOT = os.environ.get(
    "LOKI_CHUNKS", "/var/lib/docker/volumes/webkinpred_loki_data/_data/chunks/fake"
)
MAX_AGE_DAYS = float(os.environ.get("MAX_AGE_DAYS", "8"))
# Same pacing as tools/cleanup_legacy_sequence_cache.py: 750 unlinks a second gave one-minute
# full-pressure peaks of 8-10 % on this disk; 200 a second, and only while no request is
# unfinished, kept launches unaffected. Stat calls are paced with the unlinks.
LIMIT = float(os.environ.get("CLEANUP_MAX_IO_FULL", "2.0"))
LIMIT60 = float(os.environ.get("CLEANUP_MAX_IO_FULL60", "2.5"))
BATCH = 100
PAUSE = 0.5
UNFINISHED = (
    "docker", "exec", "webkinpred-execution-db-1", "psql", "-U", "webkinpred",
    "-d", "webkinpred_live_20261028", "-tAc",
    "select count(*) from api_job where status in ('Pending', 'Processing')",
)
checked_at = 0.0
LOG = sys.argv[1]


def log(**record):
    record["at"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    with open(LOG, "a") as handle:
        handle.write(json.dumps(record) + "\n")


def pressure(field=1):
    with open("/proc/pressure/io") as handle:
        for line in handle:
            if line.startswith("full"):
                return float(line.split()[field].split("=")[1])
    return 0.0


def busy():
    """True while requests are unfinished (or the count fails); checked every 10 s."""
    global checked_at
    if time.monotonic() - checked_at < 10:
        return False
    try:
        result = subprocess.run(UNFINISHED, capture_output=True, text=True, timeout=20)
        unfinished = int(result.stdout.strip())
    except (OSError, ValueError, subprocess.SubprocessError):
        unfinished = -1
    if unfinished == 0:
        checked_at = time.monotonic()
        return False
    return True


def quiet():
    waited = 0
    while busy():
        time.sleep(30)
        waited += 30
    while pressure() > LIMIT or pressure(2) > LIMIT60:
        time.sleep(5)
        waited += 5
    return waited


def main():
    cutoff = time.time() - MAX_AGE_DAYS * 86400
    totals = {"seen": 0, "files": 0, "bytes": 0, "dirs": 0, "paused_s": 0}
    log(event="start", root=ROOT, max_age_days=MAX_AGE_DAYS, pressure=pressure())
    operations = 0
    totals["paused_s"] += quiet()
    with os.scandir(ROOT) as streams:
        stream_dirs = [e.path for e in streams if e.is_dir(follow_symlinks=False)]
    for scanned, directory in enumerate(stream_dirs, 1):
        if os.path.islink(directory) or not os.path.realpath(directory).startswith(ROOT + "/"):
            raise RuntimeError("unexpected path: " + directory)
        with os.scandir(directory) as entries:
            files = [e for e in entries if e.is_file(follow_symlinks=False)]
        for entry in files:
            info = entry.stat(follow_symlinks=False)
            totals["seen"] += 1
            operations += 1
            if info.st_mtime < cutoff:
                os.unlink(entry.path)
                totals["files"] += 1
                totals["bytes"] += info.st_size
                operations += 1
            if operations >= BATCH:
                operations = 0
                totals["paused_s"] += quiet()
                time.sleep(PAUSE)
        try:
            os.rmdir(directory)  # only succeeds when every chunk in the stream was old
            totals["dirs"] += 1
        except OSError:
            pass
        if scanned % 500 == 0:
            log(event="progress", streams=scanned, of=len(stream_dirs), **totals)
    log(event="done", **totals)


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        log(event="error", error=repr(error))
        raise
