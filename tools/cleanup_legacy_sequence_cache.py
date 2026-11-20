#!/usr/bin/env python3
"""Remove the parts of the pre-migration sequence cache that nothing reads (20 November 2026).

Execution v2 never reads /home/saleh/webKinPred/media/sequence_info: attempts stage their
inputs from the blob store, and the cutover imported jobs, not caches. The only reader is the
OpenKinetics data portal, which opens seqmap.sqlite3 and <root>/<sequence_id>.npy for the four
roots in KEEP. Those are never touched. Deletion is throttled on production's I/O pressure so
the supervisors' 5 % host-pressure gate stays open.
"""
import json
import os
import sys
import time

ROOT = os.environ.get("CLEANUP_ROOT", "/home/saleh/webKinPred/media/sequence_info")
KEEP = {
    "seqmap.sqlite3", "seqmap.sqlite3-shm", "seqmap.sqlite3-wal",
    "esm2_layer_33/residue_vecs", "esmc_layer_32/residue_vecs",
    "prot_t5_last/residue_vecs", "pseq2sites_scores",
}
FILES = ["seqmap.corrupt.1790169998", "seqmap.sqlite3.bak", "seqmap.sqlite3.bak2",
         "seqmap_recovered.sqlite3"]
DIRECTORIES = [
    "esm1v", "omniesi_esm2", "iecata_prot_t5_residues", "esm2_layer_last_mean",
    "catrange_esmc", "esm1b_turnup",
    "esm2_layer_26", "esm2_layer_29", "esmc_layer_24", "prot_t5_layer_19",
    "esmc_layer_32/mean_vecs", "esmc_layer_32/weighted_vecs",
    "prot_t5_last/mean_vecs", "prot_t5_last/weighted_vecs",
    "catpred_esm2",
]
# Production's own load keeps full I/O pressure near 1 %; unthrottled unlinking raised the
# one-minute average to 5 % within minutes (its journal writes land after each burst).
LIMIT = float(os.environ.get("CLEANUP_MAX_IO_FULL", "2.0"))
LIMIT60 = float(os.environ.get("CLEANUP_MAX_IO_FULL60", "2.5"))
BATCH = 200
PAUSE = 0.25
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


def quiet():
    waited = 0
    while pressure() > LIMIT or pressure(2) > LIMIT60:
        time.sleep(5)
        waited += 5
    return waited


def checked(relative):
    for kept in KEEP:
        if relative == kept or relative.startswith(kept + "/"):
            raise RuntimeError("protected path: " + relative)
    path = os.path.join(ROOT, relative)
    if os.path.islink(path) or not os.path.realpath(path).startswith(ROOT + "/"):
        raise RuntimeError("unexpected path: " + relative)
    return path


def empty(path, totals):
    """Delete a directory tree bottom-up, keeping any directory that holds a protected root."""
    count = 0
    with os.scandir(path) as entries:
        for entry in entries:
            if entry.is_dir(follow_symlinks=False):
                relative = os.path.relpath(entry.path, ROOT)
                if any(k == relative or k.startswith(relative + "/") for k in KEEP):
                    continue
                empty(entry.path, totals)
                continue
            size = entry.stat(follow_symlinks=False).st_size
            os.unlink(entry.path)
            totals["files"] += 1
            totals["bytes"] += size
            count += 1
            if count % BATCH == 0:
                totals["paused_s"] += quiet()
                time.sleep(PAUSE)
    relative = os.path.relpath(path, ROOT)
    if not any(k.startswith(relative + "/") for k in KEEP):
        os.rmdir(path)


def main():
    totals = {"files": 0, "bytes": 0, "paused_s": 0}
    log(event="start", limit=LIMIT, pressure=pressure())
    for relative in FILES:
        path = checked(relative)
        if os.path.lexists(path):
            quiet()
            size = os.lstat(path).st_size
            os.unlink(path)
            totals["files"] += 1
            totals["bytes"] += size
            log(event="file", path=relative, bytes=size)
    for relative in DIRECTORIES:
        path = checked(relative)
        if not os.path.isdir(path):
            log(event="missing", path=relative)
            continue
        before = dict(totals)
        began = time.monotonic()
        empty(path, totals)
        log(event="directory", path=relative, files=totals["files"] - before["files"],
            bytes=totals["bytes"] - before["bytes"],
            seconds=round(time.monotonic() - began), paused_s=totals["paused_s"])
    log(event="done", **totals)


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        log(event="error", error=repr(error))
        raise
