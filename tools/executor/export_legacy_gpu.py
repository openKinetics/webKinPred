"""Export legacy in-memory GPU requests through status and preserved sequence IDs.

Run only after stopping callers/dispatch. Every legacy submission creates its
UUID log before entering the in-memory queue; status exposes its exact step work.
Reconstruct sequence text from the frozen sequence registry, never current job
defaults. A nonzero active/queued count without a matching export blocks cutover.
"""

import argparse
import json
import os
import sqlite3
import urllib.error
import urllib.request
import uuid
from pathlib import Path

from api.execution.artifacts import atomic_json


def export(endpoint, token, log_directory, sequence_database, destination):
    destination.mkdir(parents=True, exist_ok=False, mode=0o700)

    def get(path):
        request = urllib.request.Request(
            endpoint.rstrip("/") + path, headers={"Authorization": "Bearer " + token}
        )
        with urllib.request.urlopen(request, timeout=10) as response:
            return json.load(response)

    before = get("/health")
    statuses, requests, absent = [], [], []
    # Open a consistent SQLite backup supplied by the maintenance procedure.
    with sqlite3.connect(sequence_database.resolve().as_uri() + "?mode=ro", uri=True) as db:
        for path in sorted(log_directory.glob("*.log")):
            try:
                uuid.UUID(path.stem)
                ident = path.stem  # Legacy IDs use hex without UUID hyphens.
            except ValueError:
                continue
            try:
                status = get("/embed/jobs/" + ident)
            except urllib.error.HTTPError as error:
                if error.code != 404:
                    raise
                absent.append(ident)
                continue
            statuses.append(status)
            if status["status"] not in {"queued", "running"}:
                continue
            sequences = {}
            for sid in dict.fromkeys(sid for ids in status["step_work"].values() for sid in ids):
                row = db.execute("SELECT seq FROM sequences WHERE id = ?", [sid]).fetchone()
                if row is None:
                    raise ValueError("Missing historical sequence identity: " + sid)
                sequences[sid] = row[0]
            requests.append(
                {
                    "legacy_job_id": ident,
                    "status": status["status"],
                    "request": {
                        "method_key": status["method_key"],
                        "target": status["target"],
                        "profile": status.get("profile"),
                        "step_work": status["step_work"],
                        "seq_id_to_seq": sequences,
                    },
                }
            )
    after = get("/health")
    expected = {"queued": after["queued_jobs"], "running": after["active_jobs"]}
    observed = {name: sum(row["status"] == name for row in requests) for name in expected}
    consistent = (
        all(before[key] == after[key] for key in ("queued_jobs", "active_jobs"))
        and expected == observed
    )
    result = {
        "ready": consistent,
        "health_before": before,
        "health_after": after,
        "statuses": statuses,
        "unfinished_requests": requests,
        "historical_logs_without_live_state": absent,
        "counts": observed,
    }
    atomic_json(destination / "gpu-requests.json", result)
    if not consistent:
        raise RuntimeError("GPU requests changed or logs are incomplete; repeat frozen export")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", required=True)
    parser.add_argument("--log-directory", required=True, type=Path)
    parser.add_argument("--sequence-database", required=True, type=Path)
    parser.add_argument("destination", type=Path)
    args = parser.parse_args()
    result = export(
        args.url,
        os.environ["GPU_EMBED_SERVICE_TOKEN"],
        args.log_directory,
        args.sequence_database,
        args.destination,
    )
    print(json.dumps({"ready": result["ready"], "counts": result["counts"]}))
