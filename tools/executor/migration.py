"""Non-destructive queue export and reconciliation. Does not purge or revoke tasks."""

from __future__ import annotations

import base64
import hashlib
import json
import sqlite3
from pathlib import Path

from api.execution.artifacts import atomic_json, sha256_file

SUPPORTED = {
    "api.tasks.run_prediction",
    "api.tasks.run_both_prediction",
    "api.tasks.run_multi_prediction",
    "api.tasks.run_recon_xkg_cache_prediction",
}


def decode_message(raw):
    message = json.loads(raw)
    if not isinstance(message, dict):
        return None
    task = message.get("headers", {}).get("task")
    if task not in SUPPORTED:
        return None
    body = message["body"]
    if message.get("properties", {}).get("body_encoding") == "base64":
        body = base64.b64decode(body, validate=True)
    if message.get("content-type", "application/json") != "application/json" or message.get(
        "headers", {}
    ).get("compression"):
        raise ValueError("Unsupported task serialization; preserve envelope for explicit review")
    args, kwargs, _ = json.loads(body)
    return normalize_request(task, args, kwargs, message["headers"].get("id"))


def normalize_request(task, args, kwargs, task_id=None):
    import ast

    if isinstance(args, str):
        args = ast.literal_eval(args)
    if isinstance(kwargs, str):
        kwargs = ast.literal_eval(kwargs)
    if task not in SUPPORTED or not isinstance(args, (list, tuple)) or not isinstance(kwargs, dict):
        raise ValueError("Unsupported legacy task request")

    def value(name, index, default=None):
        return kwargs.get(name, args[index] if len(args) > index else default)

    public_id = value("public_id", 0)
    if not isinstance(public_id, str) or not public_id.isalnum():
        raise ValueError("Invalid legacy public ID")
    common = {
        "public_id": public_id,
        "task_id": task_id,
        "canonicalize_substrates": value("canonicalize_substrates", 4, True),
        "include_similarity_columns": value("include_similarity_columns", 5, True),
        "disable_gpu_precompute": value("disable_gpu_precompute", 6, False),
    }
    if task == "api.tasks.run_prediction":
        target = value("target", 2)
        common.update(
            targets=[target],
            methods={target: value("method_key", 1)},
            experimental_results={target: value("experimental_results", 3, [])},
        )
    elif task == "api.tasks.run_both_prediction":
        common.update(
            targets=["kcat", "Km"],
            methods={"kcat": value("kcat_key", 1), "Km": value("km_key", 2)},
            experimental_results=value("experimental_results", 3, {}),
        )
    else:
        common.update(
            targets=value("targets", 1),
            methods=value("methods", 2),
            experimental_results=value("experimental_results", 3, {}),
            recon_xkg=task.endswith("cache_prediction") or value("recon_xkg", 7, False),
        )
    targets, methods = common["targets"], common["methods"]
    if (
        not isinstance(targets, list)
        or not targets
        or any(target not in {"kcat", "Km", "kcat/Km"} for target in targets)
        or not isinstance(methods, dict)
        or any(
            not isinstance(methods.get(target), str) or not methods[target] for target in targets
        )
    ):
        raise ValueError("Incomplete historical method/target specification")
    if any(
        not isinstance(common[key], bool)
        for key in (
            "canonicalize_substrates",
            "include_similarity_columns",
            "disable_gpu_precompute",
        )
    ):
        raise ValueError("Historical execution options must be booleans")
    if common.get("experimental_results") is None:
        common["experimental_results"] = {}
    return common


def export_broker(redis, destination: Path):
    destination.mkdir(parents=True, exist_ok=True)
    records = []
    for key in redis.scan_iter():
        kind = redis.type(key)
        if kind != b"list" and kind != "list":
            continue
        # Export envelopes without restricting priority suffixes or relying on queue length alone.
        for offset in range(0, redis.llen(key), 100):
            for index, raw in enumerate(redis.lrange(key, offset, offset + 99)):
                try:
                    decoded = decode_message(raw)
                except (ValueError, KeyError, TypeError):
                    decoded = None
                if decoded is None and not key.startswith(b"webkinpred"):
                    continue
                ident = hashlib.sha256(raw).hexdigest()
                (destination / (ident + ".envelope")).write_bytes(raw)
                records.append(
                    {
                        "source": base64.b64encode(key).decode(),
                        "index": offset + index,
                        "envelope": ident,
                        "request": decoded,
                    }
                )
    # Capture unacknowledged deliveries independently of visible queue lists.
    for key in redis.scan_iter(match="*unacked*"):
        value = redis.dump(key)
        if value:
            ident = hashlib.sha256(value).hexdigest()
            (destination / (ident + ".redis-dump")).write_bytes(value)
            records.append(
                {
                    "unacked_key": base64.b64encode(key).decode(),
                    "dump": ident,
                    "ttl_ms": redis.pttl(key),
                }
            )
            if redis.type(key) in {b"hash", "hash"}:
                for delivery, encoded in redis.hgetall(key).items():
                    try:
                        envelope, exchange, routing = json.loads(encoded)
                        request = decode_message(json.dumps(envelope))
                    except (ValueError, TypeError, KeyError):
                        request = None
                    records.append(
                        {
                            "unacked_delivery": base64.b64encode(delivery).decode(),
                            "request": request,
                        }
                    )
    atomic_json(destination / "broker.json", records)
    return records


def reconcile(jobs, requests):
    by_job = {}
    errors = []
    for request in requests:
        if request is None:
            errors.append("Undecoded source request")
            continue
        by_job.setdefault(request["public_id"], []).append(request)
    unfinished = {j["public_id"]: j for j in jobs if j["status"] in {"Pending", "Processing"}}
    specifications = {}
    dispositions = []
    for ident, entries in by_job.items():
        canonical = [{k: v for k, v in entry.items() if k != "task_id"} for entry in entries]
        hashes = {json.dumps(entry, sort_keys=True, separators=(",", ":")) for entry in canonical}
        if ident not in unfinished:
            dispositions.append(
                {
                    "public_id": ident,
                    "disposition": "terminal_or_orphan",
                    "task_ids": [x.get("task_id") for x in entries],
                }
            )
        elif len(hashes) != 1:
            errors.append("Conflicting requests for " + ident)
        else:
            specifications[ident] = canonical[0]
            dispositions.append(
                {
                    "public_id": ident,
                    "disposition": "restart"
                    if unfinished[ident]["status"] == "Processing"
                    else "requeue",
                    "duplicate_deliveries": len(entries) - 1,
                    "task_ids": [x.get("task_id") for x in entries],
                }
            )
    errors.extend(
        "Missing request for " + ident for ident in unfinished if ident not in specifications
    )
    return {
        "ready": not errors,
        "unfinished_count": len(unfinished),
        "specifications": specifications,
        "dispositions": dispositions,
        "errors": errors,
    }


def backup_sqlite(source: Path, destination: Path):
    with sqlite3.connect(f"file:{source.resolve()}?mode=ro", uri=True) as incoming:
        with sqlite3.connect(destination) as outgoing:
            incoming.backup(outgoing)
            if outgoing.execute("PRAGMA integrity_check").fetchone()[0] != "ok":
                raise ValueError("SQLite backup integrity check failed")
    return {
        "source": str(source),
        "destination": str(destination),
        "sha256": sha256_file(destination),
    }
