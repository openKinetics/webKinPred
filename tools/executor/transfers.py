"""Bounded immutable packs and batched readiness commits; no path extraction."""

from __future__ import annotations

import json
import shutil
import tarfile
import time
import urllib.error
import urllib.request

from api.execution.artifacts import HEX, safe_relative
from api.execution.errors import MissingEmbedding


def windows(rows, max_items=1024, max_bytes=256 * 1024**2):
    batch, size = [], 0
    for row in rows:
        if batch and (len(batch) >= max_items or size + row["size"] > max_bytes):
            yield batch
            batch, size = [], 0
        batch.append(row)
        size += row["size"]
    if batch:
        yield batch


def fetch(agent, host, rows):
    peer = agent.config["peers"][host]
    base = peer["url"].rstrip("/")
    headers = {"Authorization": "Bearer " + peer["token"], "Content-Type": "application/json"}
    wanted = {row["checksum"]: row["size"] for row in rows}
    if len(rows) == 1 and rows[0]["size"] > 256 * 1024**2:
        fallback = True
    else:
        fallback = False
        req = urllib.request.Request(
            base + "/blobs", data=json.dumps({"blobs": rows}).encode(), headers=headers
        )
        try:
            response = urllib.request.urlopen(req, timeout=30)
        except urllib.error.HTTPError as error:
            if error.code not in {404, 405, 501}:
                raise
            fallback = True  # Safe rolling deployment against an older gateway.
    if fallback:
        missing, transferred = [], 0
        for row in rows:
            req = urllib.request.Request(base + "/blobs/" + row["checksum"], headers=headers)
            try:
                with urllib.request.urlopen(req, timeout=30) as stream:
                    agent.store.publish(
                        stream,
                        row["checksum"],
                        row["size"],
                        minimum_free=agent.config["budgets"]["disk_headroom"],
                    )
                transferred += row["size"]
            except urllib.error.HTTPError as error:
                if error.code != 404:
                    raise
                missing.append(row["checksum"])
        return missing, transferred
    with response, tarfile.open(fileobj=response, mode="r|") as archive:
        first = archive.next()
        if (
            first is None
            or first.name != "manifest.json"
            or not first.isfile()
            or first.size > 1024**2
        ):
            raise ValueError("Invalid artifact pack manifest")
        manifest = json.loads(archive.extractfile(first).read())
        if not isinstance(manifest, dict):
            raise ValueError("Invalid artifact pack manifest")
        present, missing = manifest.get("present"), manifest.get("missing")
        if (
            not isinstance(present, dict)
            or not isinstance(missing, list)
            or not all(isinstance(k, str) for k in missing)
        ):
            raise ValueError("Invalid artifact pack manifest")
        if (
            manifest.get("version") != 1
            or set(present) & set(missing)
            or set(present) | set(missing) != set(wanted)
            or any(wanted.get(k) != v for k, v in present.items())
        ):
            raise ValueError("Artifact pack does not match request")
        seen, transferred = set(), 0
        # Streaming tar readers include the already-read first member in iteration.
        for member in archive:
            if member is first:
                continue
            if (
                not member.isfile()
                or member.name not in present
                or member.name in seen
                or member.size != wanted[member.name]
            ):
                raise ValueError("Unexpected artifact pack member")
            stream = archive.extractfile(member)
            agent.store.publish(
                stream,
                member.name,
                member.size,
                minimum_free=agent.config["budgets"]["disk_headroom"],
            )
            seen.add(member.name)
            transferred += member.size
        if seen != set(present):
            raise ValueError("Incomplete artifact pack")
        return missing, transferred


def materialize(agent, request, work):
    started = time.monotonic()
    artifacts = request["artifacts"]
    pending, updates, missing_ids = {}, {}, set()
    hits, transferred = 0, 0

    def record(artifact, host, ready):
        updates[(artifact["id"], host)] = {
            "artifact_id": artifact["id"],
            "checksum": artifact["checksum"],
            "size": artifact["size"],
            "source_host": host,
            "ready": ready,
        }

    for artifact in artifacts:
        checksum = artifact["checksum"]
        if not HEX.fullmatch(checksum):
            missing_ids.add(artifact["id"])
            continue
        blob = agent.store.path(checksum)
        if blob.exists() and (
            blob.stat().st_size != artifact["size"] or not agent.store.verified(blob, checksum)
        ):
            blob.unlink()
        if blob.exists():
            hits += 1
            if agent.host not in artifact["hosts"]:
                record(artifact, agent.host, True)
            continue
        if agent.host in artifact["hosts"]:
            record(artifact, agent.host, False)
        entry = pending.setdefault(
            checksum,
            {
                "checksum": checksum,
                "size": artifact["size"],
                "artifacts": [],
                "hosts": set(),
                "confirmed_missing": False,
            },
        )
        if entry["size"] != artifact["size"]:
            raise ValueError("Conflicting immutable artifact sizes")
        entry["artifacts"].append(artifact)
        entry["hosts"].update(
            host
            for host in artifact["hosts"]
            if host != agent.host and host in agent.config.get("peers", {})
        )
        entry["confirmed_missing"] |= agent.host in artifact["hosts"] or not artifact["hosts"]
    for host in sorted({host for entry in pending.values() for host in entry["hosts"]}):
        rows = [
            {"checksum": checksum, "size": entry["size"]}
            for checksum, entry in pending.items()
            if host in entry["hosts"] and not agent.store.path(checksum).exists()
        ]
        for batch in windows(rows):
            try:
                absent, count = fetch(agent, host, batch)
                transferred += count
            except (OSError, ValueError, tarfile.TarError):
                # Already published members are verified and reusable. A transport
                # error does not prove any remote catalog location is missing.
                continue
            for checksum in absent:
                entry = pending[checksum]
                entry["confirmed_missing"] = True
                for artifact in entry["artifacts"]:
                    if host in artifact["hosts"]:
                        record(artifact, host, False)
    network_missing = False
    for checksum, entry in pending.items():
        if agent.store.path(checksum).exists():
            for artifact in entry["artifacts"]:
                record(artifact, agent.host, True)
        elif entry["confirmed_missing"]:
            missing_ids.update(a["id"] for a in entry["artifacts"])
        else:
            network_missing = True
    rows = list(updates.values())
    for offset in range(0, len(rows), 1024):
        batch = rows[offset : offset + 1024]
        if request.get("artifact_protocol", 1) >= 2:
            agent.request(
                "artifact-locations", {"token": request["token"], "locations": batch}, request["id"]
            )
        else:
            for row in batch:
                artifact = next(a for a in artifacts if a["id"] == row["artifact_id"])
                agent.artifact_location(request, artifact, row["source_host"], row["ready"])
    if missing_ids:
        error = MissingEmbedding("Required artifacts must be rescheduled")
        error.missing_artifacts = sorted(missing_ids)
        raise error
    if network_missing:
        raise ConnectionError("Required artifact transfer unavailable")
    for artifact in artifacts:
        destination = work / "media" / safe_relative(artifact["path"])
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(agent.store.path(artifact["checksum"]), destination)
    from tools.executor.checkpoints import restore

    restore(agent.root, request, work)
    return {
        "transfer_seconds": time.monotonic() - started,
        "transferred_bytes": transferred,
        "local_cache_hits": hits,
        "artifact_requests": len(artifacts),
    }
