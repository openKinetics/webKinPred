"""Bounded PostgreSQL and authoritative-media backup to an existing S3 destination.

Run as an operator service with CPU/memory limits. Credentials remain in the
configured AWS user's normal credential chain. A completion manifest is uploaded
only after every object has passed a full download/checksum verification.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import tempfile
from datetime import datetime, timezone
from pathlib import Path


def checksum(path):
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def aws_command(config):
    return ["runuser", "-u", config["aws_user"], "--", config["aws_binary"]]


def verify_object(config, uri, expected, size):
    digest, count = hashlib.sha256(), 0
    process = subprocess.Popen(
        aws_command(config) + ["s3", "cp", uri, "-", "--only-show-errors"], stdout=subprocess.PIPE
    )
    try:
        while block := process.stdout.read(1024**2):
            digest.update(block)
            count += len(block)
    finally:
        process.stdout.close()
    if process.wait() or digest.hexdigest() != expected or count != size:
        raise RuntimeError("Off-host backup verification failed")


def upload(config, stream, uri, expected_size):
    """Stream with bounded memory, preserving a digest for round-trip verification."""
    command = aws_command(config) + [
        "s3",
        "cp",
        "-",
        uri,
        "--only-show-errors",
        "--sse",
        "AES256",
        "--expected-size",
        str(max(1, expected_size)),
    ]
    process = subprocess.Popen(command, stdin=subprocess.PIPE)
    digest, count = hashlib.sha256(), 0
    try:
        while block := stream.read(1024**2):
            digest.update(block)
            count += len(block)
            process.stdin.write(block)
        process.stdin.close()
        if process.wait():
            raise RuntimeError("Backup object upload failed")
    except BaseException:
        process.kill()
        process.wait()
        raise
    result = {"sha256": digest.hexdigest(), "bytes": count}
    verify_object(config, uri, result["sha256"], count)
    return result


def dump_command(config, database):
    return [
        "docker",
        "exec",
        config["container"],
        "nice",
        "-n",
        "19",
        "pg_dump",
        "-U",
        config["role"],
        "-Fc",
        database,
    ]


# Only for the CLI's multipart part size (it needs one above 50 GB); a dump is
# verified by its streamed digest and length, not by this bound.
DUMP_SIZE_BOUND = 128 * 1024**3


def backup(config):
    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H-%M-%SZ")
    prefix = config["s3_prefix"].rstrip("/") + "/" + stamp
    scratch = Path(config["scratch"])
    scratch.mkdir(mode=0o700, parents=True, exist_ok=True)
    # Scratch now holds only the manifest. Dumps written there first no longer
    # fit on 14 November: the 25 GB live database dumped to about 5 GB beside
    # production's 15 GiB free-space floor, so every retry hit the guard, and
    # meanwhile held the host below the headroom that admits execution work.
    if shutil.disk_usage(scratch).free < config["minimum_free"]:
        raise RuntimeError("Insufficient backup scratch headroom")
    records = {}
    with tempfile.TemporaryDirectory(prefix="snapshot-", dir=scratch) as directory:
        for database in config["databases"]:
            name = database + ".dump"
            dump = subprocess.Popen(dump_command(config, database), stdout=subprocess.PIPE)
            try:
                records[name] = upload(config, dump.stdout, prefix + "/" + name, DUMP_SIZE_BOUND)
            except BaseException:
                dump.kill()
                dump.wait()
                raise
            finally:
                dump.stdout.close()
            # A dump that failed part-way was uploaded and verified as bytes, but
            # it is not a snapshot: fail before any completion manifest exists.
            if dump.wait():
                raise RuntimeError("Database dump failed")
        media = Path(config["media"])
        # Jobs contain original inputs and final/target results. Shared embedding
        # caches are regenerable and are not copied into the backup scratch disk.
        with tempfile.TemporaryFile() as errors:
            tar = subprocess.Popen(
                ["tar", "-C", str(media), "-cf", "-", "jobs"], stdout=subprocess.PIPE, stderr=errors
            )
            gzip = subprocess.Popen(["gzip", "-1"], stdin=tar.stdout, stdout=subprocess.PIPE)
            tar.stdout.close()
            try:
                estimate = sum(p.stat().st_size for p in (media / "jobs").rglob("*") if p.is_file())
                if estimate > 128 * 1024**3:
                    raise RuntimeError("Media backup exceeds its validated streaming envelope")
                records["jobs.tar.gz"] = upload(
                    config, gzip.stdout, prefix + "/jobs.tar.gz", estimate
                )
                if gzip.wait():
                    raise RuntimeError("Backup compression failed")
                code = tar.wait()
                errors.seek(0)
                warnings = errors.read().decode(errors="replace").splitlines()
                if code and not (
                    code == 1
                    and warnings
                    and all(line.endswith("file changed as we read it") for line in warnings)
                ):
                    raise RuntimeError("Authoritative media archive failed")
            finally:
                gzip.stdout.close()
                for process in (tar, gzip):
                    if process.poll() is None:
                        process.kill()
                        process.wait()
        manifest = {
            "format": "webkinpred-execution-v2",
            "complete": True,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "objects": records,
            "media_changed_during_archive": bool(code),
            "scope": (
                "Dedicated application/sequence/prediction databases; "
                "immutable job inputs and results"
            ),
        }
        path = Path(directory) / "manifest.json"
        path.write_text(json.dumps(manifest, indent=2))
        with path.open("rb") as stream:
            upload(config, stream, prefix + "/manifest.json", path.stat().st_size)
    return {"complete": True, "objects_verified": len(records) + 1, "prefix": prefix}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("config", type=Path)
    arguments = parser.parse_args()
    print(json.dumps(backup(json.loads(arguments.config.read_text()))))
