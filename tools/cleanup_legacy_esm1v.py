#!/usr/bin/env python3
"""Conservative, manual cleanup of pre-migration *flat* ESM1v residue files.

Run on the production HOST as root, not inside a container. Standard library only.
Preview is the default; see docs/legacy-esm1v-cleanup.md before using --apply.
"""

from __future__ import annotations

import argparse
import ast
import contextlib
import fcntl
import json
import os
import re
import select
import stat
import struct
import subprocess
import sys
import time
import uuid
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

PREFIX = "sequence_info/esm1v/"
CUTOFF = datetime(2026, 10, 29, tzinfo=timezone.utc).timestamp()
NAME = re.compile(r"[0-9a-f]{12}(?:_[1-9][0-9]*)?\.npy\Z")
GRAPH_LOCK = 481726392
INDEX_NAME = "wk_legacy_esm1v_path"

# Operator-owned, narrowly scoped maintenance index. Its concurrent build runs
# outside the graph lock and never changes job/artifact records or file content.
PREPARE_INDEX = r"""
import json, os
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "webKinPred.settings")
import django
django.setup()
from django.db import connection
from api.execution.models import Artifact
if connection.vendor != "postgresql":
    raise RuntimeError("PostgreSQL is required")
name = "wk_legacy_esm1v_path"
with connection.cursor() as cursor:
    # The two-int advisory namespace is separate from scheduler bigint locks.
    cursor.execute("SELECT pg_try_advisory_lock(%s, %s)", [1464557617, 1163087153])
    if not cursor.fetchone()[0]:
        raise RuntimeError("Another cleanup index setup is running")
try:
    with connection.cursor() as cursor:
        cursor.execute("SET statement_timeout='600000ms'")
        cursor.execute("SET lock_timeout='30000ms'")
        cursor.execute("SET maintenance_work_mem='32MB'")
        cursor.execute("SET max_parallel_maintenance_workers=0")
        cursor.execute("SELECT indisvalid, indrelid = %s::regclass, "
                       "pg_get_expr(indexprs, indrelid), pg_get_expr(indpred, indrelid) "
                       "FROM pg_index "
                       "WHERE indexrelid=to_regclass(%s)", [Artifact._meta.db_table, name])
        existing = cursor.fetchone()
        if existing:
            valid, right_table, expression, predicate = existing
            expected = "((identity ->> 'path'::text) ~~ 'sequence\\_info/esm1v/%'::text)"
            if (not right_table or expression != "(identity ->> 'path'::text)"
                    or predicate != expected):
                raise RuntimeError("Unexpected cleanup index definition; inspect before retrying")
            if not valid:
                # A canceled concurrent build can leave an invalid index. The
                # dedicated advisory lock fences another setup; validate its
                # exact definition before removing only this maintenance index.
                cursor.execute("DROP INDEX CONCURRENTLY " + name)
                existing = None
        if not existing:
            print(json.dumps({"event": "building_esm1v_lookup_index", "concurrently": True}),
                  flush=True)
            table = connection.ops.quote_name(Artifact._meta.db_table)
            cursor.execute("CREATE INDEX CONCURRENTLY " + name + " ON " + table +
                           " ((identity ->> 'path') text_pattern_ops) "
                           "WHERE (identity ->> 'path') LIKE 'sequence\\_info/esm1v/%'")
        cursor.execute("SELECT indisvalid, pg_relation_size(indexrelid) FROM pg_index "
                       "WHERE indexrelid=to_regclass(%s)", [name])
        valid, size = cursor.fetchone()
        if not valid:
            raise RuntimeError("Cleanup lookup index did not become valid")
        print(json.dumps({"event": "esm1v_lookup_index_ready", "bytes": size}), flush=True)
finally:
    with connection.cursor() as cursor:
        cursor.execute("SELECT pg_advisory_unlock(%s, %s)", [1464557617, 1163087153])
"""

# Executed using the backend's existing settings/credentials. No DB writes and no
# Redis operations. The session lock fences graph publication for at most 20s;
# it does not stop active attempts, heartbeats, or existing-stage dispatch.
DATABASE_GUARD = r"""
import json, os, select, sys
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "webKinPred.settings")
import django
django.setup()
from django.conf import settings
from django.db import connection
from django.db.models import Q
from django.utils import timezone
from pathlib import Path
from api.models import Job
from api.execution.models import (
    Artifact, ArtifactDependency, ExecutionAttempt, ExecutorHost, WorkStage,
)

prefix = "sequence_info/esm1v/"
if connection.vendor != "postgresql" or not settings.EXECUTION_V2_ENABLED:
    raise RuntimeError("Requires PostgreSQL and EXECUTION_V2_ENABLED")
with connection.cursor() as cursor:
    cursor.execute("SET statement_timeout = '5000ms'")
    cursor.execute("SET jit = off")
    cursor.execute("SET max_parallel_workers_per_gather = 0")
    cursor.execute("SELECT indisvalid, indrelid = %s::regclass FROM pg_index "
                   "WHERE indexrelid=to_regclass(%s)",
                   [Artifact._meta.db_table, "wk_legacy_esm1v_path"])
    if cursor.fetchone() != (True, True):
        raise RuntimeError("Cleanup lookup index is missing/invalid; run --prepare-index first")
    cursor.execute("SELECT pg_try_advisory_lock(%s)", [481726392])
    if not cursor.fetchone()[0]:
        raise RuntimeError("Graph is busy; retry later")
try:
    now = timezone.now()
    hosts = list(ExecutorHost.objects.filter(pk__in=["production", "gpu"]))
    if len(hosts) != 2:
        raise RuntimeError("Both registered executor hosts are required")
    for host in hosts:
        age = (now - host.heartbeat_at).total_seconds() if host.heartbeat_at else float("inf")
        if not host.reconciled or not -5 <= age <= 30:
            raise RuntimeError("Executor telemetry is stale/unreconciled: " + host.pk)
        if host.telemetry.get("legacy_consumers", None) != []:
            raise RuntimeError("Legacy consumer state is unsafe/unknown: " + host.pk)
    if Job.objects.filter(status__in=["Pending", "Processing"]).exclude(
        execution__policy_version="v2-gpu-only"
    ).exists():
        raise RuntimeError("An unfinished job has no compatible managed execution")

    # One bounded scan per relation, never one query per file or per queued job.
    # Include held/canceled-with-live-attempt work, not just currently running work.
    live = ExecutionAttempt.objects.filter(state__in=["leased", "running", "uncertain"])
    refs = ArtifactDependency.objects.filter(path__startswith=prefix).filter(
        ~Q(stage__state__in=["completed", "canceled"]) | Q(stage_id__in=live.values("stage_id"))
    )
    reference_rows = list(refs.values_list("path", flat=True)[:200001])
    if len(reference_rows) > 200000:
        raise RuntimeError("Dependency snapshot exceeds safety bound")
    protected = set(reference_rows)
    # Avoid a catalogue-wide producer join/OR plan. Read narrow rows once and
    # use the unfinished-stage index for producer protection.
    active_rows = list(WorkStage.objects.exclude(state__in=["completed", "canceled"])
                       .values_list("pk", flat=True)[:200001])
    if len(active_rows) > 200000:
        raise RuntimeError("Producer snapshot exceeds safety bound")
    active = set(active_rows) | set(live.values_list("stage_id", flat=True))
    artifact_rows = list(Artifact.objects.filter(identity__path__startswith=prefix).values_list(
        "identity__path", "retention", "regenerable", "producer_id"
    )[:200001])
    if len(artifact_rows) > 200000:
        raise RuntimeError("Artifact snapshot exceeds safety bound")
    protected.update(
        path for path, retention, regenerable, producer in artifact_rows
        if retention in {"protected", "permanent"} or not regenerable or producer in active
    )
    if len(protected) > 200000:
        raise RuntimeError("Protection snapshot exceeds safety bound")
    root = Path(settings.MEDIA_ROOT) / "sequence_info" / "esm1v"
    info = root.stat()
    print(json.dumps({"protected": sorted(protected), "device": info.st_dev,
                      "inode": info.st_ino}), flush=True)
    if not select.select([sys.stdin], [], [], 20)[0] or sys.stdin.readline() != "release\n":
        raise RuntimeError("Cleanup guard expired")
finally:
    with connection.cursor() as cursor:
        cursor.execute("SELECT pg_advisory_unlock(%s)", [481726392])
"""


def docker(*args):
    return subprocess.check_output(["docker", *args], text=True, timeout=30)


def database_entry(source):
    """Compact errors rather than a multi-kilobyte production traceback."""
    return (
        "import json,sys\ntry:\n exec(" + repr(source) + ")\n"
        "except Exception as error:\n"
        " cause = getattr(error, '__cause__', None)\n"
        " state = getattr(cause, 'sqlstate', None) or getattr(cause, 'pgcode', None)\n"
        " message = str(error) if isinstance(error, RuntimeError) else type(error).__name__\n"
        " if state: message += ' (SQLSTATE ' + state + ')'\n"
        " print(json.dumps({'error':message,'files_preserved':True}), flush=True)\n"
        " sys.exit(1)\n"
    )


def identity(info):
    return [info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns, info.st_ctime_ns]


def residue_header(handle, size):
    """Read at most 8KiB, never unpickle or load the residue matrix."""
    magic = handle.read(8)
    if magic not in (b"\x93NUMPY\x01\x00", b"\x93NUMPY\x02\x00", b"\x93NUMPY\x03\x00"):
        return False
    width = 2 if magic[6] == 1 else 4
    raw = handle.read(width)
    if len(raw) != width:
        return False
    length = struct.unpack("<H" if width == 2 else "<I", raw)[0]
    if not 0 < length <= 8192:
        return False
    try:
        header = ast.literal_eval(handle.read(length).decode("utf-8"))
        shape = header["shape"]
        return (
            header["descr"] == "<f4"
            and header["fortran_order"] is False
            and isinstance(shape, tuple)
            and len(shape) == 2
            and all(type(value) is int for value in shape)
            and shape[0] > 1
            and shape[1] == 1280
            and size == 8 + width + length + shape[0] * shape[1] * 4
        )
    except (ValueError, SyntaxError, UnicodeError, KeyError, TypeError):
        return False


def inspect_file(directory_fd, name, protected, opened, keep):
    row = {"name": name, "reason": "unknown_name"}
    if not NAME.fullmatch(name):
        return row
    try:
        before = os.stat(name, dir_fd=directory_fd, follow_symlinks=False)
        row.update(identity=identity(before), bytes=before.st_size)
        if not stat.S_ISREG(before.st_mode) or before.st_nlink != 1:
            row["reason"] = "not_single_link_regular_file"
        elif before.st_mtime >= CUTOFF:
            row["reason"] = "post_migration"
        elif name in keep:
            row["reason"] = "operator_keep"
        elif PREFIX + name in protected:
            row["reason"] = "database_reference"
        elif (before.st_dev, before.st_ino) in opened:
            row["reason"] = "open_or_mapped"
        else:
            fd = os.open(name, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK, dir_fd=directory_fd)
            with os.fdopen(fd, "rb") as handle:
                valid = residue_header(handle, before.st_size)
                unchanged = identity(os.fstat(handle.fileno())) == identity(before)
            after = os.stat(name, dir_fd=directory_fd, follow_symlinks=False)
            if not unchanged or identity(after) != identity(before):
                row["reason"] = "changed"
            else:
                row["reason"] = "eligible" if valid else "not_verified_residue_matrix"
    except FileNotFoundError:
        row["reason"] = "vanished"
    return row


def open_inodes(proc_root=Path("/proc")):
    """Host-wide open FD + mmap audit. Permission failures abort the cleanup."""
    result = set()
    for process in proc_root.iterdir():
        if not process.name.isdigit() or int(process.name) == os.getpid():
            continue
        try:
            for descriptor in (process / "fd").iterdir():
                try:
                    info = descriptor.stat()
                    result.add((info.st_dev, info.st_ino))
                except FileNotFoundError:
                    pass  # Closed FD or exited process.
            with (process / "maps").open() as mappings:
                for line in mappings:
                    parts = line.split(None, 5)
                    if len(parts) >= 5 and parts[4] != "0":
                        major, minor = (int(value, 16) for value in parts[3].split(":"))
                        result.add((os.makedev(major, minor), int(parts[4])))
        except (FileNotFoundError, ProcessLookupError):
            pass
    return result


def check_containers(root, backend):
    ids = docker("ps", "-q").split()
    containers = json.loads(docker("inspect", *ids)) if ids else []
    trusted = {
        backend,
        "webkinpred-execution-controller-1",
        "webkinpred-execution-control-worker-1",
        "webkinpred-execution-agent-1",
        "webkinpred-execution-artifacts-1",
        "webkinpred-execution-cleanup-1",
        "webkinpred-about-stats-worker-1",
    }
    for container in containers:
        name = container["Name"].lstrip("/")
        labels = container["Config"].get("Labels") or {}
        service = labels.get("com.docker.compose.service", "")
        if labels.get("com.docker.compose.project") == "webkinpred" and service in {
            "celery",
            "celery-cache",
            "celery-beat",
        }:
            raise RuntimeError("Legacy worker is running: " + name)
        for mount in container.get("Mounts", []):
            source = Path(mount["Source"]).resolve()
            if source == root or source in root.parents or root in source.parents:
                if name not in trusted:
                    raise RuntimeError("Unrecognized container can access legacy cache: " + name)
    if backend not in {c["Name"].lstrip("/") for c in containers}:
        raise RuntimeError("Backend container is not running")


@contextlib.contextmanager
def database_guard(backend, root_info):
    with subprocess.Popen(
        ["docker", "exec", "-i", backend, "python", "-c", database_entry(DATABASE_GUARD)],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    ) as process:
        try:
            if not select.select([process.stdout], [], [], 30)[0]:
                raise RuntimeError("Database safety snapshot timed out")
            line = process.stdout.readline()
            if not line:
                raise RuntimeError(
                    "Database safety snapshot refused: " + process.stderr.read()[-2000:]
                )
            snapshot = json.loads(line)
            if "error" in snapshot:
                raise RuntimeError("Database safety snapshot refused: " + snapshot["error"])
            if (snapshot["device"], snapshot["inode"]) != (root_info.st_dev, root_info.st_ino):
                raise RuntimeError("Backend MEDIA_ROOT is a different filesystem/directory")
            # Leave five seconds for graceful release before the DB watchdog.
            yield set(snapshot["protected"]), time.monotonic() + 15, process
            process.stdin.write("release\n")
            process.stdin.flush()
            process.stdin.close()
            if process.wait(timeout=5) != 0:
                raise RuntimeError("Database guard ended unexpectedly")
        finally:
            if process.poll() is None:
                process.kill()
                process.wait()


def remove_unchanged(directory_fd, quarantine_fd, row):
    """Unlink only the exact regular file inspected in this pass; never recurse."""
    current = os.stat(row["name"], dir_fd=directory_fd, follow_symlinks=False)
    if (
        not stat.S_ISREG(current.st_mode)
        or current.st_nlink != 1
        or identity(current) != row["identity"]
    ):
        raise RuntimeError("Candidate changed; stopped before deleting " + row["name"])
    # Claim the name atomically, then revalidate inside our private directory.
    # A stat+unlink alone could delete a replacement installed between syscalls.
    name = row["name"]
    os.rename(name, name, src_dir_fd=directory_fd, dst_dir_fd=quarantine_fd)
    moved = os.stat(name, dir_fd=quarantine_fd, follow_symlinks=False)
    # rename may update ctime; the remaining identity fields must still match.
    if (
        not stat.S_ISREG(moved.st_mode)
        or moved.st_nlink != 1
        or identity(moved)[:4] != row["identity"][:4]
    ):
        # Restore without overwriting any newly-created source. If occupied,
        # leave the unexpected file in quarantine; never recursively remove it.
        try:
            os.link(
                name, name, src_dir_fd=quarantine_fd, dst_dir_fd=directory_fd, follow_symlinks=False
            )
            os.unlink(name, dir_fd=quarantine_fd)
        except FileExistsError:
            pass
        raise RuntimeError("Replacement preserved; stopped before deleting " + name)
    os.unlink(name, dir_fd=quarantine_fd)


@contextlib.contextmanager
def quarantine(directory_fd):
    name = ".cleanup-" + uuid.uuid4().hex
    os.mkdir(name, mode=0o700, dir_fd=directory_fd)
    descriptor = os.open(name, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=directory_fd)
    try:
        yield descriptor
    finally:
        os.close(descriptor)
        try:
            os.rmdir(name, dir_fd=directory_fd)  # Empty directories only.
        except OSError:
            print(json.dumps({"preserved_quarantine": name}), file=sys.stderr)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, default=Path("/home/saleh/webKinPred"))
    parser.add_argument("--backend", default="webkinpred-backend-1")
    parser.add_argument(
        "--prepare-index",
        action="store_true",
        help="Build the small PostgreSQL lookup index concurrently; never deletes files",
    )
    parser.add_argument(
        "--apply", action="store_true", help="Delete eligible files (default: preview)"
    )
    parser.add_argument(
        "--limit", type=int, default=1000, help="Maximum deletions in one invocation"
    )
    parser.add_argument(
        "--report", type=Path, help="Create a new JSONL audit file; required with --apply"
    )
    parser.add_argument(
        "--keep-file", type=Path, help="Additional filenames to preserve, one per line"
    )
    args = parser.parse_args(argv)
    if os.geteuid() != 0:
        parser.error("Run on the production host using sudo so /proc can be audited completely")
    if args.prepare_index:
        if args.apply:
            parser.error("--prepare-index and --apply must be separate invocations")
        result = subprocess.run(
            ["docker", "exec", args.backend, "python", "-c", database_entry(PREPARE_INDEX)],
            timeout=620,
        )
        if result.returncode:
            raise RuntimeError("Index setup refused; see preceding JSON error. No files deleted")
        return
    if not 1 <= args.limit <= 10000:
        parser.error("--limit must be between 1 and 10000")
    if args.apply and not args.report:
        parser.error("--apply requires --report for an audit trail")
    root = args.repo.absolute() / "media" / "sequence_info" / "esm1v"
    if root.resolve(strict=True) != root:
        parser.error("Cache directory or a parent is a symlink; refusing")
    keep = set()
    if args.keep_file:
        keep = {line.strip() for line in args.keep_file.read_text().splitlines() if line.strip()}
        if any(not NAME.fullmatch(name) for name in keep):
            parser.error("--keep-file must contain only ESM1v .npy filenames")
    if args.report and (args.report.resolve() == root or root in args.report.resolve().parents):
        parser.error("Audit report must be outside the cache directory")

    # This file lock serializes invocations; its parent must already exist.
    with Path("/run/lock/webkinpred-legacy-esm1v.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        blockers = []
        try:
            check_containers(root, args.backend)
        except RuntimeError as error:
            if args.apply:
                raise
            # A preview can still report candidates, but must expose why an
            # apply would refuse. It never overrides the admission safeguards.
            blockers.append(str(error))
        fd = os.open(root, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
        try:
            root_info = os.fstat(fd)
            with database_guard(args.backend, root_info) as (protected, deadline, guard):
                opened = open_inodes()
                if (root_info.st_dev, root_info.st_ino) in opened:
                    raise RuntimeError("Another process has the cache directory open")
                rows = [
                    inspect_file(fd, name, protected, opened, keep)
                    for name in sorted(os.listdir(fd))
                ]
                eligible = [row for row in rows if row["reason"] == "eligible"]
                selected = eligible[: args.limit]
                deleted = 0
                deleted_bytes = 0
                report = args.report.open("x") if args.report else contextlib.nullcontext(None)
                with report as audit:
                    if audit:
                        audit.write(
                            json.dumps(
                                {
                                    "event": "scan",
                                    "apply": args.apply,
                                    "root": str(root),
                                    "at": datetime.now(timezone.utc).isoformat(),
                                }
                            )
                            + "\n"
                        )
                        for row in rows:
                            audit.write(json.dumps(row) + "\n")
                        audit.flush()
                        os.fsync(audit.fileno())
                    if args.apply:
                        # Recheck container mounts and readers immediately before deletion.
                        check_containers(root, args.backend)
                        opened = open_inodes()
                        if (root_info.st_dev, root_info.st_ino) in opened:
                            raise RuntimeError("Another process opened the cache directory")
                        with quarantine(fd) as quarantine_fd:
                            for row in selected:
                                if time.monotonic() >= deadline or guard.poll() is not None:
                                    raise RuntimeError(
                                        "Safety window expired; remaining files preserved"
                                    )
                                if (row["identity"][0], row["identity"][1]) in opened:
                                    continue
                                # Refuse path replacement even though rename uses a pinned dirfd.
                                if identity(root.stat())[:2] != identity(root_info)[:2]:
                                    raise RuntimeError("Cache directory changed")
                                remove_unchanged(fd, quarantine_fd, row)
                                deleted += 1
                                deleted_bytes += row["bytes"]
                                audit.write(
                                    json.dumps(
                                        {
                                            "event": "deleted",
                                            "name": row["name"],
                                            "bytes": row["bytes"],
                                        }
                                    )
                                    + "\n"
                                )
                        os.fsync(fd)
                    summary = {
                        "apply": args.apply,
                        "files": len(rows),
                        "reasons": dict(Counter(row["reason"] for row in rows)),
                        "eligible_bytes": sum(row["bytes"] for row in eligible),
                        "selected": len(selected),
                        "deleted": deleted,
                        "deleted_bytes": deleted_bytes,
                        "deletion_blockers": blockers,
                    }
                    if audit:
                        audit.write(json.dumps({"event": "summary", **summary}) + "\n")
                        audit.flush()
                        os.fsync(audit.fileno())
                    print(json.dumps(summary), flush=True)
        finally:
            os.close(fd)


if __name__ == "__main__":
    try:
        main()
    except (OSError, RuntimeError, ValueError, subprocess.SubprocessError) as error:
        print(json.dumps({"error": str(error), "stopped": True}), file=sys.stderr)
        sys.exit(1)
