"""Quiescent backup inventory, integrity checks, and restore rehearsal helpers."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sqlite3
from pathlib import Path

from api.execution.artifacts import atomic_json, safe_relative, sha256_file

from tools.executor.migration import backup_sqlite


def verify(directory):
    manifest = json.loads((directory / "backup.json").read_text())
    for entry in manifest["files"]:
        path = directory / safe_relative(entry["path"])
        if (
            not path.is_file()
            or path.stat().st_size != entry["size"]
            or sha256_file(path) != entry["sha256"]
        ):
            raise ValueError("Backup mismatch: " + entry["path"])
        if entry.get("sqlite"):
            with sqlite3.connect(f"file:{path}?mode=ro", uri=True) as db:
                if db.execute("PRAGMA integrity_check").fetchone()[0] != "ok":
                    raise ValueError("Backup SQLite integrity failure")
    return manifest


def create(configuration, directory):
    if configuration.get("writers_quiescent") is not True:
        raise ValueError("Quiesce and confirm every platform writer before final backup")
    directory.mkdir(mode=0o700, parents=True, exist_ok=False)
    files = []
    for entry in configuration["files"]:
        source = Path(entry["source"])
        relative = safe_relative(entry["path"])
        destination = directory / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        if (
            source.stat().st_size + configuration.get("disk_headroom", 15 * 1024**3)
            > shutil.disk_usage(directory).free
        ):
            raise OSError(28, "Use off-production storage for backups")
        if entry.get("sqlite"):
            backup_sqlite(source, destination)
        else:
            with source.open("rb") as incoming, destination.open("xb") as outgoing:
                shutil.copyfileobj(incoming, outgoing)
                outgoing.flush()
                os.fsync(outgoing.fileno())
        files.append(
            {
                "path": str(relative),
                "size": destination.stat().st_size,
                "sha256": sha256_file(destination),
                "sqlite": entry.get("sqlite", False),
            }
        )
        atomic_json(directory / "backup.json", {"files": files, "complete": False})
    atomic_json(directory / "backup.json", {"files": files, "complete": True})
    verify(directory)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=["create", "verify", "rehearse-restore"])
    parser.add_argument("directory", type=Path)
    parser.add_argument("--configuration", type=Path)
    parser.add_argument("--restore-to", type=Path)
    args = parser.parse_args()
    if args.action == "create":
        create(json.loads(args.configuration.read_text()), args.directory)
    else:
        manifest = verify(args.directory)
        if not manifest.get("complete"):
            raise ValueError("Incomplete backup")
        if args.action == "rehearse-restore":
            args.restore_to.mkdir(mode=0o700, parents=True, exist_ok=False)
            for entry in manifest["files"]:
                destination = args.restore_to / safe_relative(entry["path"])
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(args.directory / entry["path"], destination)
            shutil.copyfile(args.directory / "backup.json", args.restore_to / "backup.json")
            verify(args.restore_to)


if __name__ == "__main__":
    main()
