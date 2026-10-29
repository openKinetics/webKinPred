"""Content-addressed local storage. Writes are checksum verified before publication."""

from __future__ import annotations

import hashlib
import json
import os
import re
import tempfile
from pathlib import Path

HEX = re.compile(r"^[a-f0-9]{64}$")


def sha256_file(path: Path) -> str:
    result = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def atomic_json(path: Path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=".publish-", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w") as handle:
            json.dump(data, handle, sort_keys=True, allow_nan=False)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        parent = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(parent)
        finally:
            os.close(parent)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def safe_relative(path: str) -> Path:
    relative = Path(path)
    if (
        not relative.parts
        or relative.is_absolute()
        or any(p in {"..", "."} for p in relative.parts)
    ):
        raise ValueError("Invalid relative artifact path")
    return relative


class ArtifactStore:
    def __init__(self, root: Path):
        self.root = root.resolve()

    def path(self, checksum: str) -> Path:
        if not HEX.fullmatch(checksum):
            raise ValueError("Invalid checksum")
        return self.root / checksum[:2] / checksum

    def publish(self, stream, checksum: str, size: int, *, minimum_free=0) -> Path:
        import shutil

        destination = self.path(checksum)
        destination.parent.mkdir(parents=True, exist_ok=True)
        if size < 0:
            raise ValueError("Negative artifact size")
        if (
            destination.is_file()
            and destination.stat().st_size == size
            and sha256_file(destination) == checksum
        ):
            return destination
        if shutil.disk_usage(self.root).free - size < minimum_free:
            raise OSError(28, "Artifact storage budget exhausted")
        descriptor, temporary = tempfile.mkstemp(prefix=".transfer-", dir=destination.parent)
        total = 0
        result = hashlib.sha256()
        try:
            with os.fdopen(descriptor, "wb") as handle:
                while block := stream.read(min(1024 * 1024, size - total + 1)):
                    total += len(block)
                    if total > size:
                        raise ValueError("Artifact exceeds declared size")
                    if (
                        minimum_free
                        and shutil.disk_usage(self.root).free - len(block) < minimum_free
                    ):
                        raise OSError(28, "Artifact storage headroom exhausted during transfer")
                    result.update(block)
                    handle.write(block)
                if total != size or result.hexdigest() != checksum:
                    raise ValueError("Incomplete or corrupt artifact")
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, destination)
            return destination
        finally:
            if os.path.exists(temporary):
                os.unlink(temporary)
