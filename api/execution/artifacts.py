"""Content-addressed local storage. Writes are checksum verified before publication."""

from __future__ import annotations

import hashlib
import json
import os
import re
import tempfile
import time
from pathlib import Path

HEX = re.compile(r"^[a-f0-9]{64}$")
VERIFIED_SECONDS = 6 * 3600
VERIFIED_LIMIT = 200_000


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
        # Blobs are only ever replaced atomically, never rewritten in place. Every
        # attempt hashed each input in full before copying it, so most host reads
        # were repeated verification. A blob keeps its verdict while its inode,
        # size and modification time are unchanged, for a bounded time.
        self._verified = {}

    def _identity(self, path: Path):
        stat = path.stat()
        return (stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns)

    def _remember(self, path: Path, identity):
        if len(self._verified) >= VERIFIED_LIMIT:
            self._verified.clear()
        self._verified[str(path)] = (identity, time.monotonic())

    def verified(self, path: Path, checksum: str) -> bool:
        """Whether ``path`` holds ``checksum``; rehashed after any replacement."""
        try:
            identity = self._identity(path)
        except FileNotFoundError:
            return False
        known = self._verified.get(str(path))
        if known and known[0] == identity and time.monotonic() - known[1] < VERIFIED_SECONDS:
            return True
        if sha256_file(path) != checksum:
            self._verified.pop(str(path), None)
            return False
        try:
            if self._identity(path) == identity:
                self._remember(path, identity)
        except FileNotFoundError:
            return False
        return True

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
            and self.verified(destination, checksum)
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
            self._remember(destination, self._identity(destination))
            return destination
        finally:
            if os.path.exists(temporary):
                os.unlink(temporary)
