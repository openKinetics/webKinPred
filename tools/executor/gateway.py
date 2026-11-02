"""Authenticated read-only peer transfer; no arbitrary filesystem paths."""

from __future__ import annotations

import argparse
import functools
import hmac
import io
import json
import tarfile
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from api.execution.artifacts import ArtifactStore


class Handler(BaseHTTPRequestHandler):
    def __init__(self, *args, store, token, **kwargs):
        self.store = store
        self.token = token
        super().__init__(*args, **kwargs)

    def do_GET(self):
        if not self.token or not hmac.compare_digest(
            self.headers.get("Authorization", ""), "Bearer " + self.token
        ):
            self.send_error(401)
            return
        if not self.path.startswith("/blobs/"):
            self.send_error(404)
            return
        try:
            path = self.store.path(self.path.removeprefix("/blobs/"))
            handle = path.open("rb")
        except (ValueError, OSError):
            self.send_error(404)
            return
        with handle:
            self.send_response(200)
            self.send_header("Content-Type", "application/octet-stream")
            self.send_header("Content-Length", str(path.stat().st_size))
            self.end_headers()
            while block := handle.read(1024 * 1024):
                self.wfile.write(block)

    def do_POST(self):
        if not self.token or not hmac.compare_digest(
            self.headers.get("Authorization", ""), "Bearer " + self.token
        ):
            self.send_error(401)
            return
        if self.path != "/blobs":
            self.send_error(404)
            return
        try:
            length = int(self.headers.get("Content-Length", 0))
            if not 0 < length <= 2 * 1024**2:
                raise ValueError("Manifest too large")
            rows = json.loads(self.rfile.read(length))["blobs"]
            if not isinstance(rows, list) or not 1 <= len(rows) <= 1024:
                raise ValueError("Manifest count outside limits")
            wanted = {}
            for row in rows:
                path = self.store.path(row["checksum"])
                size = row["size"]
                if not isinstance(size, int) or isinstance(size, bool) or size < 0:
                    raise ValueError("Invalid artifact size")
                if row["checksum"] in wanted and wanted[row["checksum"]][1] != size:
                    raise ValueError("Conflicting artifact sizes")
                wanted[row["checksum"]] = (path, size)
            if sum(size for _, size in wanted.values()) > 256 * 1024**2:
                raise ValueError("Pack exceeds transfer window")
        except (ValueError, KeyError, TypeError):
            self.send_error(400)
            return
        present, missing = {}, []
        for checksum, (path, size) in wanted.items():
            try:
                valid = path.is_file() and path.stat().st_size == size
            except OSError:
                valid = False
            if valid:
                present[checksum] = size
            else:
                missing.append(checksum)
        manifest = json.dumps({"version": 1, "present": present, "missing": missing}).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/x-tar")
        self.end_headers()
        try:
            with tarfile.open(fileobj=self.wfile, mode="w|") as archive:
                info = tarfile.TarInfo("manifest.json")
                info.size = len(manifest)
                archive.addfile(info, io.BytesIO(manifest))
                for checksum, size in present.items():
                    info = tarfile.TarInfo(checksum)
                    info.size = size
                    with wanted[checksum][0].open("rb") as stream:
                        archive.addfile(info, stream)
        except (OSError, BrokenPipeError):
            # A lost connection or concurrently removed source is never a ready
            # partial artifact. The receiver verifies each immutable member.
            self.close_connection = True

    def log_message(self, _format, *args):
        return  # Never log bearer tokens or user-controlled artifact paths.


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("config", type=Path)
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    handler = functools.partial(
        Handler, store=ArtifactStore(Path(config["root"]) / "blobs"), token=config["artifact_token"]
    )
    server = ThreadingHTTPServer((config["bind"], config.get("artifact_port", 8082)), handler)
    server.serve_forever()


if __name__ == "__main__":
    main()
