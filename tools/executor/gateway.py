"""Authenticated read-only peer transfer; no arbitrary filesystem paths."""

from __future__ import annotations

import argparse
import functools
import hmac
import json
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
