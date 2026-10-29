#!/usr/bin/env python3
from __future__ import annotations

import socket
import tempfile
import threading
import unittest
from pathlib import Path

import numpy as np

from tools.gpu_embed_service.kinform_stream_ipc import (
    StreamClient,
    connect_unix_socket,
    recv_frame,
    send_frame,
)


class KinFormStreamIPCTests(unittest.TestCase):
    def test_frame_roundtrip_socketpair(self):
        try:
            left, right = socket.socketpair()
        except PermissionError:
            self.skipTest("Socket operations are restricted in this environment.")
        try:
            payload = np.arange(12, dtype=np.float32).reshape(3, 4).tobytes(order="C")
            header = {"type": "RESIDUE_READY", "seq_id": "sid_1", "shape": [3, 4], "dtype": "float32"}
            try:
                send_frame(left, header, payload)
                out_header, out_payload = recv_frame(right)
            except PermissionError:
                self.skipTest("Socket operations are restricted in this environment.")
            self.assertEqual(out_header["type"], "RESIDUE_READY")
            self.assertEqual(out_header["seq_id"], "sid_1")
            arr = np.frombuffer(out_payload, dtype=np.float32).reshape(3, 4)
            np.testing.assert_allclose(arr, np.arange(12, dtype=np.float32).reshape(3, 4))
        finally:
            left.close()
            right.close()

    def test_connect_unix_socket_and_stream_client(self):
        with tempfile.TemporaryDirectory(prefix="kinform_ipc_") as tmp:
            sock_path = Path(tmp) / "ipc.sock"
            srv = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
            try:
                srv.bind(str(sock_path))
            except PermissionError:
                srv.close()
                self.skipTest("Unix socket bind is restricted in this environment.")
            srv.listen(1)

            def _server():
                # The direct connectivity probe closes without sending a frame.
                probe, _ = srv.accept()
                probe.close()
                conn, _ = srv.accept()
                conn.settimeout(3)
                try:
                    header, payload = recv_frame(conn)
                    self.assertEqual(header["type"], "PING")
                    send_frame(conn, {"type": "PONG"}, payload)
                finally:
                    conn.close()

            thread = threading.Thread(target=_server, daemon=True)
            thread.start()

            try:
                direct = connect_unix_socket(str(sock_path))
            except PermissionError:
                srv.close()
                self.skipTest("Socket connect is restricted in this environment.")
            direct.close()

            client = StreamClient(str(sock_path))
            client.connect().settimeout(3)
            client.send({"type": "PING"}, b"abc")
            header, payload = client.recv()
            self.assertEqual(header["type"], "PONG")
            self.assertEqual(payload, b"abc")
            client.close()
            srv.close()
            thread.join(timeout=1.0)


if __name__ == "__main__":
    unittest.main()
