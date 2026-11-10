import functools
import hashlib
import io
import json
import tarfile
import threading
from http.server import ThreadingHTTPServer
from types import SimpleNamespace

import pytest
from api.execution.artifacts import ArtifactStore
from api.execution.errors import MissingEmbedding
from tools.executor.gateway import Handler
from tools.executor.transfers import fetch, materialize, windows


@pytest.fixture
def transfer_pair(tmp_path):
    source, target = ArtifactStore(tmp_path / "source"), ArtifactStore(tmp_path / "target")
    source.root.mkdir()
    target.root.mkdir()
    http = ThreadingHTTPServer(
        ("127.0.0.1", 0), functools.partial(Handler, store=source, token="test-token")
    )
    thread = threading.Thread(target=http.serve_forever, daemon=True)
    thread.start()
    updates = []
    agent = SimpleNamespace(
        store=target,
        host="target",
        root=tmp_path,
        config={
            "budgets": {"disk_headroom": 0},
            "peers": {
                "source": {"url": f"http://127.0.0.1:{http.server_port}", "token": "test-token"}
            },
        },
        request=lambda op, data, attempt: updates.append((op, data, attempt)),
    )
    yield source, agent, updates
    http.shutdown()
    http.server_close()
    thread.join()


def blob(store, value):
    checksum = hashlib.sha256(value).hexdigest()
    store.publish(io.BytesIO(value), checksum, len(value))
    return checksum


def request(rows):
    return {
        "id": "attempt",
        "stage_id": "stage",
        "token": "ownership",
        "kind": "predict",
        "artifact_protocol": 2,
        "artifacts": rows,
    }


def test_thousand_small_files_use_one_pack_and_one_readiness_commit(
    transfer_pair, tmp_path, monkeypatch
):
    source, agent, updates = transfer_pair
    rows = []
    for index in range(1000):
        value = f"vector-{index}".encode()
        checksum = blob(source, value)
        rows.append(
            {
                "id": str(index),
                "path": f"inputs/{index}.npy",
                "checksum": checksum,
                "size": len(value),
                "hosts": ["source"],
            }
        )
    from tools.executor import transfers

    calls = []
    original = transfers.fetch

    def counted(*args):
        calls.append(len(args[-1]))
        return original(*args)

    monkeypatch.setattr(transfers, "fetch", counted)
    metrics = materialize(agent, request(rows), tmp_path / "work")
    assert calls == [1000]
    assert len(updates) == 1 and updates[0][0] == "artifact-locations"
    assert len(updates[0][1]["locations"]) == 1000
    assert metrics["transferred_bytes"] == sum(row["size"] for row in rows)
    assert (tmp_path / "work/media/inputs/999.npy").read_bytes() == b"vector-999"
    updates.clear()
    calls.clear()
    for row in rows:
        row["hosts"].append("target")
    metrics = materialize(agent, request(rows), tmp_path / "second")
    assert not calls and not updates and metrics["local_cache_hits"] == 1000


def test_missing_files_are_reported_together_and_no_private_partial_input_is_exposed(
    transfer_pair, tmp_path
):
    _, agent, updates = transfer_pair
    rows = [
        {"id": str(i), "path": f"{i}.npy", "checksum": str(i) * 64, "size": 12, "hosts": ["source"]}
        for i in range(3)
    ]
    with pytest.raises(MissingEmbedding) as caught:
        materialize(agent, request(rows), tmp_path / "work")
    assert caught.value.missing_artifacts == ["0", "1", "2"]
    assert len(updates) == 1
    assert all(row["ready"] is False for row in updates[0][1]["locations"])
    assert not (tmp_path / "work/media").exists()


@pytest.mark.parametrize(
    "malformation", ["symlink", "unexpected", "incomplete", "checksum", "manifest"]
)
def test_invalid_pack_never_publishes_an_unverified_blob(transfer_pair, monkeypatch, malformation):
    _, agent, _ = transfer_pair
    value = b"expected"
    checksum = hashlib.sha256(value).hexdigest()
    stream = io.BytesIO()
    with tarfile.open(fileobj=stream, mode="w") as archive:
        manifest = {"version": 1, "present": {checksum: len(value)}, "missing": []}
        data = json.dumps(manifest if malformation != "manifest" else []).encode()
        info = tarfile.TarInfo("manifest.json")
        info.size = len(data)
        archive.addfile(info, io.BytesIO(data))
        if malformation != "incomplete":
            info = tarfile.TarInfo("other" if malformation == "unexpected" else checksum)
            info.size = len(value)
            if malformation == "symlink":
                info.type = tarfile.SYMTYPE
                info.linkname = "/tmp/target"
            archive.addfile(
                info, io.BytesIO(b"x" * len(value) if malformation == "checksum" else value)
            )
    stream.seek(0)
    monkeypatch.setattr("urllib.request.urlopen", lambda *args, **kwargs: stream)
    with pytest.raises(ValueError):
        fetch(agent, "source", [{"checksum": checksum, "size": len(value)}])
    assert not agent.store.path(checksum).exists()


def test_transfer_windows_bound_count_and_bytes():
    rows = [{"size": 60} for _ in range(8)]
    assert list(map(len, windows(rows, max_items=3, max_bytes=130))) == [2, 2, 2, 2]


def test_capability_reload_cannot_change_images_or_allow_unknown_mixtures(tmp_path):
    from tools.executor.agent import Agent

    registry = tmp_path / "capabilities.json"
    registry.write_text(
        json.dumps(
            {
                "profiles": ["a" * 64, "b" * 64],
                "validated_gpu_combinations": [["a" * 64, "b" * 64]],
                "images": ["untrusted"],
            }
        )
    )
    config = {
        "root": str(tmp_path / "agent"),
        "host_id": "test",
        "images": ["pinned"],
        "capabilities_file": str(registry),
        "capabilities": {"runtime_fingerprint": "verified"},
    }
    agent = Agent(config)
    agent.refresh_capabilities()
    assert (
        config["images"] == ["pinned"]
        and config["capabilities"]["runtime_fingerprint"] == "verified"
    )
    assert config["capabilities"]["profiles"] == ["a" * 64, "b" * 64]
    registry.write_text(
        json.dumps({"profiles": ["a" * 64], "validated_gpu_combinations": [["a" * 64, "b" * 64]]})
    )
    with pytest.raises(ValueError):
        agent.refresh_capabilities()


def test_blob_verification_is_reused_until_the_blob_is_replaced_or_changed(tmp_path, monkeypatch):
    import os

    from api.execution import artifacts

    store = ArtifactStore(tmp_path / "blobs")
    data = b"residue matrix"
    checksum = hashlib.sha256(data).hexdigest()
    blob = store.publish(io.BytesIO(data), checksum, len(data))
    hashed = []
    real = artifacts.sha256_file
    monkeypatch.setattr(artifacts, "sha256_file", lambda path: hashed.append(path) or real(path))
    # A blob this store wrote, or once hashed, is not read again for each use.
    assert store.verified(blob, checksum) and store.verified(blob, checksum)
    assert hashed == []
    # Atomic replacement gives a new inode: hash again and detect corruption.
    replacement = blob.with_name(".replacement")
    replacement.write_bytes(b"residue matrlx")
    os.replace(replacement, blob)
    assert not store.verified(blob, checksum)
    assert len(hashed) == 1
    # An in-place rewrite changes the modification time.
    os.utime(blob, ns=(1, 1))
    blob.write_bytes(data)
    assert store.verified(blob, checksum) and store.verified(blob, checksum)
    assert len(hashed) == 2
    blob.unlink()
    assert not store.verified(blob, checksum)
