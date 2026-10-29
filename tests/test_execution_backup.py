import hashlib
import io
import json
import tarfile
from types import SimpleNamespace

import pytest
from tools.executor import backup


@pytest.mark.parametrize("fail_archive", [False, True])
def test_backup_publishes_completion_only_after_verified_objects(
    tmp_path, monkeypatch, fail_archive
):
    media = tmp_path / "media"
    source = media / "jobs" / "sample"
    source.mkdir(parents=True)
    (source / "input.csv").write_text("Protein Sequence,Substrate\nMAAA,CC\n")
    config = {
        "scratch": str(tmp_path / "scratch"),
        "minimum_free": 0,
        "container": "isolated-db",
        "role": "test",
        "databases": ["application", "sequences", "predictions"],
        "media": str(media),
        "s3_prefix": "s3://existing-backup/execution-v2",
    }
    uploaded = {}

    def dump(command, stdout, check):
        stdout.write(b"isolated postgres snapshot")

    def upload(cfg, stream, uri, size):
        data = stream.read()
        if fail_archive and uri.endswith("jobs.tar.gz"):
            raise RuntimeError("Injected off-host verification failure")
        uploaded[uri] = data
        return {"sha256": hashlib.sha256(data).hexdigest(), "bytes": len(data)}

    monkeypatch.setattr(backup.subprocess, "run", dump)
    monkeypatch.setattr(backup, "upload", upload)
    if fail_archive:
        with pytest.raises(RuntimeError, match="Injected"):
            backup.backup(config)
        assert not any(k.endswith("manifest.json") for k in uploaded)
    else:
        assert backup.backup(config)["complete"] is True
        manifest = json.loads(next(v for k, v in uploaded.items() if k.endswith("manifest.json")))
        assert manifest["complete"] is True and len(manifest["objects"]) == 4
        archive = next(v for k, v in uploaded.items() if k.endswith("jobs.tar.gz"))
        with tarfile.open(fileobj=io.BytesIO(archive), mode="r:gz") as tar:
            assert (
                tar.extractfile("jobs/sample/input.csv").read()
                == (source / "input.csv").read_bytes()
            )
    assert not list((tmp_path / "scratch").iterdir())


@pytest.mark.parametrize("corrupt", [False, True])
def test_offhost_backup_requires_downloaded_bytes_and_checksum(tmp_path, monkeypatch, corrupt):
    original = b"verified snapshot"
    monkeypatch.setattr(
        backup.subprocess,
        "Popen",
        lambda *a, **kw: SimpleNamespace(
            stdout=io.BytesIO(original if not corrupt else original + b"corrupt"), wait=lambda: 0
        ),
    )
    config = {"aws_user": "operator", "aws_binary": "/usr/bin/aws"}
    if corrupt:
        with pytest.raises(RuntimeError, match="verification failed"):
            backup.verify_object(
                config,
                "s3://existing-backup/file",
                hashlib.sha256(original).hexdigest(),
                len(original),
            )
    else:
        backup.verify_object(
            config, "s3://existing-backup/file", hashlib.sha256(original).hexdigest(), len(original)
        )
