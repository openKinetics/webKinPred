import os

import pytest
from tools.executor.stage_models import stage


def test_model_release_preserves_dirty_source_and_reuses_weights(tmp_path):
    source, patches, dest = (tmp_path / name for name in ["original", "patches", "release"])
    source.mkdir()
    patches.mkdir()
    (source / "weights.bin").write_bytes(b"deployment-specific-weights")
    (source / "adapter.py").write_text("old deployment code")
    (patches / "adapter.py").write_text("new code")
    stage(source, dest, patches)
    assert (source / "adapter.py").read_text() == "old deployment code"
    assert (dest / "adapter.py").read_text() == "new code"
    assert os.stat(source / "weights.bin").st_ino == os.stat(dest / "weights.bin").st_ino
    assert os.stat(source / "adapter.py").st_ino != os.stat(dest / "adapter.py").st_ino
    with pytest.raises(ValueError, match="new model release"):
        stage(source, dest, patches)


def test_model_release_rejects_patch_through_symlink(tmp_path):
    source, patches = tmp_path / "source", tmp_path / "patches"
    source.mkdir()
    patches.mkdir()
    original = tmp_path / "protected.py"
    original.write_text("preserve")
    (source / "adapter.py").symlink_to(original)
    (patches / "adapter.py").write_text("replacement")
    with pytest.raises(ValueError, match="symlink"):
        stage(source, tmp_path / "release", patches)
    assert original.read_text() == "preserve"


def test_model_inventory_includes_extensionless_published_checkpoints(tmp_path):
    from tools.executor.model_inventory import fingerprint

    checkpoint = tmp_path / "all--radius2--iteration50"
    checkpoint.write_bytes(b"published checkpoint weights")
    (tmp_path / "adapter.py").write_text("pass\n")
    manifest = fingerprint({"model": str(tmp_path)})
    assert "model/all--radius2--iteration50" in manifest["files"]
    checkpoint.write_bytes(b"updated checkpoint weights")
    assert fingerprint({"model": str(tmp_path)})["fingerprint"] != manifest["fingerprint"]


def test_control_broker_preserves_the_legacy_redis_acknowledgement_namespace(monkeypatch):
    from webKinPred.execution_settings import configure

    monkeypatch.setenv("EXECUTION_V2_ENABLED", "1")
    monkeypatch.delenv("EXECUTION_CONTROL_BROKER_URL", raising=False)
    namespace = {
        "ALLOWED_HOSTS": [],
        "DATABASES": {"default": {"ENGINE": "django.db.backends.postgresql"}},
        "CELERY_BROKER_URL": "redis://redis:6379/0",
    }
    configure(namespace)
    assert namespace["CELERY_BROKER_URL"] == "redis://redis:6379/3"
    assert namespace["CELERY_RESULT_BACKEND"] == "redis://redis:6379/3"
    monkeypatch.setenv("EXECUTION_CONTROL_BROKER_URL", "redis://redis:6379/0?other=true")
    namespace["CELERY_BROKER_URL"] = "redis://redis:6379/0"
    with pytest.raises(ValueError, match="separate Redis"):
        configure(namespace)
