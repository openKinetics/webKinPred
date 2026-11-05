import contextlib
import io
import json
import os
import struct
import time
from datetime import timedelta
from pathlib import Path
from unittest.mock import Mock

import pytest
from tools import cleanup_legacy_esm1v as cleanup


@pytest.mark.parametrize("terminal", [False, True])
def test_progress_is_compact_and_never_prints_per_file(terminal):
    class Output(io.StringIO):
        def isatty(self):
            return terminal

    stream = Output()
    elapsed = [0]
    with cleanup.Progress(stream=stream, clock=lambda: elapsed[0]) as progress:
        progress.update("checking database")
        progress.update("scan", 0, 10000, eligible=0, size=0)
        for number in range(1, 10000):
            progress.update("scan", number, 10000, eligible=number, size=number)
        assert stream.getvalue().count("[esm1v]") == 2
        elapsed[0] = 2
        progress.update("scan", 9999, 10000, eligible=9999, size=1024**3)
        progress.update("scan", 10000, 10000, eligible=10000, size=1024**3)
    output = stream.getvalue()
    assert output.count("[esm1v]") == 4
    assert "10000/10000 | eligible 10000 (1.00 GiB)" in output
    assert output.count("\n") == (1 if terminal else 4)


def matrix(path, shape=(3, 1280), descriptor="<f4"):
    header = repr({"descr": descriptor, "fortran_order": False, "shape": shape}).encode() + b"\n"
    count = 1
    for value in shape:
        count *= value
    path.write_bytes(
        b"\x93NUMPY\x01\x00" + struct.pack("<H", len(header)) + header + b"\0" * (count * 4)
    )
    os.utime(path, (cleanup.CUTOFF - 100, cleanup.CUTOFF - 100))
    return path


@pytest.fixture
def directory(tmp_path):
    root = tmp_path / "media" / "sequence_info" / "esm1v"
    root.mkdir(parents=True)
    fd = os.open(root, os.O_RDONLY | os.O_DIRECTORY)
    try:
        yield root, fd
    finally:
        os.close(fd)


def test_identifies_only_old_unreferenced_single_link_residue_arrays(directory):
    root, fd = directory
    good = matrix(root / "012345abcdef.npy")
    needed = matrix(root / "112345abcdef.npy")
    opened = matrix(root / "212345abcdef.npy")
    vector = matrix(root / "312345abcdef.npy", (1280,))
    recent = matrix(root / "412345abcdef.npy")
    os.utime(recent, (cleanup.CUTOFF + 10, cleanup.CUTOFF + 10))
    symlink = root / "512345abcdef.npy"
    symlink.symlink_to(good)
    hardlink = matrix(root / "612345abcdef.npy")
    os.link(hardlink, root / "kept-hardlink")
    unknown = matrix(root / "research.npy")
    corrupt = root / "712345abcdef.npy"
    corrupt.write_bytes(b"not numpy")
    os.utime(corrupt, (cleanup.CUTOFF - 100, cleanup.CUTOFF - 100))
    manual = matrix(root / "812345abcdef.npy")
    other_dtype = matrix(root / "912345abcdef.npy", descriptor="<i4")
    protected = {cleanup.PREFIX + needed.name}
    in_use = {(opened.stat().st_dev, opened.stat().st_ino)}
    reasons = {
        p.name: cleanup.inspect_file(fd, p.name, protected, in_use, {manual.name})["reason"]
        for p in [
            good,
            needed,
            opened,
            vector,
            recent,
            symlink,
            hardlink,
            unknown,
            corrupt,
            manual,
            other_dtype,
        ]
    }
    assert reasons == {
        good.name: "eligible",
        needed.name: "database_reference",
        opened.name: "open_or_mapped",
        vector.name: "not_verified_residue_matrix",
        recent.name: "post_migration",
        symlink.name: "not_single_link_regular_file",
        hardlink.name: "not_single_link_regular_file",
        unknown.name: "unknown_name",
        corrupt.name: "not_verified_residue_matrix",
        manual.name: "operator_keep",
        other_dtype.name: "not_verified_residue_matrix",
    }


def test_changed_file_and_name_replacement_are_never_deleted(directory, monkeypatch):
    root, fd = directory
    path = matrix(root / "012345abcdef.npy")
    row = cleanup.inspect_file(fd, path.name, set(), set(), set())
    with path.open("ab") as handle:
        handle.write(b"changed")
    with cleanup.quarantine(fd) as qfd:
        with pytest.raises(RuntimeError, match="changed"):
            cleanup.remove_unchanged(fd, qfd, row)
    assert path.exists()

    row = cleanup.inspect_file(fd, matrix(path).name, set(), set(), set())
    original_rename = os.rename

    def replace_before_rename(*args, **kwargs):
        path.unlink()
        path.write_bytes(b"someone else's new file")
        return original_rename(*args, **kwargs)

    monkeypatch.setattr(os, "rename", replace_before_rename)
    with cleanup.quarantine(fd) as qfd:
        with pytest.raises(RuntimeError, match="Replacement preserved"):
            cleanup.remove_unchanged(fd, qfd, row)
    assert path.read_bytes() == b"someone else's new file"


def test_quarantine_never_recursively_deletes_unexpected_files(directory):
    root, fd = directory
    with cleanup.quarantine(fd) as qfd:
        file_fd = os.open("unexpected", os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600, dir_fd=qfd)
        os.write(file_fd, b"keep me")
        os.close(file_fd)
    assert next(root.glob(".cleanup-*/unexpected")).read_bytes() == b"keep me"


def test_host_audit_fails_closed_on_unreadable_process(tmp_path, monkeypatch):
    process = tmp_path / "123456"
    (process / "fd").mkdir(parents=True)
    (process / "maps").write_text("")
    original = Path.open

    def denied(self, *args, **kwargs):
        if self.name == "maps":
            raise PermissionError("can't inspect maps")
        return original(self, *args, **kwargs)

    monkeypatch.setattr(Path, "open", denied)
    with pytest.raises(PermissionError):
        cleanup.open_inodes(tmp_path)


def test_unrecognized_container_mount_or_legacy_worker_stops_cleanup(tmp_path, monkeypatch):
    root = tmp_path / "media" / "sequence_info" / "esm1v"
    backend = {"Name": "/webkinpred-backend-1", "Config": {"Labels": {}}, "Mounts": []}
    stranger = {
        "Name": "/research",
        "Config": {"Labels": {}},
        "Mounts": [{"Source": str(tmp_path / "media")}],
    }
    containers = [backend, stranger]
    monkeypatch.setattr(
        cleanup, "docker", lambda *args: "id" if args[0] == "ps" else json.dumps(containers)
    )
    with pytest.raises(RuntimeError, match="Unrecognized container"):
        cleanup.check_containers(root, "webkinpred-backend-1")
    stranger["Mounts"] = []
    stranger["Config"]["Labels"] = {
        "com.docker.compose.project": "webkinpred",
        "com.docker.compose.service": "celery",
    }
    with pytest.raises(RuntimeError, match="Legacy worker"):
        cleanup.check_containers(root, "webkinpred-backend-1")


def test_operator_trusted_reader_may_keep_only_a_read_only_view(tmp_path, monkeypatch):
    root = tmp_path / "media" / "sequence_info" / "esm1v"
    backend = {"Name": "/webkinpred-backend-1", "Config": {"Labels": {}}, "Mounts": []}
    # e.g. a data portal that reads other embedding families under sequence_info.
    portal = {
        "Name": "/portal",
        "Config": {"Labels": {}},
        "Mounts": [{"Source": str(tmp_path / "media" / "sequence_info"), "RW": False}],
    }
    containers = [backend, portal]
    monkeypatch.setattr(
        cleanup, "docker", lambda *args: "id" if args[0] == "ps" else json.dumps(containers)
    )
    with pytest.raises(RuntimeError, match="Unrecognized container"):
        cleanup.check_containers(root, "webkinpred-backend-1")
    cleanup.check_containers(root, "webkinpred-backend-1", {"portal"})
    with pytest.raises(RuntimeError, match="Unrecognized container"):
        cleanup.check_containers(root, "webkinpred-backend-1", {"another"})
    portal["Mounts"][0]["RW"] = True
    with pytest.raises(RuntimeError, match="can write"):
        cleanup.check_containers(root, "webkinpred-backend-1", {"portal"})
    portal["Mounts"][0]["RW"] = False
    portal["Config"]["Labels"] = {
        "com.docker.compose.project": "webkinpred",
        "com.docker.compose.service": "celery",
    }
    with pytest.raises(RuntimeError, match="Legacy worker"):
        cleanup.check_containers(root, "webkinpred-backend-1", {"portal"})


def test_preview_and_apply_keep_dependencies_and_protected_other_directories(
    directory, tmp_path, monkeypatch, capsys
):
    root, _ = directory
    unused = matrix(root / "012345abcdef.npy")
    needed = matrix(root / "112345abcdef.npy")
    research = matrix(root.parent / "t5-research.npy")
    monkeypatch.setattr(cleanup.os, "geteuid", lambda: 0)
    checked = []
    monkeypatch.setattr(cleanup, "check_containers", lambda *args: checked.append(args[2:]))
    monkeypatch.setattr(cleanup, "open_inodes", lambda: set())
    monkeypatch.setattr(cleanup.fcntl, "flock", lambda *args: None)
    original_open = Path.open

    def redirect_lock(self, *args, **kwargs):
        if str(self) == "/run/lock/webkinpred-legacy-esm1v.lock":
            return original_open(tmp_path / "lock", *args, **kwargs)
        return original_open(self, *args, **kwargs)

    monkeypatch.setattr(Path, "open", redirect_lock)

    @contextlib.contextmanager
    def guard(*args):
        yield {cleanup.PREFIX + needed.name}, time.monotonic() + 15, Mock(poll=lambda: None)

    monkeypatch.setattr(cleanup, "database_guard", guard)
    cleanup.main(["--repo", str(tmp_path)])
    assert unused.exists() and needed.exists() and research.exists()
    assert json.loads(capsys.readouterr().out)["deleted"] == 0
    report = tmp_path / "audit.jsonl"
    cleanup.main(["--repo", str(tmp_path), "--apply", "--report", str(report)])
    assert not unused.exists() and needed.exists() and research.exists()
    assert json.loads(capsys.readouterr().out)["deleted"] == 1
    assert json.loads(report.read_text().splitlines()[-1])["deleted"] == 1
    # Refuse to overwrite evidence, even when an apply is repeated.
    with pytest.raises(FileExistsError):
        cleanup.main(["--repo", str(tmp_path), "--apply", "--report", str(report)])
    # Operator-reviewed readers reach both container checks and the audit trail.
    checked.clear()
    reviewed = tmp_path / "reviewed.jsonl"
    cleanup.main(
        ["--repo", str(tmp_path), "--apply", "--report", str(reviewed), "--trust-reader", "portal"]
    )
    assert checked == [(frozenset({"portal"}),)] * 2
    assert json.loads(capsys.readouterr().out)["trusted_readers"] == ["portal"]
    assert json.loads(reviewed.read_text().splitlines()[0])["trusted_readers"] == ["portal"]


def execute_database_guard(monkeypatch):
    from django.db import connection

    # This is an operator index, not an application model migration. Build it
    # on the isolated test DB too; never use production settings for these tests.
    with connection.cursor() as cursor:
        cursor.execute(
            "CREATE INDEX IF NOT EXISTS wk_legacy_esm1v_path ON api_artifact "
            "((identity ->> 'path') text_pattern_ops) "
            "WHERE (identity ->> 'path') LIKE 'sequence\\_info/esm1v/%'"
        )
        cursor.execute(
            "CREATE INDEX IF NOT EXISTS wk_legacy_esm1v_dependency ON api_artifactdependency "
            "(path text_pattern_ops) INCLUDE (stage_id) "
            "WHERE path LIKE 'sequence\\_info/esm1v/%'"
        )
    output = io.StringIO()
    monkeypatch.setattr(cleanup.select, "select", lambda *args: ([1], [], []))
    monkeypatch.setattr("sys.stdin", io.StringIO("release\n"))
    monkeypatch.setattr("sys.stdout", output)
    exec(cleanup.DATABASE_GUARD, {})
    return json.loads(output.getvalue())


@pytest.mark.django_db(transaction=True)
def test_index_preparation_is_idempotent_and_query_uses_partial_index(monkeypatch):
    from api.execution.models import Artifact, ArtifactDependency
    from django.db import connection

    with connection.cursor() as cursor:
        cursor.execute("DROP INDEX IF EXISTS wk_legacy_esm1v_path")
        cursor.execute("DROP INDEX IF EXISTS wk_legacy_esm1v_dependency")
    output = io.StringIO()
    monkeypatch.setattr("sys.stdout", output)
    exec(cleanup.PREPARE_INDEX, {})
    exec(cleanup.PREPARE_INDEX, {})
    events = [
        (event["event"], event["index"])
        for event in map(json.loads, output.getvalue().splitlines())
    ]
    assert events == [
        ("building_esm1v_lookup_index", cleanup.INDEX_NAME),
        ("esm1v_lookup_index_ready", cleanup.INDEX_NAME),
        ("building_esm1v_lookup_index", cleanup.DEPENDENCY_INDEX_NAME),
        ("esm1v_lookup_index_ready", cleanup.DEPENDENCY_INDEX_NAME),
        ("esm1v_lookup_index_ready", cleanup.INDEX_NAME),
        ("esm1v_lookup_index_ready", cleanup.DEPENDENCY_INDEX_NAME),
    ]
    # Force an index path even on this tiny test DB to prove the predicate and
    # expression match Django's exact SQL, including the escaped underscore.
    with connection.cursor() as cursor:
        cursor.execute("SET enable_seqscan=off")
    try:
        query = Artifact.objects.filter(identity__path__startswith=cleanup.PREFIX)
        assert "wk_legacy_esm1v_path" in query.explain()
        # The guard's dependency snapshot reads only this small index.
        references = ArtifactDependency.objects.filter(path__startswith=cleanup.PREFIX).values_list(
            "path", "stage_id"
        )
        assert "Index Only Scan using wk_legacy_esm1v_dependency" in references.explain()
    finally:
        with connection.cursor() as cursor:
            cursor.execute("RESET enable_seqscan")


@pytest.mark.django_db(transaction=True)
def test_preparation_lock_does_not_collide_with_live_scheduler_priority_lock(monkeypatch):
    from django.db import connection

    priority_connection = connection.copy(alias="cleanup_priority_collision_test")
    output = io.StringIO()
    monkeypatch.setattr("sys.stdout", output)
    try:
        with priority_connection.cursor() as cursor:
            cursor.execute("SELECT pg_advisory_lock(481726393)")
        exec(cleanup.PREPARE_INDEX, {})
        assert json.loads(output.getvalue().splitlines()[-1])["event"] == "esm1v_lookup_index_ready"
    finally:
        with priority_connection.cursor() as cursor:
            cursor.execute("SELECT pg_advisory_unlock(481726393)")
        priority_connection.close()


def test_database_errors_are_compact_and_do_not_emit_tracebacks():
    import subprocess
    import sys

    result = subprocess.run(
        [sys.executable, "-c", cleanup.database_entry("raise ValueError('sensitive detail')")],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 1
    assert result.stderr == ""
    assert json.loads(result.stdout) == {"error": "ValueError", "files_preserved": True}


@pytest.mark.django_db(transaction=True)
def test_database_guard_protects_held_work_live_attempts_and_retained_artifacts(
    tmp_path, monkeypatch
):
    from api.execution.models import (
        Artifact,
        ArtifactDependency,
        ExecutionAttempt,
        ExecutionSpec,
        ExecutorHost,
        WorkStage,
    )
    from django.test import override_settings
    from django.utils import timezone

    root = tmp_path / "sequence_info" / "esm1v"
    root.mkdir(parents=True)
    for host in ["production", "gpu"]:
        ExecutorHost.objects.create(
            id=host,
            reconciled=True,
            heartbeat_at=timezone.now(),
            telemetry={"legacy_consumers": []},
        )
    execution = ExecutionSpec.objects.create(
        specification={}, checksum="a" * 64, input_checksum="b" * 64
    )
    held = WorkStage.objects.create(execution=execution, key="held", kind="predict", state="held")
    done = WorkStage.objects.create(
        execution=execution, key="done", kind="predict", state="completed"
    )
    canceled = WorkStage.objects.create(
        execution=execution, key="cancel", kind="predict", state="canceled"
    )
    ExecutionAttempt.objects.create(
        stage=canceled, host_id="production", state="uncertain", lease_until=timezone.now()
    )
    expected = set()
    for number, stage, policy in [
        (1, held, "transient"),
        (2, done, "transient"),
        (3, canceled, "transient"),
        (4, done, "permanent"),
        (5, None, "protected"),
    ]:
        path = cleanup.PREFIX + f"{number:012x}.npy"
        artifact = Artifact.objects.create(
            id=str(number) * 64,
            checksum="c" * 64,
            size=100,
            identity={"path": path},
            retention=policy,
        )
        if stage:
            ArtifactDependency.objects.create(artifact=artifact, stage=stage, path=path)
        if number != 2:
            expected.add(path)
    with override_settings(EXECUTION_V2_ENABLED=True, MEDIA_ROOT=str(tmp_path)):
        snapshot = execute_database_guard(monkeypatch)
    assert set(snapshot["protected"]) == expected
    assert (snapshot["device"], snapshot["inode"]) == (root.stat().st_dev, root.stat().st_ino)


@pytest.mark.django_db(transaction=True)
@pytest.mark.parametrize("unsafe", ["stale", "legacy", "unknown", "unmigrated"])
def test_database_guard_refuses_missing_safety_evidence(tmp_path, monkeypatch, unsafe):
    from api.execution.models import ExecutorHost
    from api.models import Job
    from django.test import override_settings
    from django.utils import timezone

    for host in ["production", "gpu"]:
        ExecutorHost.objects.create(
            id=host,
            reconciled=True,
            heartbeat_at=timezone.now(),
            telemetry={"legacy_consumers": []},
        )
    if unsafe == "stale":
        ExecutorHost.objects.filter(pk="gpu").update(
            heartbeat_at=timezone.now() - timedelta(minutes=2)
        )
    elif unsafe == "legacy":
        ExecutorHost.objects.filter(pk="gpu").update(telemetry={"legacy_consumers": ["celery"]})
    elif unsafe == "unknown":
        ExecutorHost.objects.filter(pk="production").update(telemetry={})
    else:
        Job.objects.create(public_id="waiting", status="Pending", prediction_type="kcat")
    with override_settings(EXECUTION_V2_ENABLED=True, MEDIA_ROOT=str(tmp_path)):
        with pytest.raises(RuntimeError):
            execute_database_guard(monkeypatch)


@pytest.mark.django_db(transaction=True)
def test_index_preparation_refuses_an_unexpected_dependency_index(monkeypatch):
    from django.db import connection

    with connection.cursor() as cursor:
        cursor.execute("DROP INDEX IF EXISTS wk_legacy_esm1v_dependency")
        cursor.execute("CREATE INDEX wk_legacy_esm1v_dependency ON api_artifactdependency (path)")
    monkeypatch.setattr("sys.stdout", io.StringIO())
    try:
        with pytest.raises(RuntimeError, match="Unexpected cleanup index definition"):
            exec(cleanup.PREPARE_INDEX, {})
    finally:
        with connection.cursor() as cursor:
            cursor.execute("DROP INDEX IF EXISTS wk_legacy_esm1v_dependency")


def guard_hosts():
    from api.execution.models import ExecutorHost
    from django.utils import timezone

    for host in ["production", "gpu"]:
        ExecutorHost.objects.create(
            id=host,
            reconciled=True,
            heartbeat_at=timezone.now(),
            telemetry={"legacy_consumers": []},
        )


@pytest.mark.django_db(transaction=True)
def test_database_guard_requires_the_dependency_index(tmp_path, monkeypatch):
    from django.db import connection
    from django.test import override_settings

    guard_hosts()
    (tmp_path / "sequence_info" / "esm1v").mkdir(parents=True)

    with override_settings(EXECUTION_V2_ENABLED=True, MEDIA_ROOT=str(tmp_path)):
        execute_database_guard(monkeypatch)  # Builds both indexes, then succeeds.
        with connection.cursor() as cursor:
            cursor.execute("DROP INDEX wk_legacy_esm1v_dependency")
        monkeypatch.setattr("sys.stdin", io.StringIO("release\n"))
        with pytest.raises(RuntimeError, match="run --prepare-index first"):
            exec(cleanup.DATABASE_GUARD, {})


@pytest.mark.django_db(transaction=True)
@pytest.mark.parametrize("held_seconds", [0.5, 5])
def test_database_guard_waits_briefly_for_the_shared_graph_lock(
    tmp_path, monkeypatch, held_seconds
):
    import threading

    from django.db import connection
    from django.test import override_settings

    guard_hosts()
    (tmp_path / "sequence_info" / "esm1v").mkdir(parents=True)
    monkeypatch.setenv("WEBKINPRED_ESM1V_LOCK_WAIT", "2")
    locked, finished = threading.Event(), threading.Event()

    def hold():
        # Stands in for routine artifact collection or graph publication.
        holder = connection.copy(alias="cleanup_graph_lock_holder")
        try:
            with holder.cursor() as cursor:
                cursor.execute("SELECT pg_advisory_lock(%s)", [cleanup.GRAPH_LOCK])
                locked.set()
                finished.wait(held_seconds)
                cursor.execute("SELECT pg_advisory_unlock(%s)", [cleanup.GRAPH_LOCK])
        finally:
            holder.close()

    thread = threading.Thread(target=hold)
    thread.start()
    assert locked.wait(5)
    try:
        with override_settings(EXECUTION_V2_ENABLED=True, MEDIA_ROOT=str(tmp_path)):
            if held_seconds < 2:
                assert execute_database_guard(monkeypatch)["protected"] == []
            else:
                with pytest.raises(RuntimeError, match="Graph is busy"):
                    execute_database_guard(monkeypatch)
    finally:
        finished.set()
        thread.join()
