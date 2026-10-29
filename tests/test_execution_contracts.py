import json

import numpy as np
import pytest
from api.execution import control
from api.execution.bounded import frames
from api.execution.middleware import ExecutionMaintenanceMiddleware
from api.execution.models import ResourceProfile, WorkStage
from api.execution.resources import GIB, admit
from api.execution.views import executor
from api.models import Job
from django.test import RequestFactory, override_settings


def test_maintenance_blocks_before_submission_or_quota_and_allows_downloads(tmp_path):
    flag = tmp_path / "maintenance"
    flag.touch()
    seen = []
    middleware = ExecutionMaintenanceMiddleware(lambda request: seen.append(request.path))
    with override_settings(EXECUTION_MAINTENANCE_FILE=str(flag)):
        result = middleware(RequestFactory().post("/api/v1/submit/"))
        assert result.status_code == 503
        assert result["Retry-After"] == "300"
        assert seen == []
        middleware(RequestFactory().get("/api/jobs/example/download/"))
        assert seen == ["/api/jobs/example/download/"]
        assert middleware(RequestFactory().post("/api/api-key/generate/")).status_code == 503
        assert middleware(RequestFactory().get("/admin/")).status_code == 503
        middleware(RequestFactory().post("/internal/v2/executors/gpu/register"))
        assert seen[-1] == "/internal/v2/executors/gpu/register"


@pytest.mark.django_db(transaction=True)
def test_maintenance_api_authentication_does_not_write_to_the_frozen_database(tmp_path):
    from api.models import ApiKey, ApiUser
    from api.utils.api_auth import require_api_key
    from django.http import JsonResponse

    key = ApiKey.objects.create(user=ApiUser.objects.create(ip_address="127.0.0.1"))
    previous = key.last_used
    flag = tmp_path / "maintenance"
    flag.touch()
    view = require_api_key(lambda request: JsonResponse({"available": True}))
    request = RequestFactory().get(
        "/api/v1/status/example/", HTTP_AUTHORIZATION="Bearer " + key.key
    )
    with override_settings(EXECUTION_MAINTENANCE_FILE=str(flag)):
        assert view(request).status_code == 200
    key.refresh_from_db()
    assert key.last_used == previous


def test_executor_credentials_are_bound_to_the_addressed_host():
    with override_settings(
        EXECUTION_HOSTS={"production": {"token": "production-token"}, "gpu": {"token": "gpu-token"}}
    ):
        request = RequestFactory().post(
            "/internal/v2/executors/gpu/claim",
            data="{}",
            content_type="application/json",
            HTTP_AUTHORIZATION="Bearer production-token",
        )
        assert executor(request, "gpu", "claim").status_code == 401


@pytest.mark.django_db(transaction=True)
def test_operational_metrics_are_authenticated_and_exclude_request_details():
    from api.execution.metrics import metrics
    from api.execution.models import ExecutorHost
    from django.utils import timezone

    ExecutorHost.objects.create(
        id="gpu",
        heartbeat_at=timezone.now(),
        telemetry={"ram_available": 123456, "attempts": {}},
        budgets={"ram": 1000000},
    )
    job = Job.objects.create(public_id="private-id", prediction_type="kcat", ip_address="10.1.2.3")
    control.ingest(job.public_id, {"private": "secret-input"}, "a" * 64)
    factory = RequestFactory()
    with override_settings(EXECUTION_METRICS_TOKEN="metrics-test-token"):
        assert metrics(factory.get("/internal/v2/metrics")).status_code == 401
        response = metrics(
            factory.get("/internal/v2/metrics", HTTP_AUTHORIZATION="Bearer metrics-test-token")
        )
    assert response.status_code == 200
    body = response.content.decode()
    assert 'webkinpred_execution_host_ram_available{host="gpu"} 123456' in body
    assert (
        'webkinpred_execution_stages{kind="prepare",method="",state="ready",waiting_reason=""} 1'
        in body
    )
    assert "private-id" not in body and "secret-input" not in body and "10.1.2.3" not in body


def test_byte_bounded_parsing_preserves_quoted_newlines_and_row_order(tmp_path):
    source = tmp_path / "input.csv"
    source.write_text(
        'Protein Sequence,Substrate,Note\nMAAA,CC,"line1\nline2"\nMCCC,O,next\nMDDD,N,last\n'
    )
    chunks = list(frames(source, max_rows=2))
    assert [len(frame) for frame in chunks] == [2, 1]
    assert chunks[0]["Note"].iloc[0] == "line1\nline2"
    assert chunks[1]["Protein Sequence"].iloc[0] == "MDDD"
    with pytest.raises(ValueError, match="smaller child-unit"):
        list(frames(source, max_bytes=16))


def test_nonfinite_telemetry_never_admits_work():
    result = admit(
        {"ram": GIB, "cpu": 1, "disk": 0},
        [],
        {"ram": 8 * GIB, "cpu": 4, "disk": GIB},
        {"ram_available": float("nan"), "disk_free": 8 * GIB},
        gpu=False,
    )
    assert not result.fits
    assert result.reason == "telemetry_unavailable"


@pytest.mark.django_db(transaction=True)
def test_public_waiting_reason_tracks_a_shared_gpu_dependency():
    from api.execution.status import execution_status

    jobs = [
        Job.objects.create(public_id="shared" + str(i), status="Pending", prediction_type="kcat")
        for i in range(2)
    ]
    executions = [control.ingest(job.public_id, {}, "a" * 64, paused=False) for job in jobs]
    producer = WorkStage.objects.create(execution=executions[0], key="gpu", kind="embedding")
    consumer = WorkStage.objects.create(execution=executions[1], key="cpu", kind="predict")
    consumer.dependencies.add(producer)
    with override_settings(EXECUTION_V2_ENABLED=True):
        assert execution_status(jobs[1])["execution_state"] == "waiting_gpu"
        producer.state = "held"
        producer.save(update_fields=["state"])
        assert execution_status(jobs[1])["execution_state"] == "needs_attention"


def test_artifact_transfer_rechecks_disk_headroom_during_stream(tmp_path, monkeypatch):
    import hashlib
    import io
    import shutil
    from types import SimpleNamespace

    from api.execution.artifacts import ArtifactStore

    contents = b"x" * (2 * 1024**2)
    free = iter([10 * 1024**2, 4 * 1024**2, 1024**2])
    monkeypatch.setattr(shutil, "disk_usage", lambda path: SimpleNamespace(free=next(free)))
    store = ArtifactStore(tmp_path)
    checksum = hashlib.sha256(contents).hexdigest()
    with pytest.raises(OSError, match="during transfer"):
        store.publish(io.BytesIO(contents), checksum, len(contents), minimum_free=1024**2)
    assert not store.path(checksum).exists()
    assert not list(tmp_path.rglob(".transfer-*"))


def test_executor_stops_only_its_attempt_when_scratch_exceeds_reservation(tmp_path):
    import time

    from tools.executor.agent import Agent

    work = tmp_path / "attempts/test-attempt"
    work.mkdir(parents=True)
    (work / "temporary-output").write_bytes(b"x" * 128)
    agent = Agent({"root": str(tmp_path), "host_id": "test", "budgets": {"disk_headroom": 0}})
    calls = []

    def docker(*arguments):
        calls.append(arguments)
        return json.dumps([{"State": {"Running": True}}]) if arguments[0] == "inspect" else ""

    agent.docker = docker
    job = {
        "request": {"resources": {"disk": 64}},
        "container": "owned-container",
        "renewed": time.monotonic(),
    }
    try:
        agent.tick_job("test-attempt", job)
        assert job["abort_category"] == "disk"
        assert calls[-1] == ("stop", "-t", "5", "owned-container")
    finally:
        agent.transfers.shutdown(wait=True)


@pytest.mark.parametrize("renewal_error", [None, ConnectionError, ValueError])
def test_completed_transfer_requires_fresh_ownership_before_compute(tmp_path, renewal_error):
    import time
    from concurrent.futures import Future

    from tools.executor.agent import Agent

    agent = Agent({"root": str(tmp_path), "host_id": "test"})
    transfer = Future()
    transfer.set_result({})
    job = {
        "request": {"id": "ready-transfer", "token": "isolated-test"},
        "container": None,
        "transfer": transfer,
        "renewed": time.monotonic(),
    }
    events = []

    def renew(operation, data, attempt):
        events.append("renew")
        if renewal_error:
            raise renewal_error("Controller unavailable or ownership expired")

    agent.request = renew
    agent.launch = lambda request: events.append("launch")
    try:
        if renewal_error:
            with pytest.raises(renewal_error):
                agent.tick_job("ready-transfer", job)
            assert events == ["renew"]
        else:
            agent.tick_job("ready-transfer", job)
            assert events == ["renew", "launch"]
    finally:
        agent.transfers.shutdown(wait=True)


def test_expired_transfer_reconciles_after_drain_and_retries_lost_response(tmp_path):
    import time
    from concurrent.futures import Future

    from tools.executor.agent import Agent

    agent = Agent({"root": str(tmp_path), "host_id": "test"})
    transfer = Future()
    job = {
        "request": {"id": "expired-transfer", "token": "isolated-test"},
        "container": None,
        "transfer": transfer,
        "renewed": time.monotonic() - 31,
    }
    agent.jobs["expired-transfer"] = job
    events = []

    def reconcile(operation, data):
        assert operation == "reconcile" and data == {"running": []}
        events.append(operation)
        if len(events) == 1:
            raise ConnectionError("Acknowledgement lost")

    agent.request = reconcile
    agent.docker = lambda *args: pytest.fail("A transfer has no compute container to stop")
    agent.launch = lambda request: pytest.fail("Expired transfer must never launch compute")
    try:
        agent.expire_unrenewed()
        assert job["abandoned"]
        agent.tick_job("expired-transfer", job)
        assert not events and agent.jobs
        transfer.set_result({})
        with pytest.raises(ConnectionError):
            agent.tick_job("expired-transfer", job)
        assert agent.jobs
        agent.tick_job("expired-transfer", job)
        assert not agent.jobs and len(events) == 2
    finally:
        agent.transfers.shutdown(wait=True)


def test_runtime_digest_cache_deduplicates_hardlinks_and_rechecks_preserved_mtime(
    tmp_path, monkeypatch
):
    import os

    from tools.executor import agent as executor

    first = tmp_path / "weights.bin"
    first.write_bytes(b"original")
    second = tmp_path / "alias.bin"
    os.link(first, second)
    expected = executor.sha256_file(first)
    agent = executor.Agent(
        {
            "root": str(tmp_path / "agent"),
            "host_id": "test",
            "runtime_files": {str(first): expected, str(second): expected},
        }
    )
    original = executor.sha256_file
    calls = []

    def measure(path):
        calls.append(path)
        return original(path)

    monkeypatch.setattr(executor, "sha256_file", measure)
    try:
        agent.verify_runtime_files(threads=2)
        agent.verify_runtime_files(threads=2)
        assert len(calls) == 1
        saved = first.stat()
        first.write_bytes(b"modified")
        os.utime(first, ns=(saved.st_atime_ns, saved.st_mtime_ns))
        with pytest.raises(ValueError, match="checksum mismatch"):
            agent.verify_runtime_files()
    finally:
        agent.transfers.shutdown(wait=True)


def test_corrupt_runtime_digest_cache_reverts_to_full_verification(tmp_path):
    from tools.executor import agent as executor

    model = tmp_path / "weights.bin"
    model.write_bytes(b"original")
    agent = executor.Agent(
        {
            "root": str(tmp_path / "agent"),
            "host_id": "test",
            "runtime_files": {str(model): executor.sha256_file(model)},
        }
    )
    (agent.root / "model-validation.json").write_text("incomplete cache")
    try:
        assert str(model) in agent.verify_runtime_files()
        assert json.loads((agent.root / "model-validation.json").read_text())["version"] == 1
    finally:
        agent.transfers.shutdown(wait=True)


def test_capacity_lost_during_transfer_prevents_model_launch(tmp_path):
    from api.execution.errors import RecoverableExecutionError
    from tools.executor.agent import Agent

    image = "sha256:" + "a" * 64
    agent = Agent(
        {
            "root": str(tmp_path),
            "host_id": "test",
            "images": [image],
            "budgets": {"ram": 8 * GIB, "cpu": 4, "disk": 8 * GIB},
        }
    )
    request = {
        "id": "late-capacity",
        "image": image,
        "kind": "predict",
        "resources": {"ram": 2 * GIB, "cpu": 1, "disk": GIB},
    }
    agent.jobs[request["id"]] = {"request": request, "container": None}
    agent.observe = lambda: {"ram_available": GIB, "disk_free": 16 * GIB, "attempts": {}}
    calls = []
    agent.docker = lambda *args: calls.append(args)
    try:
        with pytest.raises(RecoverableExecutionError) as error:
            agent.launch(request)
        assert error.value.category == "waiting_ram"
        assert calls == []
    finally:
        agent.transfers.shutdown(wait=True)


def test_supervisor_classifies_child_cuda_oom_and_retains_its_log(tmp_path, monkeypatch):
    import subprocess
    import time

    from tools.executor.agent import Agent

    agent = Agent({"root": str(tmp_path), "host_id": "gpu", "budgets": {"disk_headroom": 0}})
    request = {"id": "cuda-attempt", "stage_id": "cuda-stage", "kind": "predict", "token": "test"}
    job = {"request": request, "container": "owned", "renewed": time.monotonic()}
    agent.jobs[request["id"]] = job
    agent.docker = lambda *args: json.dumps(
        [
            {
                "State": {
                    "Running": False,
                    "ExitCode": 1,
                    "OOMKilled": False,
                    "FinishedAt": "2026-10-28T12:00:00Z",
                }
            }
        ]
    )
    monkeypatch.setattr(
        subprocess, "check_output", lambda *args, **kwargs: "RuntimeError: CUDA out of memory\n"
    )
    reports = []
    agent.request = lambda operation, data, ident: reports.append((operation, data))
    try:
        agent.tick_job(request["id"], job)
        assert reports[-1][0] == "error"
        assert reports[-1][1]["category"] == "oom"
        assert (
            "CUDA out of memory" in (tmp_path / "attempts/cuda-attempt/container.log").read_text()
        )
        assert not agent.jobs
    finally:
        agent.transfers.shutdown(wait=True)


def test_restart_stops_owned_containers_before_contacting_unavailable_controller(
    tmp_path, monkeypatch
):
    from tools.executor import agent as executor

    class Reconciled(BaseException):
        pass

    request = {"id": "old-attempt", "stage_id": "stage"}
    work = tmp_path / "attempts/old-attempt"
    work.mkdir(parents=True)
    (work / "request.json").write_text(json.dumps(request))
    agent = executor.Agent(
        {"root": str(tmp_path), "host_id": "test", "capabilities": {}, "images": []}
    )
    calls = []
    agent.inventory = lambda: [
        {
            "Id": "owned",
            "Config": {"Labels": {"org.webkinpred.attempt": "old-attempt"}},
            "State": {"Running": True},
        }
    ]

    def docker(*args):
        calls.append(args[0])
        return json.dumps([{"State": {"FinishedAt": "confirmed"}}])

    def contact(operation, data):
        calls.append(operation)
        if operation == "register" and calls.count("register") == 1:
            raise ConnectionError("controller offline")
        if operation == "reconcile":
            assert data == {"running": [], "stopped": {"old-attempt": "confirmed"}}
            raise Reconciled()

    agent.docker = docker
    agent.request = contact
    agent.observe = lambda: {}
    monkeypatch.setattr(
        "tools.executor.runtime_identity.identity", lambda gpu: {"fingerprint": "x"}
    )
    monkeypatch.setattr(executor.time, "sleep", lambda seconds: calls.append("sleep"))
    try:
        with pytest.raises(Reconciled):
            agent.run()
        assert calls == ["stop", "inspect", "register", "sleep", "register", "reconcile"]
        assert (tmp_path / "checkpoints/stage.json").is_file()
    finally:
        agent.transfers.shutdown(wait=True)


def test_similarity_sequences_follow_row_offsets_across_variable_sized_chunks(tmp_path):
    from tools.executor.runner import selected_sequence_rows

    paths = []
    for chunk, (offset, values) in enumerate(
        [(0, ["MAAA"] * 3), (3, ["MCCC"] * 128), (131, ["MDDD"])]
    ):
        path = tmp_path / f"{chunk}.json"
        path.write_text(
            json.dumps(
                {
                    "target": "kcat",
                    "offset": offset,
                    "result": {"preds": [1] * len(values), "selected_sequences": values},
                }
            )
        )
        paths.append(path)
    actual = list(selected_sequence_rows(paths))
    assert len(actual) == 132
    assert actual[:3] == ["MAAA"] * 3
    assert actual[3:131] == ["MCCC"] * 128
    assert actual[131] == "MDDD"
    with pytest.raises(ValueError, match="missing or out of order"):
        list(selected_sequence_rows(paths[1:]))


@pytest.mark.django_db(transaction=True)
def test_v2_rejects_malformed_method_mapping_before_profile_lookup():
    from api.services.job_service import process_job_submission_from_params

    with override_settings(EXECUTION_V2_ENABLED=True, EXECUTION_MODEL_VERSIONS={}):
        response, job = process_job_submission_from_params(
            {"targets": ["kcat"], "methods": [], "handle_long_sequences": "skip"}, None, "127.0.0.1"
        )
    assert response.status_code == 400
    assert job is None


def test_molecular_input_rejection_is_distinct_from_resource_failure(monkeypatch, tmp_path):
    from tools.executor import policy

    reason = "Substrate is too long for CatRange's ChemBERTa encoder"
    np.save(tmp_path / "unsupported.npy", np.asarray([reason]))
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({"CatRange": [{"input": "CC", "path": "unsupported.npy"}]}))
    monkeypatch.setenv("WEBKINPRED_MOLECULAR_MANIFEST", str(manifest))
    monkeypatch.setenv("WEBKINPRED_MEDIA_ROOT", str(tmp_path))
    monkeypatch.setattr(policy, "_cursors", {})
    with pytest.raises(policy.DeterministicEmbeddingInputError, match="too long"):
        policy.molecular_vectors("CatRange", ["CC"])


def test_interrupted_embedding_attempt_restores_complete_arrays_and_binding_rows(tmp_path):
    from api.execution.artifacts import atomic_json
    from tools.executor.checkpoints import remember, restore

    old = tmp_path / "attempts/previous"
    new = tmp_path / "attempts/retry"
    (old / "media").mkdir(parents=True)
    request = {
        "id": "previous",
        "stage_id": "same-stage",
        "kind": "embedding",
        "method": "KinForm-H",
        "specification": {"pinned": "weights"},
        "payload": {
            "step": "kinform_pseq2sites",
            "seq_id_to_seq": {"sid": "MAAA", "later": "MCCC"},
            "outputs": [
                {"id": "vector", "identity": {"model": "pinned"}, "path": "vector.npy"},
                {"id": "binding", "identity": {"model": "binding"}, "path": "binding.tsv"},
            ],
        },
    }
    atomic_json(old / "request.json", request)
    np.save(old / "media/vector.npy", np.arange(8, dtype=np.float32))
    (old / "media/binding.tsv").write_text(
        "PDB\tPred_BS_Scores\nsid\t0.1,0.2,0.3,0.4\nlater\t0.1,0.2\n"
    )
    remember(tmp_path, request)
    restore(tmp_path, {**request, "id": "retry"}, new)
    np.testing.assert_array_equal(np.load(new / "media/vector.npy"), np.arange(8, dtype=np.float32))
    binding = (new / "media/binding.tsv").read_text()
    assert "sid\t0.1,0.2,0.3,0.4" in binding
    assert "later" not in binding  # Incomplete row is recomputed under the new lease.


@pytest.mark.django_db(transaction=True)
def test_profile_binding_requires_the_measured_model_identity():
    job = Job.objects.create(public_id="pinning", status="Pending", prediction_type="kcat")
    execution = control.ingest(
        job.public_id,
        {
            "targets": ["kcat"],
            "methods": {"kcat": "DLKcat"},
            "model_versions": {"DLKcat": {"weights": "old"}},
        },
        "a" * 64,
    )
    stage = WorkStage.objects.create(
        execution=execution,
        key="prediction",
        kind="predict",
        method="DLKcat",
        target="kcat",
        payload={"shape": {"units": 1}},
    )
    profile = ResourceProfile.objects.create(
        id="pin-profile",
        fingerprint="f" * 64,
        image="sha256:" + "a" * 64,
        stage_kind="predict",
        method="DLKcat",
        validated=True,
        settings={"model_identity": {"weights": "new"}},
        limits={"units": 1},
    )
    control.bind_profiles()
    stage.refresh_from_db()
    assert stage.profile_id is None
    profile.settings["model_identity"] = {"weights": "old"}
    profile.save()
    control.bind_profiles()
    stage.refresh_from_db()
    assert stage.profile_id == profile.pk


@pytest.mark.django_db(transaction=True)
def test_assembly_profiles_pin_pipeline_independently_of_requested_model_subset():
    pipeline = {"assembly_code": "a" * 64}
    profile = ResourceProfile.objects.create(
        id="assembly-version",
        fingerprint="f" * 64,
        image="sha256:" + "a" * 64,
        stage_kind="assemble",
        method="",
        validated=True,
        settings={"model_identity": pipeline},
        limits={"rows": 10000},
    )
    for index, methods in enumerate([{"kcat": "DLKcat"}, {"kcat": "DLKcat", "Km": "UniKP"}]):
        job = Job.objects.create(public_id=f"asmver{index}", prediction_type="kcat")
        execution = control.ingest(
            job.public_id,
            {
                "targets": list(methods),
                "methods": methods,
                "model_versions": {method: {"weights": method} for method in methods.values()},
                "execution_versions": {"assemble": pipeline},
            },
            "a" * 64,
        )
        stage = WorkStage.objects.create(
            execution=execution,
            key="assembly",
            kind="assemble",
            payload={"shape": {"rows": 1}},
        )
        control.bind_profiles()
        stage.refresh_from_db()
        assert stage.profile_id == profile.pk


@pytest.mark.django_db(transaction=True)
def test_postgres_rejects_mutation_of_a_frozen_request():
    from api.execution.models import ExecutionSpec
    from django.db import IntegrityError

    job = Job.objects.create(public_id="frozen1", prediction_type="kcat")
    execution = control.ingest(job.public_id, {"targets": ["kcat"]}, "a" * 64)
    with pytest.raises(IntegrityError, match="immutable"):
        ExecutionSpec.objects.filter(pk=execution.pk).update(specification={"targets": ["Km"]})
    execution.paused = False
    execution.save(update_fields=["paused"])


def test_similarity_missing_storage_never_becomes_a_negative_cache_entry(tmp_path, monkeypatch):
    from api.execution.context import StageContext, stage_context
    from api.services import similarity_service

    path = tmp_path / "output.csv"
    path.write_text("Protein Sequence,kcat (1/s)\nMAAA,1\n")
    monkeypatch.setattr(
        similarity_service,
        "_resolve_similarity_dataset_for_method",
        lambda method: ("test-dataset", str(tmp_path / "missing-database")),
    )
    context = StageContext("isolated")
    with stage_context(context), pytest.raises(FileNotFoundError, match="similarity database"):
        similarity_service.append_kcat_similarity_columns_to_output_csv(
            str(path), "DLKcat", recon_xkg=True
        )
    assert context.similarity_writes == []
    assert path.read_text() == "Protein Sequence,kcat (1/s)\nMAAA,1\n"


def test_similarity_search_respects_reserved_cpu_threads(monkeypatch):
    from api.utils.similarity_utils import _mmseqs_cmd

    monkeypatch.setenv("WEBKINPRED_EXECUTION_POLICY", "gpu-only")
    monkeypatch.setenv("OMP_NUM_THREADS", "2")
    assert _mmseqs_cmd("search", "input", "database")[-2:] == ["--threads", "2"]
    assert _mmseqs_cmd("convertalis", "input", "database")[-2:] == ["--threads", "2"]
    assert "--threads" not in _mmseqs_cmd("createdb", "input", "database")


def test_similarity_attempt_reuses_bounded_results_and_separates_databases(monkeypatch):
    from api.execution.context import StageContext, stage_context
    from api.services import similarity_service as service

    calls = []

    def compute(sequences, database, cleanup):
        calls.append((list(sequences), database))
        return ({s: 75.0 for s in sequences}, {s: 20.0 for s in sequences})

    monkeypatch.setattr(service, "_compute_mmseqs_similarity_uncached", compute)
    context = StageContext("isolated")
    with stage_context(context):
        first = service._compute_mmseqs_similarity(["MAAA"], "db-one", [])
        assert service._compute_mmseqs_similarity(["MAAA"], "db-one", []) == first
        service._compute_mmseqs_similarity(["MAAA"], "db-two", [])
        assert len(calls) == 2
        service._compute_mmseqs_similarity([str(i) for i in range(5000)], "db-one", [])
        assert len(context.similarity_recent) == 4096
    assert context.similarity_writes == []


def test_similarity_attempt_does_not_memoize_resource_failures(monkeypatch):
    from api.execution.context import StageContext, stage_context
    from api.services import similarity_service as service

    def failed(*args):
        raise MemoryError("isolated allocation failure")

    monkeypatch.setattr(service, "_compute_mmseqs_similarity_uncached", failed)
    context = StageContext("isolated")
    with stage_context(context), pytest.raises(MemoryError):
        service._compute_mmseqs_similarity(["MAAA"], "db", [])
    assert not context.similarity_recent


@pytest.mark.parametrize("running", [True, False])
def test_approved_image_keeper_is_idempotent_and_isolated(tmp_path, running):
    from tools.executor.agent import Agent

    image = "sha256:" + "a" * 64
    agent = Agent({"root": str(tmp_path), "host_id": "test"})
    calls = []

    def docker(*args):
        calls.append(args)
        if args[0] == "ps":
            return "existing"
        if args[0] == "inspect":
            return json.dumps(
                [
                    {
                        "Image": image,
                        "Config": {"Labels": {"org.webkinpred.image-keeper": "true"}},
                        "State": {"Running": running},
                    }
                ]
            )
        return ""

    agent.docker = docker
    try:
        agent.retain_image(image)
        assert not any(c[0] == "run" for c in calls)
        assert any(c[0] == "start" for c in calls) != running
        calls.clear()
        agent.docker = lambda *args: calls.append(args) or ""
        agent.retain_image(image)
        create = next(c for c in calls if c[0] == "run")
        assert "--read-only" in create and "none" in create
        assert "org.webkinpred.executor=" in create
        assert "com.docker.compose.project=webkinpred-image-keepers" in create
    finally:
        agent.transfers.shutdown(wait=True)


def test_docker_create_failure_preserves_missing_image_evidence(tmp_path):
    import subprocess

    from tools.executor.agent import Agent

    image = "sha256:" + "a" * 64
    agent = Agent(
        {
            "root": str(tmp_path),
            "host_id": "test",
            "images": [image],
            "budgets": {"ram": 8 * GIB, "cpu": 4, "disk": 8 * GIB},
        }
    )
    request = {
        "id": "missing-runtime",
        "kind": "assemble",
        "image": image,
        "resources": {"ram": GIB, "cpu": 1, "disk": GIB},
    }
    agent.observe = lambda: {"ram_available": 16 * GIB, "disk_free": 16 * GIB, "attempts": {}}
    original = subprocess.CalledProcessError(
        1, ["docker", "create"], stderr="No such image: " + image
    )

    def docker(*args):
        if args[0] == "create":
            raise original
        raise subprocess.CalledProcessError(1, ["docker", "inspect"], stderr="No such container")

    agent.docker = docker
    try:
        with pytest.raises(subprocess.CalledProcessError) as error:
            agent.launch(request)
        assert error.value is original
        assert agent.error_details(original, request)["message"].startswith("No such image")
        agent.config["token"] = "secret-control"
        error = RuntimeError("secret-control secret-owner")
        assert "secret-" not in agent.error_details(error, {"token": "secret-owner"})["message"]
    finally:
        agent.transfers.shutdown(wait=True)


@pytest.mark.parametrize("registered", [True, False])
def test_verified_local_inputs_do_not_repeat_location_writes(tmp_path, registered):
    import hashlib
    import io

    from tools.executor.agent import Agent

    agent = Agent({"root": str(tmp_path), "host_id": "test"})
    data = b"immutable embedding"
    checksum = hashlib.sha256(data).hexdigest()
    agent.store.publish(io.BytesIO(data), checksum, len(data))
    work = tmp_path / "attempts" / "test"
    (work / "media").mkdir(parents=True)
    artifact = {
        "id": "a" * 64,
        "path": "embedding.npy",
        "checksum": checksum,
        "size": len(data),
        "hosts": ["test"] if registered else [],
    }
    writes = []
    agent.artifact_location = lambda *args: writes.append(args)
    try:
        result = agent.materialize(
            {"id": "test", "kind": "predict", "stage_id": "stage", "artifacts": [artifact]}, work
        )
        assert (work / "media/embedding.npy").read_bytes() == data
        assert len(writes) == (0 if registered else 1)
        assert result["local_cache_hits"] == 1
    finally:
        agent.transfers.shutdown(wait=True)
