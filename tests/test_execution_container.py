"""Opt-in real Docker/PG rehearsal; no model weights, production state, or GPU."""

import json
import os
import subprocess
import time

import pytest
from api.execution import control
from api.execution.artifacts import sha256_file
from api.execution.context import StageContext, stage_context
from api.execution.models import ExecutorHost, ResourceProfile
from api.execution.planning import prepare
from api.execution.resources import GIB
from api.models import Job
from api.services.prediction_batch_service import build_sequence_batch_plan, build_target_batch_plan
from api.services.prediction_store import build_unit_keys
from django.test import override_settings
from django.utils import timezone
from tests.test_execution import ready_work  # noqa: F401
from tools.executor.agent import Agent

pytestmark = [
    pytest.mark.django_db(transaction=True),
    pytest.mark.skipif(
        not os.environ.get("WEBKINPRED_TEST_EXECUTOR_IMAGE"),
        reason="Set a locally built immutable test runtime image",
    ),
]


def test_real_cpu_container_checkpoints_and_final_publication(tmp_path):
    import pandas as pd
    from api.methods.registry import get

    image = os.environ["WEBKINPRED_TEST_EXECUTOR_IMAGE"]
    config = {
        "host_id": "production",
        "token": "isolated-test",
        "root": str(tmp_path),
        "control_url": "http://unused",
        "images": [image],
        "peers": {},
        "budgets": {"ram": 2 * GIB, "cpu": 2, "disk": 4 * GIB, "disk_headroom": 0},
    }
    host = ExecutorHost.objects.create(
        id="production",
        paused=False,
        reconciled=True,
        heartbeat_at=timezone.now(),
        budgets=config["budgets"],
        telemetry={"ram_available": 4 * GIB, "disk_free": 8 * GIB},
        capabilities={"profiles": ["p", "a"]},
    )
    prediction_profile = ResourceProfile.objects.create(
        id="test-real-predict",
        stage_kind="predict",
        method="DLKcat",
        image=image,
        fingerprint="p",
        validated=True,
        resources={"ram": GIB, "cpu": 1, "disk": GIB, "pids": 128},
    )
    assembly_profile = ResourceProfile.objects.create(
        id="test-real-assemble",
        stage_kind="assemble",
        image=image,
        fingerprint="a",
        validated=True,
        resources=prediction_profile.resources,
    )
    job = Job.objects.create(
        public_id="realcpu1", prediction_type="kcat", status="Pending", requested_rows=2
    )
    media = tmp_path / "media"
    source = media / "jobs/realcpu1/input.csv"
    source.parent.mkdir(parents=True)
    source.write_text("Protein Sequence,Substrate\nMAAA,CC\nMCCC,O\n")
    spec = {
        "targets": ["kcat"],
        "methods": {"kcat": "DLKcat"},
        "model_versions": {"DLKcat": {"weights": "test"}},
        "canonicalize_substrates": True,
        "handle_long_sequences": "truncate",
        "include_similarity_columns": False,
        "recon_xkg": True,
    }
    execution = control.ingest(job.public_id, spec, sha256_file(source), paused=False)
    prepare(execution, media, tmp_path / "blobs")
    stage = execution.stages.get(kind="predict")
    frame = pd.read_csv(source)
    descriptor = get("DLKcat")
    batch = build_target_batch_plan(
        descriptor, "kcat", frame, build_sequence_batch_plan(frame, [descriptor], "truncate")
    )
    with stage_context(
        StageContext(job.public_id, cache_namespace=stage.payload["cache_namespace"])
    ):
        keys, _, _ = build_unit_keys(descriptor, "kcat", batch.sequences, batch.call_kwargs, True)
    stage.payload["cache_snapshot"] = dict(zip(keys, [3.0, 4.0], strict=True))
    stage.profile = prediction_profile
    stage.save()
    execution.stages.filter(kind="assemble").update(profile=assembly_profile)
    agent = Agent(config)

    def request(operation, data=None, attempt=None):
        return (
            getattr(
                control,
                {
                    "complete": "finish",
                    "error": "fail",
                    "artifact-location": "artifact_location",
                }.get(operation, operation),
            )(host.pk, attempt, data)
            if attempt
            else getattr(control, operation)(host.pk, data)
        )

    agent.request = request
    try:
        with override_settings(
            EXECUTION_V2_ENABLED=True,
            MEDIA_ROOT=str(media),
            EXECUTION_BLOB_ROOT=str(tmp_path / "blobs"),
        ):
            for _ in range(2):
                host.heartbeat_at = timezone.now()
                host.save(update_fields=["heartbeat_at"])
                attempt = control.claim(host.pk)
                assert attempt is not None
                agent.stage(attempt)
                deadline = time.monotonic() + 90
                while agent.jobs and time.monotonic() < deadline:
                    agent.tick_jobs()
                    time.sleep(0.1)
                if agent.jobs:
                    pytest.fail("Container supervision deadline exceeded")
                assert execution.stages.get(pk=attempt["stage_id"]).state == "completed"
        job.refresh_from_db()
        assert job.status == "Completed"
        output = pd.read_csv(media / job.output_file.name)
        assert output["Protein Sequence"].tolist() == ["MAAA", "MCCC"]
        assert output["kcat (1/s)"].tolist() == [3.0, 4.0]
    finally:
        agent.transfers.shutdown(wait=True)
        for info in agent.inventory():
            subprocess.run(["docker", "rm", "-f", info["Id"]], capture_output=True)


def test_container_oom_is_observed_and_requeued(ready_work, tmp_path):  # noqa: F811
    image = os.environ["WEBKINPRED_TEST_EXECUTOR_IMAGE"]
    with override_settings(EXECUTION_V2_ENABLED=True):
        attempt = control.claim("test")
    # Fault injection is contained by a 64 MiB cgroup; the host is never exhausted.
    container = subprocess.check_output(
        [
            "docker",
            "run",
            "-d",
            "--memory=64m",
            "--memory-swap=64m",
            "--cpus=0.5",
            "--network=none",
            image,
            "python",
            "-c",
            "allocation = bytearray(256 * 1024 * 1024)",
        ],
        text=True,
    ).strip()
    agent = Agent({"root": str(tmp_path), "host_id": "test", "budgets": {"disk_headroom": 0}})
    agent.jobs[attempt["id"]] = {
        "request": attempt,
        "container": container,
        "renewed": time.monotonic(),
    }
    observed = {}

    def request(operation, data, ident):
        if operation == "error":
            observed.update(
                json.loads(subprocess.check_output(["docker", "inspect", container]))[0]
            )
        return getattr(control, {"error": "fail"}.get(operation, operation))("test", ident, data)

    agent.request = request
    try:
        deadline = time.monotonic() + 30
        while agent.jobs and time.monotonic() < deadline:
            agent.tick_jobs()
            time.sleep(0.1)
        assert not agent.jobs
        assert observed["State"]["OOMKilled"]
        stage = ready_work[1][0]
        stage.refresh_from_db()
        assert stage.state == "ready"
        assert stage.waiting_reason == "oom"
        assert stage.execution.job.status != "Failed"
    finally:
        agent.transfers.shutdown(wait=True)
        subprocess.run(["docker", "rm", "-f", container], capture_output=True)


def test_legacy_fence_ignores_managed_attempts_and_probes_with_inherited_labels(tmp_path):
    import uuid

    prefix = "wk-fence-test-" + uuid.uuid4().hex
    image = os.environ["WEBKINPRED_TEST_EXECUTOR_IMAGE"]
    names = []
    agent = Agent({"host_id": "fence", "root": str(tmp_path), "legacy_project": prefix})
    try:
        for suffix, labels in [
            ("attempt", ["org.webkinpred.executor=fence"]),
            ("probe", ["org.webkinpred.profiling=true"]),
            ("legacy", []),
        ]:
            name = prefix + "-" + suffix
            command = [
                "docker",
                "run",
                "-d",
                "--name",
                name,
                "--network",
                "none",
                "--memory",
                "64m",
                "--cpus",
                "0.1",
            ]
            for label in [
                "com.docker.compose.project=" + prefix,
                "com.docker.compose.service=celery",
                *labels,
            ]:
                command.extend(["--label", label])
            subprocess.run(
                command + [image, "python", "-c", "import time;time.sleep(60)"],
                check=True,
                capture_output=True,
            )
            names.append(name)
        assert agent.observe()["legacy_consumers"] == ["celery"]
        subprocess.run(["docker", "stop", "-t", "1", names[-1]], check=True, capture_output=True)
        assert agent.observe()["legacy_consumers"] == []
    finally:
        agent.transfers.shutdown(wait=True)
        for name in names:
            subprocess.run(["docker", "rm", "-f", name], check=True, capture_output=True)
