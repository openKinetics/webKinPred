"""Peak CPU reservations may exceed spare CPUs by a bounded factor; nothing else is relaxed."""

from datetime import timedelta

import pytest
from api.execution.models import (
    Artifact,
    ExecutionAttempt,
    ExecutionSpec,
    WorkStage,
)
from api.execution.resources import GIB, admit
from api.models import Job
from django.test import override_settings
from django.utils import timezone
from tests.test_execution import ready_work as ready_work

TELEMETRY = {"ram_available": 64 * GIB, "disk_free": 64 * GIB, "cpu_count": 20, "cpu_external": 1}
BUDGETS = {"ram": 48 * GIB, "cpu": 40, "disk": 48 * GIB, "cpu_headroom": 1, "cpu_slots": 40}
CANDIDATE = {"ram": GIB, "cpu": 2, "disk": GIB, "vram": 0}


def running(cpu):
    return [{"resources": {"ram": GIB, "cpu": 2, "disk": GIB}, "observed": {}}] * (cpu // 2)


def test_default_keeps_reservations_within_spare_cpus():
    # 20 CPUs - 1 headroom - 1 external = 18 spare.
    assert admit(CANDIDATE, running(16), BUDGETS, TELEMETRY, gpu=False).fits
    assert admit(CANDIDATE, running(18), BUDGETS, TELEMETRY, gpu=False).reason == "waiting_cpu"


def test_overcommit_scales_only_spare_cpus():
    budgets = {**BUDGETS, "cpu_overcommit": 1.5}
    # 18 spare x 1.5 = 27 reservable CPUs.
    assert admit(CANDIDATE, running(24), budgets, TELEMETRY, gpu=False).fits
    assert admit(CANDIDATE, running(26), budgets, TELEMETRY, gpu=False).reason == "waiting_cpu"
    busy = {**TELEMETRY, "cpu_external": 10}  # 9 spare x 1.5 = 13.5
    assert admit(CANDIDATE, running(10), budgets, busy, gpu=False).fits
    assert admit(CANDIDATE, running(12), budgets, busy, gpu=False).reason == "waiting_cpu"


def test_overcommit_never_relaxes_the_reservation_budget_or_memory():
    budgets = {**BUDGETS, "cpu_overcommit": 2, "cpu": 20}
    assert admit(CANDIDATE, running(20), budgets, TELEMETRY, gpu=False).reason == "waiting_cpu"
    tight = {**TELEMETRY, "ram_available": 2 * GIB}
    assert admit(CANDIDATE, [], {**budgets, "ram_headroom": 2 * GIB}, tight, gpu=False).reason == (
        "waiting_ram"
    )


def test_exited_attempts_keep_disk_but_release_compute_reservations():
    gpu_telemetry = {
        **TELEMETRY,
        "gpu_uuid": "gpu",
        "vram_total": 20 * GIB,
        "vram_free": 12 * GIB,
        "ram_available": 20 * GIB,
    }
    budgets = {**BUDGETS, "cpu": 4, "ram": 12 * GIB, "disk": 10 * GIB, "gpu_slots": 1}
    publishing = {
        "resources": {"cpu": 4, "ram": 12 * GIB, "disk": 8 * GIB, "vram": 8 * GIB},
        "observed": {"released": True},
    }
    model = {"ram": GIB, "cpu": 2, "disk": GIB, "vram": 6 * GIB}
    # Its CPU, RAM, VRAM and the only GPU slot are free again...
    assert admit(model, [publishing], budgets, gpu_telemetry, gpu=True).fits
    computing = {**publishing, "observed": {}}
    assert not admit(model, [computing], budgets, gpu_telemetry, gpu=True).fits
    # ...but its outputs still occupy disk until publication completes.
    large = {**model, "disk": 4 * GIB}
    assert admit(large, [publishing], budgets, gpu_telemetry, gpu=True).reason == "waiting_disk"


@pytest.mark.parametrize("factor", [0.5, 5, True, float("nan"), "2"])
def test_invalid_overcommit_admits_nothing(factor):
    budgets = {**BUDGETS, "cpu_overcommit": factor}
    assert admit(CANDIDATE, [], budgets, TELEMETRY, gpu=False).reason == "invalid_budget"


@pytest.mark.django_db(transaction=True)
def test_status_counts_shared_producer_attempts_with_indexed_lookups(ready_work):
    from api.execution.planning import attach
    from api.execution.status import execution_status

    host, stages = ready_work
    now = timezone.now()
    execution = ExecutionSpec.objects.create(
        job=Job.objects.create(public_id="shared1", prediction_type="kcat", status="Processing"),
        specification={},
        checksum="c" * 64,
        input_checksum="d" * 64,
        created_at=now - timedelta(seconds=300),
        paused=False,
    )
    consumer = WorkStage.objects.create(execution=execution, key="predict", kind="predict")
    # A producer owned by another request (for example a shared protein batch).
    producer = stages[0]
    artifact = Artifact.objects.create(
        id="e" * 64, identity={}, checksum="f" * 64, size=1, producer=producer
    )
    # Attaching an input links its consumer to the producer stage.
    attach(consumer, artifact, "x.npy")
    ExecutionAttempt.objects.create(
        stage=producer,
        host=host,
        state="completed",
        started_at=now - timedelta(seconds=200),
        lease_until=now,
        finished_at=now - timedelta(seconds=150),
    )
    ExecutionAttempt.objects.create(
        stage=consumer,
        host=host,
        state="completed",
        started_at=now - timedelta(seconds=100),
        lease_until=now,
        finished_at=now - timedelta(seconds=80),
    )
    with override_settings(EXECUTION_V2_ENABLED=True):
        state = execution_status(execution.job)
    assert state["active_execution_seconds"] == 70


@pytest.mark.django_db(transaction=True)
def test_status_reads_input_producers_from_the_stage_graph_not_every_artifact(ready_work):
    from api.execution.status import execution_status
    from django.db import connection
    from django.test.utils import CaptureQueriesContext

    _, stages = ready_work
    execution = stages[0].execution
    with override_settings(EXECUTION_V2_ENABLED=True), CaptureQueriesContext(connection) as seen:
        execution_status(execution.job)
    assert not any(
        '"api_artifactdependency"' in query["sql"] and '"api_artifact"' in query["sql"]
        for query in seen
    )
