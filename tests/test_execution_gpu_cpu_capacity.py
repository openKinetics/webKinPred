from datetime import timedelta

import pytest
from api.execution import control
from api.execution.models import ExecutionAttempt, ResourceProfile, WorkStage
from api.execution.resources import GIB
from api.models import Job
from django.test import override_settings
from django.utils import timezone
from tests.test_execution import ready_work as ready_work

pytestmark = pytest.mark.django_db(transaction=True)


def saturated_cpu_queue(ready_work):
    host, (cpu, owner) = ready_work
    profile = cpu.profile
    profile.resources = {"cpu": 2, "ram": GIB, "disk": GIB}
    profile.save(update_fields=["resources"])
    for execution in (cpu.execution, owner.execution):
        execution.background = True
        execution.save(update_fields=["background"])
    Job.objects.filter(pk=cpu.execution.job_id).update(
        submission_time=timezone.now() - timedelta(days=1)
    )
    owner.state = "completed"
    owner.save(update_fields=["state"])
    for index in range(6):
        stage = WorkStage.objects.create(
            execution=owner.execution,
            key=f"busy-{index}",
            kind="predict",
            profile=profile,
            state="running",
        )
        ExecutionAttempt.objects.create(
            stage=stage,
            host=host,
            state="running",
            resources=profile.resources,
            profile_fingerprint=profile.fingerprint,
            lease_until=timezone.now() + timedelta(seconds=30),
        )
    gpu_profile = ResourceProfile.objects.create(
        id="gpu-profile",
        fingerprint="b" * 64,
        image="sha256:" + "b" * 64,
        stage_kind="embedding",
        validated=True,
        resources={"cpu": 2, "ram": GIB, "disk": GIB, "vram": 2 * GIB},
    )
    job = Job.objects.create(public_id="gpuqueue", prediction_type="kcat", status="Pending")
    execution = control.ingest(job.public_id, {}, "c" * 64, paused=False)
    execution.background = True
    execution.save(update_fields=["background"])
    gpu = WorkStage.objects.create(
        execution=execution,
        key="gpu",
        kind="embedding",
        profile=gpu_profile,
    )
    host.budgets = {"cpu": 16, "ram": 64 * GIB, "disk": 64 * GIB, "gpu_cpu_reserve": 4}
    host.telemetry = {
        "ram_available": 128 * GIB,
        "disk_free": 128 * GIB,
        "gpu_uuid": "test-gpu",
        "vram_total": 20 * GIB,
        "vram_free": 20 * GIB,
    }
    host.capabilities["profiles"].append(gpu_profile.fingerprint)
    host.save()
    return host, cpu, gpu


def test_cpu_backlog_leaves_admission_capacity_for_fitting_gpu_work(ready_work):
    host, cpu, gpu = saturated_cpu_queue(ready_work)
    with override_settings(EXECUTION_V2_ENABLED=True):
        assert control.claim(host.pk)["stage_id"] == str(gpu.pk)
    cpu.refresh_from_db()
    assert cpu.state == "ready" and cpu.waiting_reason == "waiting_cpu"


@pytest.mark.parametrize("unavailable", ["queue_empty", "vram"])
def test_cpu_can_use_remaining_capacity_when_no_gpu_work_can_run(ready_work, unavailable):
    host, cpu, gpu = saturated_cpu_queue(ready_work)
    if unavailable == "queue_empty":
        gpu.state = "completed"
        gpu.save(update_fields=["state"])
    else:
        host.telemetry["vram_free"] = GIB
        host.save(update_fields=["telemetry"])
    with override_settings(EXECUTION_V2_ENABLED=True):
        assert control.claim(host.pk)["stage_id"] == str(cpu.pk)


def test_foreground_cpu_work_retains_precedence_over_background_gpu_work(ready_work):
    host, cpu, _ = saturated_cpu_queue(ready_work)
    cpu.execution.background = False
    cpu.execution.save(update_fields=["background"])
    with override_settings(EXECUTION_V2_ENABLED=True):
        assert control.claim(host.pk)["stage_id"] == str(cpu.pk)
