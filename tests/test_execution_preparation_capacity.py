from types import SimpleNamespace

import pytest
from api.execution import control
from api.execution.models import ExecutionSpec, WorkStage
from api.execution.preparation import candidate_rules, cpu_only, waiting_reason
from api.execution.resources import GIB
from api.models import Job
from django.test import override_settings


def execution(ident, *, background, prepared=False, status="Pending"):
    job = Job.objects.create(public_id=ident, status=status, prediction_type="kcat")
    spec = control.ingest(job.public_id, {}, "a" * 64, paused=False)
    spec.background = background
    spec.save(update_fields=["background"])
    if prepared:
        WorkStage.objects.filter(execution=spec, key="prepare").update(state="completed")
    return spec


@pytest.mark.django_db(transaction=True)
def test_background_preparation_preserves_compute_space_and_recovers(tmp_path, monkeypatch):
    from api.execution import preparation

    background = execution("capacity1", background=True)
    normal = execution("capacity2", background=False)
    free = 16 * GIB
    monkeypatch.setattr(preparation.shutil, "disk_usage", lambda path: SimpleNamespace(free=free))
    with override_settings(
        EXECUTION_HOSTS={"production": {"budgets": {"disk_headroom": 15 * GIB}}},
        EXECUTION_PREPARATION_RESERVE_BYTES=2 * GIB,
    ):
        assert waiting_reason(background, tmp_path) == "waiting_disk"
        assert waiting_reason(normal, tmp_path) == ""
        free = 18 * GIB
        assert waiting_reason(background, tmp_path) == ""
    assert background.stages.get(key="prepare").state == "ready"


@pytest.mark.django_db(transaction=True)
def test_prepared_window_drains_without_reclassifying_or_blocking_normal_jobs(tmp_path):
    old = execution("window1", background=True, prepared=True)
    queued = execution("window2", background=True)
    normal = execution("window3", background=False)
    completed = execution("window4", background=True, prepared=True, status="Completed")
    paused = execution("window5", background=True, prepared=True)
    paused.paused = True
    paused.save(update_fields=["paused"])
    attention = execution("window6", background=True, prepared=True)
    WorkStage.objects.create(
        execution=attention, key="profile-hold", kind="predict", waiting_reason="profiling_required"
    )
    held = execution("window7", background=True, prepared=True)
    WorkStage.objects.create(execution=held, key="held", kind="predict", state="held")
    with override_settings(
        EXECUTION_HOSTS={},
        EXECUTION_PREPARATION_RESERVE_BYTES=0,
        EXECUTION_PREPARED_BACKGROUND_LIMIT=1,
    ):
        assert waiting_reason(queued, tmp_path) == "preparation_backlog"
        assert waiting_reason(normal, tmp_path) == ""
        assert waiting_reason(old, tmp_path) == ""
        old.job.status = "Completed"
        old.job.save(update_fields=["status"])
        assert waiting_reason(queued, tmp_path) == ""
    queued.refresh_from_db()
    assert queued.background
    assert queued.stages.get(key="prepare").state == "ready"
    completed.job.refresh_from_db()
    assert completed.job.status == "Completed"


def cpu_execution(ident, *, prepared=False, methods=None, targets=None):
    job = Job.objects.create(
        public_id=ident, status="Pending", prediction_type="kcat", kcat_method="DLKcat"
    )
    spec = control.ingest(
        job.public_id,
        {"targets": targets or ["kcat"], "methods": methods or {"kcat": "DLKcat"}},
        "b" * 64,
        paused=False,
    )
    spec.background = True
    spec.save(update_fields=["background"])
    if prepared:
        spec.stages.filter(key="prepare").update(state="completed")
    return spec


@pytest.mark.django_db(transaction=True)
def test_gpu_backlog_does_not_hide_independent_cpu_preparation(tmp_path):
    gpu = execution("gpupool", background=True, prepared=True)
    blocked = execution("gpuqueued", background=True)
    cpu = cpu_execution("cpuqueued")
    normal = execution("normalcpu", background=False)
    with override_settings(
        EXECUTION_HOSTS={},
        EXECUTION_PREPARATION_RESERVE_BYTES=0,
        EXECUTION_PREPARED_BACKGROUND_LIMIT=1,
        EXECUTION_PREPARED_CPU_LIMIT=2,
    ):
        assert waiting_reason(blocked, tmp_path) == "preparation_backlog"
        assert waiting_reason(cpu, tmp_path) == ""
        eligible = set(ExecutionSpec.objects.filter(candidate_rules()).values_list("pk", flat=True))
        assert cpu.pk in eligible and normal.pk in eligible
        assert gpu.pk not in eligible and blocked.pk not in eligible


@pytest.mark.django_db(transaction=True)
def test_cpu_window_is_bounded_and_turns_over_after_completion(tmp_path):
    busy = cpu_execution("cpubusy", prepared=True)
    queued = cpu_execution("cpunext")
    gpu = execution("gpunext", background=True)
    with override_settings(
        EXECUTION_HOSTS={},
        EXECUTION_PREPARATION_RESERVE_BYTES=0,
        EXECUTION_PREPARED_BACKGROUND_LIMIT=1,
        EXECUTION_PREPARED_CPU_LIMIT=1,
    ):
        assert waiting_reason(queued, tmp_path) == "preparation_backlog"
        assert waiting_reason(gpu, tmp_path) == ""
        assert not ExecutionSpec.objects.filter(candidate_rules(), pk=queued.pk).exists()
        assert ExecutionSpec.objects.filter(candidate_rules(), pk=gpu.pk).exists()
        busy.job.status = "Completed"
        busy.job.save(update_fields=["status"])
        assert waiting_reason(queued, tmp_path) == ""


@pytest.mark.django_db(transaction=True)
def test_cpu_lookahead_still_preserves_disk_headroom(tmp_path, monkeypatch):
    from api.execution import preparation

    cpu = cpu_execution("cpudisk")
    monkeypatch.setattr(
        preparation.shutil, "disk_usage", lambda path: SimpleNamespace(free=16 * GIB)
    )
    with override_settings(
        EXECUTION_HOSTS={"production": {"budgets": {"disk_headroom": 15 * GIB}}},
        EXECUTION_PREPARATION_RESERVE_BYTES=2 * GIB,
    ):
        assert waiting_reason(cpu, tmp_path) == "waiting_disk"


@pytest.mark.django_db(transaction=True)
@pytest.mark.parametrize("mixed", [False, True])
def test_cpu_class_requires_immutable_cpu_only_method_choices(tmp_path, mixed):
    execution("gpufull", background=True, prepared=True)
    methods = {"kcat": "DLKcat", "Km": "EITLEM"} if mixed else {"kcat": "EITLEM"}
    queued = cpu_execution(
        "mixedcpu", methods=methods, targets=["kcat", "Km"] if mixed else ["kcat"]
    )
    with override_settings(
        EXECUTION_HOSTS={},
        EXECUTION_PREPARATION_RESERVE_BYTES=0,
        EXECUTION_PREPARED_BACKGROUND_LIMIT=1,
        EXECUTION_PREPARED_CPU_LIMIT=2,
    ):
        assert not cpu_only(queued)
        assert waiting_reason(queued, tmp_path) == "preparation_backlog"
