"""Strict background scheduling cannot outrank normal users or bypass reservations."""

import io
from datetime import timedelta

import pytest
from api.execution import control, status
from api.execution.background import foreground_dependencies, higher_preparation
from api.execution.models import (
    ExecutionAttempt,
    ExecutionSpec,
    ResourceProfile,
    SchedulingRule,
    WorkStage,
)
from api.execution.resources import GIB
from api.models import ApiKey, ApiUser, Job
from api.utils.quotas import api_key_quota_subject
from django.core.management import call_command
from django.test import override_settings
from django.utils import timezone
from tests.test_execution import ready_work as ready_work

pytestmark = pytest.mark.django_db(transaction=True)


def lower(execution):
    ExecutionSpec.objects.filter(pk=execution.pk).update(background=True)
    execution.refresh_from_db()


def policy():
    out = io.StringIO()
    call_command(
        "execution_background_priority",
        ip=["198.51.100.187"],
        api_key_prefix=["ak_1234abc"],
        stdout=out,
    )
    return out.getvalue()


@pytest.mark.parametrize("selector", ["ip", "key"])
def test_policy_classifies_existing_and_future_jobs_without_recharging_or_recreating(selector):
    user = ApiUser.objects.create(ip_address="192.0.2.16")
    key = ApiKey.objects.create(user=user, key="ak_1234abc" + "a" * 57, is_active=False)
    attrs = (
        {"ip_address": "198.51.100.187"}
        if selector == "ip"
        else {"quota_subject": api_key_quota_subject(key.pk), "ip_address": "192.0.2.99"}
    )
    old = Job.objects.create(
        public_id="existing", status="Pending", prediction_type="kcat", **attrs
    )
    before = control.ingest(old.public_id, {}, "a" * 64, paused=False)
    checksum = before.checksum
    normal = Job.objects.create(
        public_id="normal", status="Pending", prediction_type="kcat", ip_address="192.0.2.19"
    )
    control.ingest(normal.public_id, {}, "b" * 64, paused=False)
    completed = Job.objects.create(
        public_id="done", status="Completed", prediction_type="kcat", **attrs
    )
    done = control.ingest(completed.public_id, {}, "c" * 64)
    assert "background_unfinished" in policy()
    before.refresh_from_db()
    done.refresh_from_db()
    old.refresh_from_db()
    assert before.background and not done.background
    assert before.checksum == checksum and old.status == "Pending"
    assert not normal.execution.background
    future = Job.objects.create(
        public_id="future", status="Pending", prediction_type="kcat", **attrs
    )
    assert control.ingest(future.public_id, {}, "d" * 64, paused=False).background
    assert Job.objects.count() == 4
    assert policy() and ExecutionSpec.objects.filter(background=True).count() == 2


def test_normal_priority_zero_precedes_old_background_even_if_background_is_boosted(ready_work):
    host, stages = ready_work
    low, normal = stages
    lower(low.execution)
    ExecutionSpec.objects.filter(pk=low.execution_id).update(priority=1_000_000)
    WorkStage.objects.filter(pk=low.pk).update(priority=1_000_000)
    with override_settings(EXECUTION_V2_ENABLED=True):
        assert control.claim(host.pk)["stage_id"] == str(normal.pk)


def test_preemption_preserves_reservation_until_supervisor_confirms_stop(ready_work):
    host, stages = ready_work
    low, normal = stages
    lower(low.execution)
    normal.available_at = timezone.now() + timedelta(hours=1)
    normal.save(update_fields=["available_at"])
    low.checkpoint = {"completed_units": 7}
    low.save(update_fields=["checkpoint"])
    with override_settings(EXECUTION_V2_ENABLED=True):
        first = control.claim(host.pk)
        assert first["stage_id"] == str(low.pk)
        normal.available_at = timezone.now()
        normal.save(update_fields=["available_at"])
        assert control.claim(host.pk) is None
        attempt = ExecutionAttempt.objects.get(pk=first["id"])
        assert attempt.stop_requested and attempt.state == "leased"
        assert control.claim(host.pk) is None
        control.fail(
            host.pk,
            first["id"],
            {"token": first["token"], "stopped": True, "category": "interrupted"},
        )
        second = control.claim(host.pk)
        assert second["stage_id"] == str(normal.pk)
    low.refresh_from_db()
    assert low.state == "ready" and low.checkpoint == {"completed_units": 7}
    assert Job.objects.filter(status="Failed").count() == 0


def test_preemption_never_stops_a_normal_attempt(ready_work):
    host, stages = ready_work
    first, other = stages
    other.available_at = timezone.now() + timedelta(hours=1)
    other.save(update_fields=["available_at"])
    with override_settings(EXECUTION_V2_ENABLED=True):
        claimed = control.claim(host.pk)
        other.available_at = timezone.now()
        other.save(update_fields=["available_at"])
        assert control.claim(host.pk) is None
    assert not ExecutionAttempt.objects.get(pk=claimed["id"]).stop_requested


def test_background_can_use_spare_capacity_after_normal_claim(ready_work):
    host, stages = ready_work
    for s in stages:
        s.profile.resources["ram"] = GIB
        s.profile.save(update_fields=["resources"])
    lower(stages[0].execution)
    with override_settings(EXECUTION_V2_ENABLED=True):
        assert control.claim(host.pk)["stage_id"] == str(stages[1].pk)
        assert control.claim(host.pk)["stage_id"] == str(stages[0].pk)
    assert not ExecutionAttempt.objects.filter(stop_requested=True).exists()


def test_shared_background_prerequisites_inherit_foreground_recursively_and_live(ready_work):
    host, stages = ready_work
    low, normal = stages
    lower(low.execution)
    middle = WorkStage.objects.create(
        execution=low.execution, key="middle", kind="predict", profile=low.profile
    )
    deepest = WorkStage.objects.create(
        execution=low.execution, key="deepest", kind="predict", profile=low.profile
    )
    normal.dependencies.add(middle)
    middle.dependencies.add(deepest)
    assert foreground_dependencies() == {middle.pk, deepest.pk}
    with override_settings(EXECUTION_V2_ENABLED=True):
        assert control.claim(host.pk)["stage_id"] == str(deepest.pk)
    assert not ExecutionAttempt.objects.filter(stop_requested=True).exists()
    Job.objects.filter(pk=normal.execution.job_id).update(status="Completed")
    assert foreground_dependencies() == set()


def test_blocked_normal_dependency_does_not_prevent_background_work(ready_work):
    host, stages = ready_work
    low, normal = stages
    lower(low.execution)
    held = WorkStage.objects.create(
        execution=normal.execution, key="held", kind="prepare", state="held"
    )
    normal.dependencies.add(held)
    with override_settings(EXECUTION_V2_ENABLED=True):
        assert control.claim(host.pk)["stage_id"] == str(low.pk)


def test_normal_preparation_precedes_background_despite_numeric_boost(ready_work):
    _, stages = ready_work
    low, normal = stages
    lower(low.execution)
    ExecutionSpec.objects.filter(pk=low.execution_id).update(priority=1_000_000)
    for s in stages:
        WorkStage.objects.create(execution=s.execution, key="prepare", kind="prepare")
    assert higher_preparation(low.execution).filter(pk=normal.execution_id).exists()
    assert not higher_preparation(normal.execution).exists()


def test_queue_hint_puts_new_normal_ahead_of_old_background(ready_work, monkeypatch):
    _, stages = ready_work
    low, normal = stages
    lower(low.execution)
    monkeypatch.setattr(status, "_queue_snapshot", None)
    with override_settings(EXECUTION_V2_ENABLED=True):
        assert status.execution_status(normal.execution.job)["queue_estimate"]["position"] == 1
        data = status.execution_status(low.execution.job)
    assert data["queue_estimate"]["position"] == 2 and data["scheduling_class"] == "background"


def test_queue_tail_rank_respects_background_class(ready_work, monkeypatch):
    _, stages = ready_work
    low, normal = stages
    lower(low.execution)
    monkeypatch.setattr(status, "_queue_snapshot", None)
    monkeypatch.setattr(status, "_QUEUE_SNAPSHOT_LIMIT", 1)
    assert status.queue_estimate(low.execution.job, "waiting_cpu", background=True)["position"] == 2


def test_disabled_policy_restores_normal_class_and_invalid_selectors_do_not_mutate(ready_work):
    _, stages = ready_work
    Job.objects.filter(pk=stages[0].execution.job_id).update(ip_address="198.51.100.187")
    policy()
    stages[0].execution.refresh_from_db()
    assert stages[0].execution.background
    out = io.StringIO()
    call_command("execution_background_priority", ip=["198.51.100.187"], clear=True, stdout=out)
    stages[0].execution.refresh_from_db()
    assert not stages[0].execution.background
    from django.core.management.base import CommandError

    with pytest.raises(CommandError):
        call_command("execution_background_priority", ip=["invalid-ip"], stdout=out)
    assert SchedulingRule.objects.filter(enabled=True, kind="ip").count() == 0


def test_normal_gpu_waiter_preempts_background_gpu_without_overcommitting_vram(ready_work):
    host, stages = ready_work
    low, normal = stages
    lower(low.execution)
    profile = low.profile
    profile.resources = {"ram": 2 * GIB, "cpu": 2, "disk": GIB, "vram": 10 * GIB}
    profile.stage_kind = "embedding"
    profile.save()
    for s in stages:
        s.kind = "embedding"
        s.save(update_fields=["kind"])
    host.budgets.update(ram=32 * GIB, cpu=8, gpu_slots=3)
    host.telemetry.update(
        ram_available=64 * GIB, gpu_uuid="isolated-gpu", vram_total=20 * GIB, vram_free=18 * GIB
    )
    host.save()
    normal.available_at = timezone.now() + timedelta(hours=1)
    normal.save(update_fields=["available_at"])
    with override_settings(EXECUTION_V2_ENABLED=True):
        first = control.claim(host.pk)
        host.telemetry.update(
            vram_free=8 * GIB, attempts={first["id"]: {"vram": 10 * GIB, "ram": 2 * GIB}}
        )
        host.save(update_fields=["telemetry"])
        normal.available_at = timezone.now()
        normal.save(update_fields=["available_at"])
        assert control.claim(host.pk) is None
        a = ExecutionAttempt.objects.get(pk=first["id"])
        assert a.stop_requested and a.state == "leased"
        assert ExecutionAttempt.objects.filter(state__in=["leased", "running"]).count() == 1
        control.fail(
            host.pk,
            first["id"],
            {"token": first["token"], "stopped": True, "category": "interrupted"},
        )
        host.telemetry.update(vram_free=18 * GIB, attempts={})
        host.save(update_fields=["telemetry"])
        assert control.claim(host.pk)["stage_id"] == str(normal.pk)


def test_preemption_requests_only_the_smallest_useful_background_stop_set(ready_work):
    host, stages = ready_work
    for s in stages:
        lower(s.execution)
    p = stages[0].profile
    p.resources["ram"] = 2 * GIB
    p.save(update_fields=["resources"])
    j = Job.objects.create(public_id="normal3", status="Pending", prediction_type="kcat")
    e = control.ingest(j.public_id, {}, "a" * 64, paused=False)
    e.stages.all().delete()
    large = ResourceProfile.objects.create(
        id="large",
        fingerprint="d" * 64,
        image=p.image,
        stage_kind="predict",
        resources={**p.resources, "ram": 6 * GIB},
        validated=True,
    )
    host.capabilities["profiles"].append(large.fingerprint)
    host.save(update_fields=["capabilities"])
    normal = WorkStage.objects.create(
        execution=e,
        key="large",
        kind="predict",
        profile=large,
        available_at=timezone.now() + timedelta(hours=1),
    )
    with override_settings(EXECUTION_V2_ENABLED=True):
        assert control.claim(host.pk) is not None
        assert control.claim(host.pk) is not None
        normal.available_at = timezone.now()
        normal.save(update_fields=["available_at"])
        assert control.claim(host.pk) is None
    assert ExecutionAttempt.objects.filter(stop_requested=True).count() == 1
    assert ExecutionAttempt.objects.filter(state="leased").count() == 2


def test_unrunnable_normal_stage_does_not_idle_capacity_background_can_use(ready_work):
    host, stages = ready_work
    normal, low = stages
    lower(low.execution)
    normal.profile.resources["ram"] = 20 * GIB
    normal.profile.save(update_fields=["resources"])
    p = ResourceProfile.objects.create(
        id="fitting-low",
        fingerprint="b" * 64,
        image=normal.profile.image,
        stage_kind="predict",
        resources={"ram": GIB, "cpu": 1, "disk": GIB},
        validated=True,
    )
    low.profile = p
    low.save(update_fields=["profile"])
    host.capabilities["profiles"].append(p.fingerprint)
    host.save(update_fields=["capabilities"])
    with override_settings(EXECUTION_V2_ENABLED=True):
        assert control.claim(host.pk)["stage_id"] == str(low.pk)


def test_terminal_normal_job_cannot_block_background_admission(ready_work):
    host, stages = ready_work
    low, normal = stages
    lower(low.execution)
    Job.objects.filter(pk=normal.execution.job_id).update(status="Completed")
    with override_settings(EXECUTION_V2_ENABLED=True):
        assert control.claim(host.pk)["stage_id"] == str(low.pk)


def test_running_shared_prerequisite_is_not_preemptible_background(ready_work):
    host, stages = ready_work
    low, normal = stages
    lower(low.execution)
    shared = WorkStage.objects.create(
        execution=low.execution, key="shared-running", kind="predict", profile=low.profile
    )
    normal.dependencies.add(shared)
    with override_settings(EXECUTION_V2_ENABLED=True):
        claim = control.claim(host.pk)
        assert claim["stage_id"] == str(shared.pk)
        normal.dependencies.clear()
        another = WorkStage.objects.create(
            execution=normal.execution, key="another", kind="predict", profile=low.profile
        )
        another.dependencies.add(shared)
        assert control.claim(host.pk) is None
    assert not ExecutionAttempt.objects.get(pk=claim["id"]).stop_requested
