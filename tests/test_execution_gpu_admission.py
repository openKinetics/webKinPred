from concurrent.futures import ThreadPoolExecutor

import pytest
from api.execution import control
from api.execution.models import ExecutionAttempt, ResourceProfile, WorkStage
from api.execution.resources import GIB, LIVE_STATES
from django.db import close_old_connections
from django.test import override_settings
from tests.test_execution import ready_work as ready_work


def configure_gpu_work(ready_work, *, vram=3 * GIB, free=10 * GIB):
    host, stages = ready_work
    first = stages[0].profile
    first.stage_kind, first.method = "embedding", "encoder-a"
    first.resources = {"cpu": 1, "ram": GIB, "disk": GIB, "vram": vram}
    first.save()
    second = ResourceProfile.objects.create(
        id="other-gpu",
        fingerprint="g" * 64,
        image=first.image,
        stage_kind="embedding",
        method="encoder-b",
        resources=first.resources,
        validated=True,
    )
    for stage, profile in zip(stages, [first, second], strict=True):
        stage.kind, stage.method, stage.profile = "embedding", profile.method, profile
        stage.save()
    host.telemetry.update(gpu_uuid="gpu", vram_total=20 * GIB, vram_free=free)
    host.budgets["gpu_slots"] = 3
    host.capabilities["profiles"] = [first.fingerprint, second.fingerprint]
    host.save()
    return host, stages


@pytest.mark.django_db(transaction=True)
@pytest.mark.parametrize("historical_mixtures", [None, [], [["unrelated", "old-profile"]]])
def test_mixed_gpu_admission_uses_peak_reservations_not_combination_metadata(
    ready_work, historical_mixtures
):
    host, stages = configure_gpu_work(ready_work)
    if historical_mixtures is not None:
        host.capabilities["validated_gpu_combinations"] = historical_mixtures
        host.save()
    third = WorkStage.objects.create(
        execution=stages[0].execution,
        key="third-gpu",
        kind="embedding",
        method=stages[0].method,
        profile=stages[0].profile,
    )
    with override_settings(EXECUTION_V2_ENABLED=True):
        first, second = control.claim(host.pk), control.claim(host.pk)
        assert {first["stage_id"], second["stage_id"]} == {str(s.pk) for s in stages}
        # Neither model has allocated memory yet. Their future peaks plus GPU
        # headroom must prevent admitting a third, even with ten GiB still free.
        assert control.claim(host.pk) is None
    assert ExecutionAttempt.objects.filter(state__in=LIVE_STATES).count() == 2
    assert sum(a.resources["vram"] for a in ExecutionAttempt.objects.all()) == 6 * GIB
    third.refresh_from_db()
    assert third.state == "ready" and not third.profile.suspended
    cpu = ResourceProfile.objects.create(
        id="cpu-head",
        fingerprint="c" * 64,
        image=stages[0].profile.image,
        stage_kind="predict",
        validated=True,
        resources={"cpu": 1, "ram": GIB, "disk": GIB, "vram": 0},
    )
    head = WorkStage.objects.create(
        execution=stages[1].execution, key="cpu-head", kind="predict", profile=cpu
    )
    host.capabilities["profiles"].append(cpu.fingerprint)
    host.save(update_fields=["capabilities"])
    with override_settings(EXECUTION_V2_ENABLED=True):
        assert control.claim(host.pk)["stage_id"] == str(head.pk)
    assert ExecutionAttempt.objects.filter(state__in=LIVE_STATES).count() == 3


@pytest.mark.django_db(transaction=True)
def test_concurrent_gpu_claims_cannot_overcommit_model_loading_peaks(ready_work):
    host, stages = configure_gpu_work(ready_work, vram=4 * GIB, free=8 * GIB)

    def claim():
        close_old_connections()
        try:
            return control.claim(host.pk)
        finally:
            close_old_connections()

    with override_settings(EXECUTION_V2_ENABLED=True):
        with ThreadPoolExecutor(max_workers=2) as pool:
            replies = list(pool.map(lambda _: claim(), range(2)))
        assert sum(reply is not None for reply in replies) == 1
        assert ExecutionAttempt.objects.filter(state__in=LIVE_STATES).count() == 1
        # More actual capacity permits the other model without a mixture entry.
        host.telemetry["vram_free"] = 12 * GIB
        host.save(update_fields=["telemetry"])
        assert control.claim(host.pk) is not None
    assert ExecutionAttempt.objects.filter(state__in=LIVE_STATES).count() == 2


@pytest.mark.django_db(transaction=True)
def test_removing_mixture_rules_does_not_admit_unvalidated_individual_profiles(ready_work):
    host, stages = configure_gpu_work(ready_work)
    ResourceProfile.objects.filter(pk=stages[1].profile_id).update(validated=False)
    with override_settings(EXECUTION_V2_ENABLED=True):
        assert control.claim(host.pk)["stage_id"] == str(stages[0].pk)
        assert control.claim(host.pk) is None
    stages[1].refresh_from_db()
    assert stages[1].state == "ready" and not stages[1].attempts.exists()
