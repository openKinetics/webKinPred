"""One claim may admit several stages; each admission sees the batch's reservations."""

import json
from datetime import timedelta

import pytest
from api.execution import control
from api.execution.models import (
    ExecutionAttempt,
    ExecutionSpec,
    ExecutorHost,
    ResourceProfile,
    WorkStage,
)
from api.execution.resources import GIB, LIVE_STATES
from api.execution.views import executor
from django.test import RequestFactory, override_settings
from django.utils import timezone
from tests.test_execution import ready_work as ready_work
from tests.test_execution_gpu_admission import configure_gpu_work


def small_cpu_work(ready_work, *, stages=3, cpu_budget=2):
    host, existing = ready_work
    profile = existing[0].profile
    profile.resources = {"ram": GIB, "cpu": 1, "disk": GIB}
    profile.save(update_fields=["resources"])
    for n in range(len(existing), stages):
        existing.append(
            WorkStage.objects.create(
                execution=existing[n % 2].execution, key=f"cpu-{n}", kind="predict", profile=profile
            )
        )
    host.budgets["cpu"] = cpu_budget
    host.save(update_fields=["budgets"])
    return host, existing


@pytest.mark.django_db(transaction=True)
def test_one_claim_admits_every_fitting_stage_against_a_shared_ledger(ready_work):
    host, stages = small_cpu_work(ready_work)
    with override_settings(EXECUTION_V2_ENABLED=True):
        admitted = control.claim_batch(host.pk, 8)
        assert len(admitted) == 2
        assert len({a["stage_id"] for a in admitted}) == 2
        # The two admissions already use the whole CPU budget.
        assert control.claim_batch(host.pk, 8) == []
    live = ExecutionAttempt.objects.filter(state__in=LIVE_STATES)
    assert sum(a.resources["cpu"] for a in live) == 2
    waiting = WorkStage.objects.exclude(pk__in=[a["stage_id"] for a in admitted]).get(
        pk__in=[s.pk for s in stages]
    )
    assert waiting.state == "ready" and not waiting.attempts.exists()


@pytest.mark.django_db(transaction=True)
def test_batch_limit_is_honoured_and_single_claim_keeps_its_reply(ready_work):
    host, _ = small_cpu_work(ready_work, cpu_budget=4)
    with override_settings(EXECUTION_V2_ENABLED=True):
        assert len(control.claim_batch(host.pk, 2)) == 2
        single = control.claim(host.pk)
        assert isinstance(single, dict) and single["kind"] == "predict"
        assert control.claim(host.pk) is None


@pytest.mark.django_db(transaction=True)
def test_batched_gpu_admission_counts_unspent_peaks_of_the_same_batch(ready_work):
    host, stages = configure_gpu_work(ready_work)
    WorkStage.objects.create(
        execution=stages[0].execution,
        key="third-gpu",
        kind="embedding",
        method=stages[0].method,
        profile=stages[0].profile,
    )
    with override_settings(EXECUTION_V2_ENABLED=True):
        admitted = control.claim_batch(host.pk, 8)
    # Exactly what two sequential claims admit: a third 3 GiB model plus the
    # two unloaded peaks and the 2 GiB guard would exceed 10 GiB free.
    assert {a["stage_id"] for a in admitted} == {str(s.pk) for s in stages}
    assert sum(a["resources"]["vram"] for a in admitted) == 6 * GIB


@pytest.mark.django_db(transaction=True)
def test_gpu_work_is_examined_before_older_cpu_work_in_the_same_class(ready_work):
    host, stages = configure_gpu_work(ready_work)
    cpu = ResourceProfile.objects.create(
        id="cpu-old",
        fingerprint="c" * 64,
        image=stages[0].profile.image,
        stage_kind="predict",
        validated=True,
        resources={"cpu": 1, "ram": GIB, "disk": GIB, "vram": 0},
    )
    # Belongs to the oldest request, so it ranks first by age.
    older_cpu = WorkStage.objects.create(
        execution=stages[0].execution, key="cpu-old", kind="predict", profile=cpu
    )
    WorkStage.objects.filter(pk=older_cpu.pk).update(created_at=timezone.now() - timedelta(hours=1))
    host.capabilities["profiles"].append(cpu.fingerprint)
    host.save(update_fields=["capabilities"])
    with override_settings(EXECUTION_V2_ENABLED=True):
        first = control.claim_batch(host.pk, 1)
        assert first[0]["resources"]["vram"] > 0
        rest = control.claim_batch(host.pk, 8)
    assert str(older_cpu.pk) in {a["stage_id"] for a in rest}


@pytest.mark.django_db(transaction=True)
def test_gpu_cpu_share_is_carved_from_currently_reservable_cpus(ready_work):
    host, stages = configure_gpu_work(ready_work)
    cpu = ResourceProfile.objects.create(
        id="cpu-many",
        fingerprint="c" * 64,
        image=stages[0].profile.image,
        stage_kind="predict",
        validated=True,
        resources={"cpu": 2, "ram": GIB, "disk": GIB, "vram": 0},
    )
    for n in range(6):
        WorkStage.objects.create(
            execution=stages[n % 2].execution, key=f"cpu-{n}", kind="predict", profile=cpu
        )
    host.capabilities["profiles"].append(cpu.fingerprint)
    # A 30-CPU nominal budget, but other load leaves (20 - 1 - 9) x 1.2 = 12
    # reservable CPUs. CPU-only background work may use 12 - 6 = 6 of them.
    host.budgets.update(
        cpu=30, ram=64 * GIB, disk=64 * GIB, cpu_headroom=1, cpu_overcommit=1.2, gpu_cpu_reserve=6
    )
    host.telemetry.update(cpu_count=20, cpu_external=9, ram_available=128 * GIB)
    host.save()
    # The GPU share protects background admissions; normal work keeps full rights.
    ExecutionSpec.objects.filter(pk__in=[s.execution_id for s in stages]).update(background=True)
    with override_settings(EXECUTION_V2_ENABLED=True):
        admitted = control.claim_batch(host.pk, 8)
    cpu_only = [a for a in admitted if not a["resources"]["vram"]]
    assert sum(a["resources"]["cpu"] for a in cpu_only) <= 6
    assert any(a["resources"]["vram"] for a in admitted)


@pytest.mark.django_db(transaction=True)
def test_production_serves_final_assembly_before_older_cpu_work(ready_work):
    host, stages = ready_work
    production = ExecutorHost.objects.create(
        id="production",
        paused=False,
        reconciled=True,
        heartbeat_at=timezone.now(),
        budgets=host.budgets,
        telemetry=host.telemetry,
        capabilities={"profiles": ["f" * 64, "e" * 64]},
    )
    assembly = ResourceProfile.objects.create(
        id="assembly",
        fingerprint="e" * 64,
        image=stages[0].profile.image,
        stage_kind="assemble",
        resources={"ram": GIB, "cpu": 0.5, "disk": GIB},
        validated=True,
    )
    # The newest request's assembly is served before an older request's CPU work.
    final = WorkStage.objects.create(
        execution=stages[1].execution, key="publish", kind="assemble", profile=assembly
    )
    with override_settings(EXECUTION_V2_ENABLED=True):
        assert control.claim(production.pk)["stage_id"] == str(final.pk)
    # Another host never runs assembly, even an older one it could advertise.
    older = WorkStage.objects.create(
        execution=stages[0].execution, key="publish", kind="assemble", profile=assembly
    )
    WorkStage.objects.filter(pk=older.pk).update(created_at=timezone.now() - timedelta(hours=1))
    host.capabilities["profiles"].append(assembly.fingerprint)
    host.save(update_fields=["capabilities"])
    with override_settings(EXECUTION_V2_ENABLED=True):
        assert control.claim(host.pk)["stage_id"] == str(stages[0].pk)
    older.refresh_from_db()
    assert older.state == "ready"


@pytest.mark.django_db(transaction=True)
def test_alternative_profile_covers_the_input_with_the_smallest_reservation(ready_work):
    host, stages = ready_work
    bound = stages[0].profile  # Measured elsewhere; this host advertises neither it...
    bound.fingerprint = "9" * 64
    bound.settings = {"model_identity": {"m": 1}}
    bound.save()
    common = dict(
        image=bound.image,
        stage_kind="predict",
        settings={"model_identity": {"m": 1}},
        validated=True,
    )
    # ...but two local equivalents: a measured envelope and a wider clone.
    ResourceProfile.objects.create(
        id="local-narrow",
        fingerprint="1" * 64,
        resources={"ram": GIB, "cpu": 1, "disk": GIB},
        limits={"max_molecule_length": 257},
        **common,
    )
    ResourceProfile.objects.create(
        id="local-wide",
        fingerprint="2" * 64,
        resources={"ram": 2 * GIB, "cpu": 1, "disk": GIB},
        limits={"max_molecule_length": 1024},
        **common,
    )
    host.capabilities["profiles"] = ["1" * 64, "2" * 64]
    host.save(update_fields=["capabilities"])
    narrow, wide = stages
    WorkStage.objects.filter(pk=narrow.pk).update(payload={"shape": {"max_molecule_length": 100}})
    WorkStage.objects.filter(pk=wide.pk).update(payload={"shape": {"max_molecule_length": 801}})
    with override_settings(EXECUTION_V2_ENABLED=True):
        chosen = {a["stage_id"]: a["profile_fingerprint"] for a in control.claim_batch(host.pk, 8)}
    assert chosen == {str(narrow.pk): "1" * 64, str(wide.pk): "2" * 64}


@pytest.mark.django_db(transaction=True)
def test_batched_heartbeats_renew_each_lease_and_report_stale_ones(ready_work):
    host, _ = small_cpu_work(ready_work, cpu_budget=4)
    with override_settings(EXECUTION_V2_ENABLED=True):
        first, second, third = control.claim_batch(host.pk, 3)
    ExecutionAttempt.objects.filter(pk=second["id"]).update(
        lease_until=timezone.now() - timedelta(seconds=1)
    )
    before = ExecutionAttempt.objects.get(pk=first["id"]).lease_until
    reply = control.heartbeats(
        host.pk,
        {
            "attempts": [
                {"id": first["id"], "token": first["token"], "progress": 3},
                {"id": second["id"], "token": second["token"]},
                {"id": third["id"], "token": "not-the-owner"},
                {"id": "00000000-0000-0000-0000-000000000000", "token": "x"},
            ]
        },
    )["results"]
    assert reply[first["id"]] == {"lease_seconds": 30}
    assert reply[second["id"]]["status"] == 409
    assert reply[third["id"]]["status"] == 403
    assert reply["00000000-0000-0000-0000-000000000000"]["status"] == 404
    renewed = ExecutionAttempt.objects.get(pk=first["id"])
    assert renewed.lease_until > before and renewed.state == "running"
    assert renewed.stage.progress == 3
    assert ExecutionAttempt.objects.get(pk=third["id"]).state == "leased"


@pytest.mark.django_db(transaction=True)
def test_host_lock_does_not_block_foreign_key_inserts_of_completions(ready_work):
    """A claim/registration holding the host row must not block (or deadlock with)
    a completion publishing artifact locations that reference the host."""
    import threading

    from api.execution.models import Artifact, ArtifactLocation
    from django.db import connection, transaction

    host, stages = ready_work
    artifact = Artifact.objects.create(id="a" * 64, identity={}, checksum="b" * 64, size=1)
    holding, release = threading.Event(), threading.Event()

    def hold_host_lock():
        try:
            with transaction.atomic():
                control.register(host.pk, {"telemetry": {}})  # Locks the host row.
                holding.set()
                release.wait(10)
        finally:
            connection.close()

    with override_settings(EXECUTION_HOSTS={host.pk: {"budgets": host.budgets}}):
        holder = threading.Thread(target=hold_host_lock)
        holder.start()
        try:
            assert holding.wait(10)
            with transaction.atomic():
                with connection.cursor() as cursor:
                    cursor.execute("SET LOCAL lock_timeout = '2s'")
                ArtifactLocation.objects.create(artifact=artifact, host=host, ready=True)
        finally:
            release.set()
            holder.join(10)
    assert ArtifactLocation.objects.filter(artifact=artifact, host=host).exists()


@pytest.mark.django_db(transaction=True)
def test_executor_http_claim_batch_and_heartbeats(ready_work):
    host, _ = small_cpu_work(ready_work, cpu_budget=4)
    config = {host.pk: {"token": "test-token", "budgets": host.budgets}}

    def post(operation, body):
        request = RequestFactory().post(
            f"/internal/v2/executors/test/{operation}",
            data=json.dumps(body),
            content_type="application/json",
            HTTP_AUTHORIZATION="Bearer test-token",
        )
        with override_settings(EXECUTION_HOSTS=config, EXECUTION_V2_ENABLED=True):
            return json.loads(executor(request, host.pk, operation).content)

    batch = post("claim", {"limit": 2})["attempts"]
    assert len(batch) == 2
    assert post("claim", {})["attempt"]["kind"] == "predict"  # Older agents.
    results = post(
        "heartbeats", {"attempts": [{"id": a["id"], "token": a["token"]} for a in batch]}
    )["results"]
    assert all(results[a["id"]] == {"lease_seconds": 30} for a in batch)
