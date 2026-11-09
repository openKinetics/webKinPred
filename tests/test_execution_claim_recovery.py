"""Lost admission responses must not reserve a host forever."""

import json
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import timedelta

import pytest
from api.execution import control
from api.execution.models import ExecutionAttempt
from api.execution.retention import claim_cleanup
from api.execution.views import executor
from django.db import close_old_connections, connection, transaction
from django.test import RequestFactory, override_settings
from django.utils import timezone
from tests.test_execution import ready_work as ready_work
from tools.executor.agent import Agent


@pytest.mark.django_db(transaction=True)
def test_registration_reports_claims_even_if_the_claim_response_was_lost(ready_work):
    host, _ = ready_work
    with override_settings(EXECUTION_V2_ENABLED=True):
        claim = control.claim(host.pk)  # Its HTTP response is deliberately discarded.
    config = {host.pk: {"token": "test-token", "budgets": host.budgets}}
    request = RequestFactory().post(
        "/internal/v2/executors/test/register",
        data="{}",
        content_type="application/json",
        HTTP_AUTHORIZATION="Bearer test-token",
    )
    with override_settings(EXECUTION_HOSTS=config):
        reply = json.loads(executor(request, host.pk, "register").content)
    assert reply["owned_attempts"] == [claim["id"]]
    assert reply["stop_attempts"] == []
    assert reply["reconciled"] is True


@pytest.mark.django_db(transaction=True)
def test_last_expired_attempt_cannot_leave_healthy_host_permanently_blocked(ready_work, tmp_path):
    host, _ = ready_work
    with override_settings(EXECUTION_V2_ENABLED=True):
        first = control.claim(host.pk)
    ExecutionAttempt.objects.filter(pk=first["id"]).update(
        lease_until=timezone.now() - timedelta(seconds=1)
    )
    control.reconcile(host.pk, {"running": [first["id"]]})
    host.refresh_from_db()
    assert not host.reconciled
    control.fail(
        host.pk, first["id"], {"token": first["token"], "stopped": True, "category": "interrupted"}
    )
    assert not host.attempts.filter(state__in=["leased", "running", "uncertain"]).exists()
    request = RequestFactory().post(
        "/internal/v2/executors/test/register",
        data=json.dumps({"capabilities": host.capabilities, "telemetry": host.telemetry}),
        content_type="application/json",
        HTTP_AUTHORIZATION="Bearer test-token",
    )
    with override_settings(
        EXECUTION_HOSTS={host.pk: {"token": "test-token", "budgets": host.budgets}}
    ):
        reply = json.loads(executor(request, host.pk, "register").content)
    assert reply["reconciled"] is False and reply["owned_attempts"] == []
    agent = Agent({"root": str(tmp_path), "host_id": host.pk})
    checked = []
    agent.inventory = lambda: checked.append(True) or []
    agent.request = lambda operation, data: control.reconcile(host.pk, data)
    try:
        agent.reconcile_unreceived(reply)
        assert checked == [True]
        host.refresh_from_db()
        assert host.reconciled
        with override_settings(EXECUTION_V2_ENABLED=True):
            assert control.claim(host.pk)
    finally:
        agent.transfers.shutdown(wait=True)


@pytest.mark.parametrize("known", [False, True])
def test_unreconciled_host_reports_inventory_even_without_unknown_claims(tmp_path, known):
    agent = Agent({"root": str(tmp_path), "host_id": "test"})
    calls = []
    if known:
        agent.jobs["known"] = {"container": None}
    agent.inventory = lambda: calls.append("inventory") or []
    agent.request = lambda op, data: calls.append((op, data))
    try:
        agent.reconcile_unreceived({"owned_attempts": list(agent.jobs), "reconciled": False})
        assert calls == ["inventory", ("reconcile", {"running": list(agent.jobs), "stopped": {}})]
        assert bool(agent.jobs) is known
    finally:
        agent.transfers.shutdown(wait=True)


def test_unreceived_claim_reconciles_without_releasing_known_transfers(tmp_path):
    agent = Agent({"root": str(tmp_path), "host_id": "test"})
    known = {"container": None, "request": {"id": "known"}}
    agent.jobs["known"] = known
    calls = []
    agent.inventory = lambda: []
    agent.request = lambda operation, data: calls.append((operation, data))
    try:
        agent.reconcile_unreceived({"owned_attempts": ["unknown", "known"]})
        assert calls == [("reconcile", {"running": ["known"], "stopped": {}})]
        assert agent.jobs == {"known": known}
        agent.reconcile_unreceived({"owned_attempts": ["known"]})
        assert len(calls) == 1
    finally:
        agent.transfers.shutdown(wait=True)


@pytest.mark.parametrize("stop_fails", [False, True])
def test_unknown_container_is_confirmed_stopped_before_releasing_reservation(tmp_path, stop_fails):
    agent = Agent({"root": str(tmp_path), "host_id": "test"})
    events = []
    agent.inventory = lambda: [
        {
            "Id": "owned-container",
            "Config": {"Labels": {"org.webkinpred.attempt": "unknown"}},
            "State": {"Running": True},
        }
    ]

    def docker(*args):
        events.append(args[0])
        if args[0] == "inspect":
            return json.dumps(
                [{"State": {"Running": stop_fails, "FinishedAt": "2026-10-30T13:00:00Z"}}]
            )
        return ""

    agent.docker = docker
    agent.request = lambda *args: events.append("reconcile")
    try:
        if stop_fails:
            with pytest.raises(RuntimeError, match="has not stopped"):
                agent.reconcile_unreceived({"owned_attempts": ["unknown"]})
            assert events == ["stop", "inspect"]
        else:
            agent.reconcile_unreceived({"owned_attempts": ["unknown"]})
            assert events == ["stop", "inspect", "reconcile"]
    finally:
        agent.transfers.shutdown(wait=True)


@pytest.mark.django_db(transaction=True)
def test_cleanup_skips_busy_preparation_lock_instead_of_blocking_http_worker(ready_work):
    # Queues for at most 0.9 s behind graph publication, then gives up.
    host, _ = ready_work
    held, release = threading.Event(), threading.Event()

    def preparation():
        close_old_connections()
        try:
            with transaction.atomic():
                with connection.cursor() as cursor:
                    cursor.execute("SELECT pg_advisory_xact_lock(%s)", [481726392])
                held.set()
                assert release.wait(5)
        finally:
            close_old_connections()

    with ThreadPoolExecutor(max_workers=1) as pool:
        future = pool.submit(preparation)
        try:
            assert held.wait(3)
            start = time.monotonic()
            assert claim_cleanup(host.pk) is None
            assert time.monotonic() - start < 1
        finally:
            release.set()
            future.result()


@pytest.mark.django_db(transaction=True)
def test_lease_starts_when_admitted_not_before_slow_candidate_selection(ready_work, monkeypatch):
    host, _ = ready_work
    before = timezone.now()
    admitted = before + timedelta(seconds=12)
    times = iter([before, admitted])
    monkeypatch.setattr(control.timezone, "now", lambda: next(times))
    with override_settings(EXECUTION_V2_ENABLED=True):
        result = control.claim(host.pk)
    attempt = ExecutionAttempt.objects.get(pk=result["id"])
    assert attempt.lease_until == admitted + timedelta(seconds=30)


@pytest.mark.parametrize(
    "operation,age,expected", [("claim", 0, 8), ("claim", 12, 3), ("heartbeat", 12, 5)]
)
def test_claim_wait_budget_preserves_time_for_compute_fencing(
    tmp_path, monkeypatch, operation, age, expected
):
    import io

    from tools.executor import agent as module

    agent = Agent(
        {
            "root": str(tmp_path),
            "host_id": "test",
            "control_url": "http://control",
            "token": "private-test",
        }
    )
    agent.jobs["running"] = {"renewed": 100 - age}
    monkeypatch.setattr(module.time, "monotonic", lambda: 100)
    waits = []

    def open_url(request, timeout):
        waits.append(timeout)
        return io.BytesIO(b"{}")

    monkeypatch.setattr(module.urllib.request, "urlopen", open_url)
    try:
        assert agent.request(operation) == {}
        assert waits == [expected]
    finally:
        agent.transfers.shutdown(wait=True)


@pytest.mark.django_db(transaction=True)
def test_slow_cleanup_scan_rolls_back_instead_of_blocking_heartbeats(ready_work):
    from api.execution.models import ArtifactCleanup

    host, _ = ready_work
    host.budgets["cache_free_target"] = 10**20
    host.save(update_fields=["budgets"])
    delayed = False

    def slow_scan(execute, sql, params, many, context):
        nonlocal delayed
        if not delayed and "api_artifactlocation" in sql:
            delayed = True
            execute("SELECT pg_sleep(3)", None, False, context)
        return execute(sql, params, many, context)

    with connection.execute_wrapper(slow_scan):
        assert claim_cleanup(host.pk) is None
    assert delayed
    assert not ArtifactCleanup.objects.filter(host=host).exists()
    with connection.cursor() as cursor:
        cursor.execute("SELECT 1")
        assert cursor.fetchone() == (1,)
