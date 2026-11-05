import pytest
from api.execution import metrics
from api.execution.models import Artifact, ExecutorHost
from django.db import connection
from django.db.models.query import QuerySet
from django.test import RequestFactory, override_settings
from django.utils import timezone


def scrape():
    with override_settings(EXECUTION_METRICS_TOKEN="test-metrics"):
        return metrics.metrics(
            RequestFactory().get("/internal/v2/metrics", HTTP_AUTHORIZATION="Bearer test-metrics")
        )


@pytest.mark.django_db(transaction=True)
def test_overlapping_scrape_does_not_queue_another_inventory_scan():
    other = connection.copy(alias="metrics-lock-probe")
    try:
        with other.cursor() as cursor:
            cursor.execute("SELECT pg_advisory_lock(481726395)")
        response = scrape()
        assert response.status_code == 503
        assert response["Retry-After"] == "15"
    finally:
        with other.cursor() as cursor:
            cursor.execute("SELECT pg_advisory_unlock(481726395)")
        other.close()
    assert scrape().status_code == 200


@pytest.mark.django_db(transaction=True)
def test_slow_inventory_is_omitted_without_losing_other_metrics(monkeypatch):
    ExecutorHost.objects.create(id="gpu", heartbeat_at=timezone.now())
    aggregate = QuerySet.aggregate

    def slow_artifacts(query, *args, **kwargs):
        if query.model is Artifact:
            with connection.cursor() as cursor:
                cursor.execute("SELECT pg_sleep(2)")
        return aggregate(query, *args, **kwargs)

    with connection.cursor() as cursor:
        cursor.execute("SHOW statement_timeout")
        previous = cursor.fetchone()[0]
    monkeypatch.setattr(QuerySet, "aggregate", slow_artifacts)
    with override_settings(EXECUTION_METRICS_ARTIFACT_INVENTORY=True):
        response = scrape()
    assert response.status_code == 200
    assert b'webkinpred_execution_artifact_inventory_available{host="gpu"} 0' in response.content
    assert b"webkinpred_execution_artifact_bytes" not in response.content
    assert b"webkinpred_execution_live_attempts" in response.content
    with connection.cursor() as cursor:
        cursor.execute("SHOW statement_timeout")
        assert cursor.fetchone()[0] == previous


@pytest.mark.django_db(transaction=True)
def test_artifact_inventory_scan_is_skipped_unless_enabled(monkeypatch):
    ExecutorHost.objects.create(id="gpu", heartbeat_at=timezone.now())
    aggregate = QuerySet.aggregate

    def no_inventory(query, *args, **kwargs):
        assert query.model is not Artifact, "inventory scan must not run by default"
        return aggregate(query, *args, **kwargs)

    monkeypatch.setattr(QuerySet, "aggregate", no_inventory)
    response = scrape()
    assert response.status_code == 200
    assert b'webkinpred_execution_artifact_inventory_available{host="gpu"} 0' in response.content
    with override_settings(EXECUTION_METRICS_ARTIFACT_INVENTORY=True):
        monkeypatch.setattr(QuerySet, "aggregate", aggregate)
        enabled = scrape()
    assert b'webkinpred_execution_artifact_inventory_available{host="gpu"} 1' in enabled.content
    assert b'webkinpred_execution_artifact_bytes{host="gpu"} 0' in enabled.content


@pytest.mark.django_db(transaction=True)
def test_essential_metrics_query_timeout_returns_retryable_response(monkeypatch):
    def slow_render():
        with connection.cursor() as cursor:
            cursor.execute("SELECT pg_sleep(2)")

    monkeypatch.setattr(metrics, "render", slow_render)
    assert scrape().status_code == 503
    with connection.cursor() as cursor:
        cursor.execute("SELECT 1")
        assert cursor.fetchone()[0] == 1


@pytest.mark.django_db(transaction=True)
def test_prediction_metrics_aggregate_counters_without_returning_full_manifests():
    from api.execution.models import ExecutionSpec, WorkStage
    from django.test.utils import CaptureQueriesContext

    spec = ExecutionSpec.objects.create(
        specification={}, checksum="a" * 64, input_checksum="b" * 64
    )
    for index, units in enumerate([3, 5]):
        WorkStage.objects.create(
            execution=spec,
            key=str(index),
            kind="predict",
            method="test",
            target="kcat",
            state="completed",
            completed_at=timezone.now(),
            payload={"shape": {"units": units}, "rows": [{"private_large_input": "x" * 100000}]},
        )
    with CaptureQueriesContext(connection) as captured:
        response = scrape()
    assert response.status_code == 200
    assert (
        b'webkinpred_execution_prediction_units_completed_last_hour{method="test",target="kcat"} 8'
        in response.content
    )
    counters = [q["sql"] for q in captured if 'AS "units"' in q["sql"]]
    assert len(counters) == 1 and "SUM(" in counters[0]
    assert '"api_workstage"."payload",' not in counters[0]
    assert b"private_large_input" not in response.content
