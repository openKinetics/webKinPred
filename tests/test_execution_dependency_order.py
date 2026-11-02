from copy import deepcopy
from datetime import timedelta

import pytest
from api.execution import control
from api.execution.cohorts import coalesce
from api.execution.dependency_order import refresh_dependency_order
from api.execution.models import WorkStage
from api.models import Job
from django.test import override_settings
from django.utils import timezone
from tests.test_execution import ready_work as ready_work
from tests.test_execution_cohorts import setup_cohort

pytestmark = pytest.mark.django_db(transaction=True)


def prepared(execution):
    WorkStage.objects.update_or_create(
        execution=execution, key="prepare", defaults={"kind": "prepare", "state": "completed"}
    )


def consumer(stage, name, age):
    Job.objects.create(
        public_id=name,
        prediction_type="kcat",
        status="Processing",
        submission_time=age,
    )
    execution = control.ingest(name, {"targets": ["kcat"]}, "b" * 64, paused=False)
    prepared(execution)
    head = WorkStage.objects.create(execution=execution, key="head", kind="predict")
    head.dependencies.add(stage)
    return execution, head


def refresh_all(limit=8):
    cursor, changes = 0, 0
    while True:
        cursor, changed = refresh_dependency_order(cursor, limit)
        changes += changed
        if not cursor:
            return changes


def test_shared_prerequisite_keeps_old_consumers_age_and_is_claimed_first(ready_work):
    host, (producer, other) = ready_work
    now = timezone.now()
    Job.objects.filter(pk=producer.execution.job_id).update(submission_time=now)
    Job.objects.filter(pk=other.execution.job_id).update(submission_time=now - timedelta(hours=1))
    execution, head = consumer(producer, "oldrequest", now - timedelta(days=2))
    before, checksum = deepcopy(execution.specification), execution.checksum
    assert refresh_all() == 1
    producer.refresh_from_db()
    assert producer.oldest_consumer_at == execution.job.submission_time
    assert other.oldest_consumer_at is None
    assert refresh_all() == 0
    with override_settings(EXECUTION_V2_ENABLED=True):
        assert control.claim(host.pk)["stage_id"] == str(producer.pk)
    execution.refresh_from_db()
    assert execution.specification == before and execution.checksum == checksum
    assert head.dependencies.get() == producer


def test_transitive_shared_dependencies_inherit_the_oldest_live_request(ready_work):
    _, (producer, middle) = ready_work
    middle.dependencies.add(producer)
    now = timezone.now()
    old, _ = consumer(middle, "oldest", now - timedelta(days=3))
    consumer(producer, "later", now - timedelta(days=1))
    assert refresh_all(limit=1) >= 2
    producer.refresh_from_db()
    middle.refresh_from_db()
    assert producer.oldest_consumer_at == middle.oldest_consumer_at == old.job.submission_time


@pytest.mark.parametrize("inactive", ["paused", "completed", "canceled"])
def test_inactive_consumers_do_not_promote_unrelated_work(ready_work, inactive):
    _, (producer, _) = ready_work
    execution, head = consumer(producer, "inactive", timezone.now() - timedelta(days=3))
    if inactive == "paused":
        execution.paused = True
        execution.save(update_fields=["paused"])
    elif inactive == "completed":
        Job.objects.filter(pk=execution.job_id).update(status="Completed")
    else:
        head.state = "canceled"
        head.save(update_fields=["state"])
    assert refresh_all() == 0
    producer.refresh_from_db()
    assert producer.oldest_consumer_at is None


def test_explicit_priority_still_precedes_inherited_age(ready_work):
    host, (producer, other) = ready_work
    consumer(producer, "oldprio", timezone.now() - timedelta(days=3))
    other.priority = 10
    other.save(update_fields=["priority"])
    refresh_all()
    with override_settings(EXECUTION_V2_ENABLED=True):
        assert control.claim(host.pk)["stage_id"] == str(other.pk)


def test_bounded_pages_eventually_visit_each_prepared_request(ready_work):
    _, (producer, _) = ready_work
    old, _ = consumer(producer, "firstpage", timezone.now() - timedelta(days=2))
    newest, _ = consumer(producer, "nextpage", timezone.now() - timedelta(days=1))
    cursor, _ = refresh_dependency_order(limit=1)
    assert cursor == old.pk
    cursor, _ = refresh_dependency_order(cursor, limit=1)
    assert cursor == newest.pk
    assert refresh_dependency_order(cursor, limit=1) == (0, 0)


def test_cohort_retains_inherited_age_and_all_original_consumers():
    _, _, _, _, sources, head = setup_cohort()
    age = timezone.now() - timedelta(days=3)
    WorkStage.objects.filter(pk=sources[1].pk).update(oldest_consumer_at=age)
    ids = coalesce()
    assert len(ids) == 1
    merged = WorkStage.objects.get(pk=ids[0])
    assert merged.oldest_consumer_at == age
    assert set(head.dependencies.values_list("pk", flat=True)) == {merged.pk}
