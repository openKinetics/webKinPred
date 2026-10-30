import io
from datetime import timedelta

import pytest
from api.execution import control
from api.execution.models import ExecutionAttempt, ExecutionSpec, WorkStage
from api.execution.priority import refresh_stage_priorities
from api.models import Job
from django.core.management import call_command
from django.core.management.base import CommandError
from django.db import connection
from django.test import override_settings
from django.utils import timezone
from tests.test_execution import ready_work as ready_work

pytestmark = pytest.mark.django_db(transaction=True)


def promote(ident, **options):
    output = io.StringIO()
    call_command("prioritize_job", ident, stdout=output, **options)
    return output.getvalue()


def test_younger_priority_job_runs_before_older_without_changing_request_or_quota(ready_work):
    host, stages = ready_work
    old, new = stages
    checksum = new.execution.checksum
    submitted = new.execution.job.submission_time
    promote(new.execution.job.public_id)
    with override_settings(EXECUTION_V2_ENABLED=True):
        result = control.claim(host.pk)
    assert result["stage_id"] == str(new.pk)
    new.execution.refresh_from_db()
    new.execution.job.refresh_from_db()
    assert new.execution.priority == 100 and new.execution.checksum == checksum
    assert new.execution.job.submission_time == submitted


def test_new_stages_inherit_execution_priority_before_periodic_propagation(ready_work):
    host, stages = ready_work
    current = stages[1]
    promote(current.execution.job.public_id)
    current.state = "completed"
    current.save(update_fields=["state"])
    late = WorkStage.objects.create(
        execution=current.execution, key="late", kind="predict", profile=current.profile
    )
    assert late.priority == 0
    with override_settings(EXECUTION_V2_ENABLED=True):
        assert control.claim(host.pk)["stage_id"] == str(late.pk)


def test_shared_dependencies_receive_max_priority_and_clear_preserves_another_boost(ready_work):
    host, stages = ready_work
    first, second = stages
    shared = WorkStage.objects.create(
        execution=first.execution, key="shared", kind="predict", profile=first.profile
    )
    first.dependencies.add(shared)
    second.dependencies.add(shared)
    promote(first.execution.job.public_id, priority=50)
    promote(second.execution.job.public_id, priority=100)
    shared.refresh_from_db()
    assert shared.priority == 100
    with override_settings(EXECUTION_V2_ENABLED=True):
        assert control.claim(host.pk)["stage_id"] == str(shared.pk)
    promote(second.execution.job.public_id, clear=True)
    shared.refresh_from_db()
    assert shared.priority == 50
    promote(first.execution.job.public_id, clear=True)
    shared.refresh_from_db()
    assert shared.priority == 0


def test_priority_does_not_preempt_or_overcommit_running_attempt(ready_work):
    host, stages = ready_work
    with override_settings(EXECUTION_V2_ENABLED=True):
        running = control.claim(host.pk)
        promote(stages[1].execution.job.public_id)
        assert control.claim(host.pk) is None
    attempt = ExecutionAttempt.objects.get(pk=running["id"])
    assert attempt.state == "leased" and not attempt.stop_requested


@pytest.mark.parametrize("priority", [-1, 1_000_001])
def test_invalid_priority_does_not_mutate_work(ready_work, priority):
    _, stages = ready_work
    with pytest.raises(CommandError, match="between"):
        promote(stages[0].execution.job.public_id, priority=priority)
    stages[0].execution.refresh_from_db()
    assert stages[0].execution.priority == 0


def test_completed_job_is_noop_and_never_recomputed(ready_work):
    _, stages = ready_work
    e = stages[1].execution
    Job.objects.filter(pk=e.job_id).update(status="Completed")
    before = e.stages.count()
    assert "already completed" in promote(e.job.public_id)
    e.refresh_from_db()
    assert e.priority == 0 and e.stages.count() == before
    assert not e.stages.filter(attempts__isnull=False).exists()


def test_paused_job_requires_explicit_resume_without_resuming_other_jobs(ready_work):
    _, stages = ready_work
    ExecutionSpec.objects.update(paused=True)
    with pytest.raises(CommandError, match="--resume"):
        promote(stages[1].execution.job.public_id)
    promote(stages[1].execution.job.public_id, resume=True)
    stages[0].execution.refresh_from_db()
    stages[1].execution.refresh_from_db()
    assert stages[0].execution.paused and not stages[1].execution.paused


def test_holds_backoff_completed_checkpoints_and_profiles_are_unchanged(ready_work):
    _, stages = ready_work
    s = stages[1]
    s.state = "held"
    s.failures = 3
    s.checkpoint = {"committed": 7}
    s.available_at = timezone.now() + timedelta(hours=1)
    s.save()
    output = promote(s.execution.job.public_id)
    s.refresh_from_db()
    assert "held stages" in output and s.state == "held"
    assert s.failures == 3 and s.checkpoint == {"committed": 7} and s.available_at > timezone.now()


def test_clearing_priority_is_idempotent_and_completed_dependencies_are_not_promoted(ready_work):
    _, stages = ready_work
    old, new = stages
    old.state = "completed"
    old.save()
    new.dependencies.add(old)
    promote(new.execution.job.public_id)
    old.refresh_from_db()
    assert old.priority == 0
    assert refresh_stage_priorities() == 0
    promote(new.execution.job.public_id, clear=True)
    assert refresh_stage_priorities() == 0


def test_database_defaults_allow_old_services_to_create_unprioritized_rows(ready_work):
    _, stages = ready_work
    with connection.cursor() as cursor:
        for table in ["api_executionspec", "api_workstage"]:
            cursor.execute(
                "SELECT column_default FROM information_schema.columns "
                "WHERE table_name=%s AND column_name=%s",
                [table, "priority"],
            )
            assert cursor.fetchone()[0] in ["0", "'0'::integer"]


def test_missing_or_failed_job_is_not_implicitly_resubmitted(ready_work):
    with pytest.raises(CommandError, match="does not exist"):
        promote("notfound")
    _, stages = ready_work
    Job.objects.filter(pk=stages[0].execution.job_id).update(status="Failed")
    with pytest.raises(CommandError, match="Failed"):
        promote(stages[0].execution.job.public_id)


@pytest.mark.parametrize("oversized", [False, True])
def test_prioritized_gpu_work_drains_lower_gpu_admissions_but_allows_cpu_work(
    ready_work, oversized
):
    from api.execution.models import ResourceProfile
    from api.execution.resources import GIB

    host, stages = ready_work
    urgent, normal = stages
    p = urgent.profile
    p.resources = {"cpu": 1, "ram": 20 * GIB if oversized else GIB, "disk": GIB, "vram": 2 * GIB}
    p.save()
    compatible = ResourceProfile.objects.create(
        id="compatible",
        fingerprint="g" * 64,
        image=p.image,
        stage_kind="predict",
        validated=True,
        resources={**p.resources, "ram": GIB},
    )
    cpu = ResourceProfile.objects.create(
        id="cpu-small",
        fingerprint="c" * 64,
        image=p.image,
        stage_kind="predict",
        validated=True,
        resources={"cpu": 1, "ram": GIB, "disk": GIB, "vram": 0},
    )
    normal.profile = compatible
    normal.save()
    filler = WorkStage.objects.create(
        execution=normal.execution, key="cpu-filler", kind="predict", profile=cpu
    )
    owner = ExecutionSpec.objects.create(
        specification={}, checksum="a" * 64, input_checksum="b" * 64, paused=False
    )
    active = WorkStage.objects.create(execution=owner, key="active", state="running")
    ExecutionAttempt.objects.create(
        stage=active,
        host=host,
        resources=compatible.resources,
        profile_fingerprint=compatible.fingerprint,
        lease_until=timezone.now() + timedelta(seconds=30),
    )
    host.budgets.update(cpu=8, ram=8 * GIB, disk=8 * GIB, gpu_slots=2)
    host.telemetry.update(gpu_uuid="gpu", vram_total=20 * GIB, vram_free=16 * GIB)
    host.capabilities.update(
        profiles=[p.fingerprint, compatible.fingerprint, cpu.fingerprint],
        validated_gpu_combinations=[[compatible.fingerprint] * 2],
    )
    host.save()
    promote(urgent.execution.job.public_id)
    with override_settings(EXECUTION_V2_ENABLED=True):
        assert control.claim(host.pk)["stage_id"] == str(normal.pk if oversized else filler.pk)
    assert normal.attempts.exists() == oversized


def test_queue_display_tracks_priority_before_submission_time(ready_work, monkeypatch):
    from api.execution import status

    monkeypatch.setattr(status, "_queue_snapshot", None)
    _, stages = ready_work
    promote(stages[1].execution.job.public_id)
    with override_settings(EXECUTION_V2_ENABLED=True):
        first = status.execution_status(stages[1].execution.job)
        second = status.execution_status(stages[0].execution.job)
    assert first["queue_estimate"]["position"] == 1
    assert first["queue_estimate"]["basis"] == "priority_and_submission_order"
    assert second["queue_estimate"]["position"] == 2
    assert first["scheduling_priority"] == 100


def test_preparation_yields_between_chunks_and_resumes_without_losing_work(tmp_path, monkeypatch):
    from api.execution import bounded
    from api.execution.artifacts import sha256_file
    from api.execution.models import ExecutorHost
    from api.execution.planning import prepare

    media = tmp_path / "media"
    blobs = tmp_path / "blobs"
    ExecutorHost.objects.create(id="production")
    data = media / "jobs" / "normal" / "input.csv"
    data.parent.mkdir(parents=True)
    data.write_text("Protein Sequence,Substrate\n" + "MAAA,CC\n" * 129)
    normal = Job.objects.create(public_id="normal", prediction_type="kcat", status="Pending")
    urgent = Job.objects.create(public_id="urgent", prediction_type="kcat", status="Pending")
    spec = {
        "targets": ["kcat"],
        "methods": {"kcat": "DLKcat"},
        "handle_long_sequences": "truncate",
        "canonicalize_substrates": True,
        "include_similarity_columns": False,
        "model_versions": {"DLKcat": {"weights": "test"}},
    }
    e = control.ingest(normal.public_id, spec, sha256_file(data), paused=False)
    u = control.ingest(urgent.public_id, spec, "a" * 64, paused=False)
    original = bounded.frames
    promoted = []

    def frames(path, *args, **kwargs):
        for index, frame in enumerate(original(path, *args, **kwargs)):
            if index == 1 and not promoted:
                promote(urgent.public_id)
                promoted.append(True)
            yield frame

    monkeypatch.setattr(bounded, "frames", frames)
    with override_settings(EXECUTION_V2_ENABLED=True):
        prepare(e, media, blobs)
        assert e.stages.get(key="prepare").state == "ready"
        original_ids = set(e.stages.values_list("pk", flat=True))
        assert e.stages.filter(kind="predict").count() == 1
        ExecutionSpec.objects.filter(pk=u.pk).update(paused=True)
        prepare(e, media, blobs)
    assert e.stages.get(key="prepare").state == "completed"
    assert original_ids <= set(e.stages.values_list("pk", flat=True))
    assert e.stages.filter(kind="predict").count() == 2


def test_priority_ranks_beyond_bounded_queue_snapshot_use_priority_aware_counts(
    ready_work, monkeypatch
):
    from api.execution import status

    monkeypatch.setattr(status, "_queue_snapshot", None)
    monkeypatch.setattr(status, "_QUEUE_SNAPSHOT_LIMIT", 1)
    _, stages = ready_work
    promote(stages[1].execution.job.public_id)
    with override_settings(EXECUTION_V2_ENABLED=True):
        data = status.execution_status(stages[0].execution.job)
    assert len(status._queue_snapshot["jobs"]) == 1
    assert data["queue_estimate"]["position"] == 2
    assert data["queue_estimate"]["basis"] == "priority_and_submission_order"
