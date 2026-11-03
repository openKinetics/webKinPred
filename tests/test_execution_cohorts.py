from copy import deepcopy
from datetime import timedelta

import pytest
from api.execution import control
from api.execution.cohorts import coalesce
from api.execution.models import (
    Artifact,
    ArtifactDependency,
    ExecutionAttempt,
    ExecutorHost,
    ResourceProfile,
    WorkStage,
)
from api.models import Job
from django.utils import timezone

pytestmark = pytest.mark.django_db(transaction=True)


def setup_cohort():
    original = ResourceProfile.objects.create(
        id="serial",
        fingerprint="a" * 64,
        image="sha256:" + "1" * 64,
        stage_kind="embedding",
        method="EITLEM",
        settings={
            "step": "eitlem_esm1v",
            "model_identity": {"weights": "pinned"},
            "targets": ["kcat"],
        },
        resources={"cpu": 1, "ram": 2**30, "vram": 2**30, "disk": 2**30},
        limits={"sequences": 16, "max_length": 1024},
        validated=True,
    )
    large = ResourceProfile.objects.create(
        id="cohort",
        fingerprint="b" * 64,
        image=original.image,
        stage_kind=original.stage_kind,
        method=original.method,
        settings={**original.settings, "cohort_sources": [original.fingerprint]},
        resources=original.resources,
        limits={"sequences": 32, "max_length": 1024},
        validated=True,
    )
    host = ExecutorHost.objects.create(
        id="gpu",
        paused=False,
        reconciled=True,
        heartbeat_at=timezone.now(),
        capabilities={"profiles": [original.fingerprint, large.fingerprint]},
    )
    job = Job.objects.create(public_id="cohortjob", prediction_type="kcat", status="Pending")
    execution = control.ingest(
        job.public_id, {"targets": ["kcat"], "methods": {"kcat": "EITLEM"}}, "f" * 64, paused=False
    )
    execution.stages.all().delete()
    stages = []
    for batch in range(4):
        mapping = {f"s{batch}_{i}": "M" + "A" * i for i in range(8)}
        stage = WorkStage.objects.create(
            execution=execution,
            key=f"part:{batch}",
            profile=original,
            kind="embedding",
            method="EITLEM",
            target="kcat",
            payload={
                "step": "eitlem_esm1v",
                "seq_id_to_seq": mapping,
                "outputs": [],
                "output_artifacts": [],
                "shape": {"sequences": 8, "max_length": 8},
            },
        )
        outputs = []
        for sid in mapping:
            identity = {"sequence": sid, "version": "pinned"}
            artifact = Artifact.objects.create(
                id=control.digest(identity), identity=identity, size=0, producer=stage
            )
            outputs.append({"id": artifact.pk, "identity": identity, "path": f"{sid}.npy"})
        stage.payload["outputs"] = outputs
        stage.payload["output_artifacts"] = [o["id"] for o in outputs]
        stage.save(update_fields=["payload"])
        stages.append(stage)
    head = WorkStage.objects.create(
        execution=execution, key="head", kind="predict", method="EITLEM", target="kcat"
    )
    head.dependencies.add(*stages)
    for stage in stages:
        for o in stage.payload["outputs"]:
            ArtifactDependency.objects.create(stage=head, artifact_id=o["id"], path=o["path"])
    return host, execution, original, large, stages, head


def test_cohort_preserves_every_artifact_consumer_and_immutable_request():
    _, e, _, large, stages, head = setup_cohort()
    before = deepcopy(e.specification)
    checksum = e.checksum
    inputs = list(head.artifacts.values_list("artifact_id", "path"))
    ids = coalesce()
    assert len(ids) == 1
    cohort = WorkStage.objects.get(pk=ids[0])
    assert cohort.profile == large and cohort.payload["shape"]["sequences"] == 32
    assert len(cohort.payload["seq_id_to_seq"]) == 32 and len(cohort.payload["outputs"]) == 32
    assert set(head.dependencies.values_list("pk", flat=True)) == {cohort.pk}
    assert list(head.artifacts.values_list("artifact_id", "path")) == inputs
    assert Artifact.objects.filter(producer=cohort).count() == 32
    assert WorkStage.objects.filter(pk__in=[s.pk for s in stages], state="canceled").count() == 4
    e.refresh_from_db()
    assert e.specification == before and e.checksum == checksum
    assert coalesce() == []


@pytest.mark.parametrize("background", [True, False])
def test_pending_cohort_buffer_bounds_bulk_graph_work_but_allows_foreground(background):
    _, execution, _, large, stages, _ = setup_cohort()
    execution.background = background
    execution.save(update_fields=["background"])
    for i in range(8):
        WorkStage.objects.create(
            execution=execution, key=f"queued:{i}", kind="embedding", profile=large
        )
    ids = coalesce()
    assert len(ids) == (0 if background else 1)
    assert WorkStage.objects.filter(pk__in=[s.pk for s in stages], state="ready").count() == (
        4 if background else 0
    )


def test_bulk_cohort_buffer_fills_only_remaining_slots_and_reopens_after_completion():
    _, execution, _, large, stages, _ = setup_cohort()
    execution.background = True
    execution.save(update_fields=["background"])
    large.limits["sequences"] = 16
    large.save(update_fields=["limits"])
    queued = [
        WorkStage.objects.create(
            execution=execution, key=f"queued:{i}", kind="embedding", profile=large
        )
        for i in range(7)
    ]
    assert len(coalesce()) == 1
    assert WorkStage.objects.filter(pk__in=[s.pk for s in stages], state="ready").count() == 2
    queued[0].state = "completed"
    queued[0].save(update_fields=["state"])
    assert len(coalesce()) == 1
    assert not WorkStage.objects.filter(pk__in=[s.pk for s in stages], state="ready").exists()


def test_candidate_ranking_does_not_read_large_payloads_before_bounded_selection():
    from django.db import connection
    from django.test.utils import CaptureQueriesContext

    setup_cohort()
    with CaptureQueriesContext(connection) as captured:
        assert coalesce()
    ranked = [
        q["sql"]
        for q in captured
        if 'SELECT "api_workstage"."id"' in q["sql"] and "FOR UPDATE" in q["sql"]
    ]
    assert ranked
    assert all('"payload"' not in sql for sql in ranked)


def test_cohort_never_changes_active_retry_or_other_jobs():
    host, e, original, large, stages, _ = setup_cohort()
    stages[0].state = "running"
    stages[0].save(update_fields=["state"])
    ExecutionAttempt.objects.create(
        stage=stages[1], host=host, state="stopped", lease_until=timezone.now()
    )
    other = Job.objects.create(public_id="otherjob", prediction_type="kcat", status="Pending")
    other_e = control.ingest(other.public_id, {"targets": ["kcat"]}, "f" * 64, paused=False)
    stages[2].execution = other_e
    stages[2].save(update_fields=["execution"])
    assert coalesce() == []
    assert not WorkStage.objects.filter(profile=large).exists()


def shared_sources(policy="resource-v1"):
    from api.execution.models import ExecutionSpec

    host, first, parent, large, stages, head = setup_cohort()
    first.background = True
    first.save(update_fields=["background"])
    job = Job.objects.create(public_id="otherbulk", prediction_type="kcat", status="Pending")
    second = ExecutionSpec.objects.create(
        job=job,
        specification=first.specification,
        checksum=first.checksum,
        input_checksum="d" * 64,
        paused=False,
        background=True,
        policy_version=first.policy_version if policy == "resource-v1" else policy,
    )
    WorkStage.objects.filter(pk__in=[s.pk for s in stages[2:]]).update(execution=second)
    return host, first, second, parent, large, stages, head


def test_cross_job_batch_preserves_requests_consumers_and_oldest_submission():
    _, first, second, _, large, stages, head = shared_sources()
    requests = {
        e.pk: (deepcopy(e.specification), e.checksum, e.input_checksum) for e in [first, second]
    }
    consumers = list(head.artifacts.values_list("artifact_id", "path"))
    cohort = WorkStage.objects.get(pk=coalesce(cross_jobs=True)[0])
    assert cohort.execution.job_id is None and cohort.execution.background
    assert cohort.profile == large and cohort.payload["shape"]["sequences"] == 32
    assert cohort.oldest_consumer_at == min(first.job.submission_time, second.job.submission_time)
    assert {
        r["id"] for r in cohort.execution.specification["shared_protein_batch"]["source_requests"]
    } == set(requests)
    assert set(head.dependencies.values_list("pk", flat=True)) == {cohort.pk}
    assert list(head.artifacts.values_list("artifact_id", "path")) == consumers
    assert Artifact.objects.filter(producer=cohort).count() == 32
    assert WorkStage.objects.filter(pk__in=[s.pk for s in stages], state="canceled").count() == 4
    for execution in [first, second]:
        execution.refresh_from_db()
        assert (execution.specification, execution.checksum, execution.input_checksum) == requests[
            execution.pk
        ]
    assert coalesce(cross_jobs=True) == []


@pytest.mark.parametrize("difference", ["foreground", "priority", "policy", "options"])
def test_cross_job_batches_keep_scheduling_and_execution_boundaries(difference):
    _, first, second, _, _, stages, _ = shared_sources(
        policy="different-policy" if difference == "policy" else "resource-v1"
    )
    if difference == "foreground":
        first.background = False
        first.save(update_fields=["background"])
    elif difference == "priority":
        second.priority = 5
        second.save(update_fields=["priority"])
    elif difference == "options":
        for stage in stages[2:]:
            stage.payload["precision"] = "different"
            stage.save(update_fields=["payload"])
    cohorts = WorkStage.objects.filter(pk__in=coalesce(cross_jobs=True))
    assert cohorts.count() == 2
    assert all(s.execution.job_id is not None for s in cohorts)


def test_background_prerequisite_for_foreground_is_not_mixed_with_bulk_jobs():
    _, first, _, _, _, stages, _ = shared_sources()
    job = Job.objects.create(public_id="interact", prediction_type="kcat", status="Pending")
    foreground = control.ingest(job.public_id, {"targets": ["kcat"]}, "e" * 64, paused=False)
    consumer = WorkStage.objects.create(execution=foreground, key="needs", kind="predict")
    consumer.dependencies.add(*stages[:2])
    cohorts = WorkStage.objects.filter(pk__in=coalesce(cross_jobs=True))
    assert cohorts.count() == 2
    assert all(s.execution.job_id is not None for s in cohorts)


@pytest.mark.parametrize("category,width,suspended", [("unknown", 1, False), ("oom", 16, True)])
def test_shared_failure_isolates_work_without_reopening_jobs_or_rebatching(
    category, width, suspended
):
    from api.execution.cohorts import split_cohort

    host, first, second, parent, large, _, head = shared_sources()
    cohort = WorkStage.objects.get(pk=coalesce(cross_jobs=True)[0])
    attempt = ExecutionAttempt.objects.create(
        stage=cohort,
        host=host,
        state="running",
        profile_fingerprint=large.fingerprint,
        lease_until=timezone.now(),
    )
    control.fail(
        host.pk, attempt.pk, {"token": str(attempt.token), "stopped": True, "category": category}
    )
    assert split_cohort(cohort.pk)
    children = list(cohort.execution.stages.filter(state="ready"))
    assert len(children) == 32 // width
    assert all(
        s.profile_id == parent.pk and s.payload["shape"]["sequences"] == width for s in children
    )
    assert all(
        s.oldest_consumer_at == cohort.oldest_consumer_at and s.payload["do_not_coalesce"]
        for s in children
    )
    assert sum(Artifact.objects.filter(producer=s).count() for s in children) == 32
    assert set(head.dependencies.values_list("pk", flat=True)) == {s.pk for s in children}
    large.refresh_from_db()
    assert large.suspended == suspended
    assert coalesce(cross_jobs=True) == []
    first.job.refresh_from_db()
    second.job.refresh_from_db()
    assert first.job.status == second.job.status == "Pending"


@pytest.mark.parametrize("shared,expected", [(True, "transient"), (False, "protected")])
def test_shared_raw_checkpoint_remains_transient_but_legacy_service_outputs_are_protected(
    shared, expected
):
    host, _, _, _, _, _, _ = shared_sources()
    cohort = WorkStage.objects.get(pk=coalesce(cross_jobs=True)[0])
    if not shared:
        from api.execution.models import ExecutionSpec

        cohort.execution = ExecutionSpec.objects.create(
            specification={"legacy_service": True},
            checksum="c" * 64,
            input_checksum="d" * 64,
            paused=False,
        )
        cohort.save(update_fields=["execution"])
    identity = {"representation_version": "execution-v2", "path": "sequence_info/esm1v/raw.npy"}
    artifact = Artifact.objects.create(
        id=control.digest(identity), identity=identity, producer=cohort, size=0
    )
    declaration = {"id": artifact.pk, "identity": identity, "path": "raw.npy"}
    cohort.payload["outputs"] = [declaration]
    cohort.payload["output_artifacts"] = [artifact.pk]
    cohort.save(update_fields=["payload"])
    attempt = ExecutionAttempt.objects.create(
        stage=cohort,
        host=host,
        state="running",
        lease_until=timezone.now() + timedelta(seconds=30),
    )
    control.checkpoint(
        host.pk,
        attempt.pk,
        {
            "token": str(attempt.token),
            "artifacts": [{**declaration, "checksum": "f" * 64, "size": 12}],
        },
    )
    artifact.refresh_from_db()
    assert artifact.retention == expected
    assert control.finish(
        host.pk,
        attempt.pk,
        {
            "token": str(attempt.token),
            "stopped": True,
            "artifacts": [{**declaration, "checksum": "f" * 64, "size": 12}],
        },
    ) == {"state": "completed"}
    artifact.refresh_from_db()
    assert artifact.retention == expected


def test_shared_claim_starts_source_jobs_and_projects_only_referenced_outputs():
    from api.execution.progress import refresh
    from api.execution.status import execution_status
    from api.models import JobProgressStage
    from django.test import override_settings

    host, first, second, _, _, stages, head = shared_sources()
    # Give each job a distinct prediction consumer and its own output references.
    other_head = WorkStage.objects.create(
        execution=second, key="head", kind="predict", target="kcat"
    )
    artifacts = set(stages[2].payload["output_artifacts"] + stages[3].payload["output_artifacts"])
    ArtifactDependency.objects.filter(stage=head, artifact_id__in=artifacts).update(
        stage=other_head
    )
    head.dependencies.remove(*stages[2:])
    other_head.dependencies.add(*stages[2:])
    for owner in [first, second]:
        JobProgressStage.objects.get_or_create(job=owner.job, target="kcat", stage_index=0)
    cohort = WorkStage.objects.get(pk=coalesce(cross_jobs=True)[0])
    host.budgets = {"cpu": 4, "ram": 8 * 2**30, "disk": 8 * 2**30}
    host.telemetry = {
        "cpu_available": 4,
        "ram_available": 16 * 2**30,
        "disk_free": 32 * 2**30,
        "gpu_uuid": "gpu",
        "vram_total": 20 * 2**30,
        "vram_free": 20 * 2**30,
    }
    host.save()
    with override_settings(EXECUTION_V2_ENABLED=True):
        admitted = control.claim(host.pk)
        assert admitted and admitted["stage_id"] == str(cohort.pk)
        assert admitted["public_id"].startswith("embed-")
        refresh(cohort.execution)
        for owner in [first, second]:
            owner.job.refresh_from_db()
            assert owner.job.status == "Processing" and owner.job.start_time
            assert execution_status(owner.job)["execution_state"] == "running"
            public = JobProgressStage.objects.get(job=owner.job, target="kcat")
            assert public.embedding_total == 16 and public.embedding_state == "running"
    cohort.state = "completed"
    cohort.save(update_fields=["state"])
    refresh(cohort.execution)
    assert list(JobProgressStage.objects.values_list("embedding_computed", flat=True)) == [16, 16]


def test_cohort_never_changes_molecular_position_context_or_unprofiled_bounds():
    _, _, original, large, stages, _ = setup_cohort()
    large.settings["step"] = "molecular"
    large.save(update_fields=["settings"])
    assert coalesce() == []
    large.settings["step"] = "eitlem_esm1v"
    large.limits = {"sequences": 4, "max_length": 1024}
    large.save(update_fields=["settings", "limits"])
    assert coalesce() == []


def test_cohort_respects_maximum_work_size():
    _, _, _, large, _, _ = setup_cohort()
    large.limits["sequences"] = 16
    large.save(update_fields=["limits"])
    ids = coalesce()
    assert len(ids) == 2
    assert all(WorkStage.objects.get(pk=i).payload["shape"]["sequences"] == 16 for i in ids)


def test_long_stage_does_not_fragment_other_compatible_measured_work():
    _, _, _, large, sources, head = setup_cohort()
    large.limits["max_length"] = 100
    large.save(update_fields=["limits"])
    long = sources[1]
    long.payload["shape"]["max_length"] = 150
    long.save(update_fields=["payload"])
    unchanged = deepcopy(long.payload)
    ids = coalesce()
    assert len(ids) == 1
    cohort = WorkStage.objects.get(pk=ids[0])
    assert cohort.payload["shape"]["sequences"] == 24
    assert cohort.payload["shape"]["max_length"] <= 100
    long.refresh_from_db()
    assert long.state == "ready" and long.payload == unchanged
    assert set(head.dependencies.values_list("pk", flat=True)) == {cohort.pk, long.pk}
    assert Artifact.objects.filter(producer=long).count() == 8
    assert Artifact.objects.filter(producer=cohort).count() == 24


def test_batch_creation_budget_leaves_original_work_eligible_for_following_pass():
    _, _, _, large, stages, _ = setup_cohort()
    large.limits["sequences"] = 16
    large.save(update_fields=["limits"])
    first = coalesce(batch_limit=1)
    assert len(first) == 1
    assert WorkStage.objects.filter(pk__in=[s.pk for s in stages], state="ready").count() == 2
    second = coalesce(batch_limit=1)
    assert len(second) == 1 and second != first
    assert not WorkStage.objects.filter(pk__in=[s.pk for s in stages], state="ready").exists()


def test_cohort_never_bypasses_missing_input_or_changed_options():
    _, _, _, _, stages, _ = setup_cohort()
    for i, stage in enumerate(stages):
        stage.payload["extra_option"] = i
        stage.save(update_fields=["payload"])
    assert coalesce() == []


def test_cohort_oom_restores_small_units_and_preserves_committed_artifacts():
    from api.execution.cohorts import split_cohort
    from api.execution.models import ArtifactLocation, ExecutionOutbox

    host, _, original, large, stages, head = setup_cohort()
    cohort = WorkStage.objects.get(pk=coalesce()[0])
    artifact = Artifact.objects.filter(producer=cohort).first()
    artifact.checksum = "e" * 64
    artifact.size = 12
    artifact.save(update_fields=["checksum", "size"])
    ArtifactLocation.objects.create(artifact=artifact, host=host, ready=True)
    third = WorkStage.objects.create(execution=head.execution, key="later-consumer", kind="predict")
    third.dependencies.add(cohort)
    ArtifactDependency.objects.create(stage=third, artifact=artifact, path="same.npy")
    attempt = ExecutionAttempt.objects.create(
        stage=cohort,
        host=host,
        state="running",
        profile_fingerprint=large.fingerprint,
        lease_until=timezone.now(),
    )
    control.fail(
        host.pk, attempt.pk, {"token": str(attempt.token), "stopped": True, "category": "oom"}
    )
    cohort.refresh_from_db()
    large.refresh_from_db()
    assert cohort.state == "recovering" and large.suspended
    assert ExecutionOutbox.objects.filter(event__split_cohort=str(cohort.pk)).exists()
    assert split_cohort(cohort.pk)
    assert not split_cohort(cohort.pk)
    cohort.refresh_from_db()
    assert cohort.state == "canceled"
    assert set(head.dependencies.values_list("pk", flat=True)) == {s.pk for s in stages}
    assert WorkStage.objects.filter(pk__in=[s.pk for s in stages], state="ready").count() == 4
    artifact.refresh_from_db()
    assert artifact.checksum == "e" * 64 and artifact.locations.get().ready
    assert set(third.dependencies.values_list("pk", flat=True)) == {artifact.producer_id}
    original.refresh_from_db()
    assert not original.suspended


def test_suspended_cohort_profile_does_not_strand_other_unstarted_cohorts():
    _, _, _, large, stages, _ = setup_cohort()
    ident = coalesce()[0]
    large.suspended = True
    large.save(update_fields=["suspended"])
    assert coalesce() == []
    assert WorkStage.objects.get(pk=ident).state == "canceled"
    assert WorkStage.objects.filter(pk__in=[s.pk for s in stages], state="ready").count() == 4


def test_recovery_cohort_splits_without_reopening_original_completed_job():
    from api.execution.cohorts import split_cohort

    host, old, _, large, stages, _ = setup_cohort()
    original = WorkStage.objects.get(pk=coalesce()[0])
    original.state = "completed"
    original.save(update_fields=["state"])
    old.job.status = "Completed"
    old.job.save(update_fields=["status"])
    job = Job.objects.create(public_id="newrecover", prediction_type="kcat", status="Processing")
    execution = control.ingest(job.public_id, old.specification, "f" * 64, paused=False)
    payload = deepcopy(original.payload)
    payload["recovery_origin"] = str(original.pk)
    recovery = WorkStage.objects.create(
        execution=execution,
        key="recovery",
        kind="embedding",
        method=original.method,
        target=original.target,
        profile=large,
        payload=payload,
        state="running",
    )
    Artifact.objects.filter(producer=original).update(producer=recovery)
    consumer = WorkStage.objects.create(execution=execution, key="prediction", kind="predict")
    consumer.dependencies.add(recovery)
    attempt = ExecutionAttempt.objects.create(
        stage=recovery, host=host, profile_fingerprint=large.fingerprint, lease_until=timezone.now()
    )
    control.fail(
        host.pk, attempt.pk, {"token": str(attempt.token), "stopped": True, "category": "oom"}
    )
    assert split_cohort(recovery.pk)
    small = list(execution.stages.filter(kind="embedding", state="ready"))
    assert len(small) == 2 and all(s.payload["shape"]["sequences"] == 16 for s in small)
    assert sum(Artifact.objects.filter(producer=s).count() for s in small) == 32
    assert set(consumer.dependencies.values_list("pk", flat=True)) == {s.pk for s in small}
    old.job.refresh_from_db()
    assert old.job.status == "Completed"
    assert WorkStage.objects.filter(pk__in=[s.pk for s in stages], state="canceled").count() == 4


def test_cpu_pool_cohort_preserves_shared_and_per_sequence_inputs():
    _, _, parent, large, stages, _ = setup_cohort()
    for profile in [parent, large]:
        profile.stage_kind = "predict"
        profile.method = "KinForm-pool"
        profile.resources["vram"] = 0
        profile.settings["step"] = "kinform_pool"
        profile.save()
    common = Artifact.objects.create(id="c" * 64, identity={}, size=12, checksum="d" * 64)
    for stage in stages:
        stage.kind = "predict"
        stage.method = "KinForm-pool"
        stage.payload["step"] = "kinform_pool"
        stage.save()
        ArtifactDependency.objects.create(stage=stage, artifact=common, path="binding.tsv")
        sid = next(iter(stage.payload["seq_id_to_seq"]))
        residue = Artifact.objects.create(
            id=control.digest("residue" + sid), identity={}, size=12, checksum="e" * 64
        )
        ArtifactDependency.objects.create(stage=stage, artifact=residue, path=f"residues/{sid}.npy")
    stage = WorkStage.objects.get(pk=coalesce()[0])
    assert stage.kind == "predict" and stage.method == "KinForm-pool"
    assert stage.artifacts.count() == 5
    assert stage.artifacts.get(path="binding.tsv").artifact_id == common.pk


def test_cpu_pool_cohort_keeps_distinct_binding_contexts_separate():
    _, _, parent, large, stages, _ = setup_cohort()
    for profile in [parent, large]:
        profile.stage_kind = "predict"
        profile.method = "KinForm-pool"
        profile.resources["vram"] = 0
        profile.settings["step"] = "kinform_pool"
        profile.save()
    for index, stage in enumerate(stages):
        stage.kind = "predict"
        stage.method = "KinForm-pool"
        stage.payload["step"] = "kinform_pool"
        stage.save()
        artifact, _ = Artifact.objects.get_or_create(
            id=control.digest("binding" + str(index // 2)), defaults={"identity": {}, "size": 12}
        )
        ArtifactDependency.objects.create(stage=stage, artifact=artifact, path="binding.tsv")
    ids = coalesce()
    assert len(ids) == 2 and all(
        WorkStage.objects.get(pk=i).payload["shape"]["sequences"] == 16 for i in ids
    )


def test_recovered_binding_cohort_splits_original_tables_instead_of_sequence_filenames():
    from api.execution.cohorts import split_cohort

    host, old, parent, large, _, _ = setup_cohort()
    parent.settings["step"] = large.settings["step"] = "kinform_pseq2sites"
    parent.save()
    large.save()
    original = WorkStage.objects.get(pk=coalesce()[0])
    original.state = "completed"
    original.save()
    old.job.status = "Completed"
    old.job.save()
    job = Job.objects.create(public_id="bindretry", prediction_type="kcat", status="Processing")
    execution = control.ingest(job.public_id, old.specification, "f" * 64, paused=False)
    mapping = original.payload["seq_id_to_seq"]
    outputs = []
    for i in range(2):
        members = list(mapping)[i * 16 : (i + 1) * 16]
        identity = {"context": {"sequences": members}, "partition": i}
        ident = control.digest(identity)
        outputs.append({"id": ident, "identity": identity, "path": "parts/" + ident + ".tsv"})
    payload = {
        **deepcopy(original.payload),
        "step": "kinform_pseq2sites",
        "outputs": outputs,
        "output_artifacts": [o["id"] for o in outputs],
    }
    recovery = WorkStage.objects.create(
        execution=execution,
        key="binding-recovery",
        kind="embedding",
        method=parent.method,
        target="kcat",
        profile=large,
        state="recovering",
        payload=payload,
    )
    for output in outputs:
        Artifact.objects.create(
            id=output["id"], identity=output["identity"], producer=recovery, size=0
        )
    assert split_cohort(recovery.pk)
    small = list(execution.stages.filter(kind="embedding", state="ready"))
    assert len(small) == 2
    assert all(
        s.payload["shape"]["sequences"] == 16 and len(s.payload["outputs"]) == 1 for s in small
    )
    assert {o["id"] for s in small for o in s.payload["outputs"]} == set(
        payload["output_artifacts"]
    )
    assert all(
        set(s.payload["seq_id_to_seq"])
        == set(s.payload["outputs"][0]["identity"]["context"]["sequences"])
        for s in small
    )
    old.job.refresh_from_db()
    assert old.job.status == "Completed"
