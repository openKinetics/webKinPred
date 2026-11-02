from copy import deepcopy

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
