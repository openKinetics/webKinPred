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
