"""Durable compatibility facade for the existing GPU service endpoints."""

from pathlib import Path

from django.conf import settings
from django.db import connection, transaction

from api.execution.control import digest
from api.execution.models import EmbeddingSubmission, ExecutionSpec, WorkStage
from api.execution.planning import attach, embedding_stage, kinform_graph, sequence_id
from api.services.embedding_plan_service import expected_paths_by_seq


@transaction.atomic
def submit(data):
    from api.execution.control import require_postgres

    require_postgres()
    with connection.cursor() as cursor:
        cursor.execute("SELECT pg_advisory_xact_lock(%s)", [481726392])
    method = data["method_key"]
    target = data["target"]
    from api.methods.registry import get

    if target not in get(method).supports:
        raise ValueError("Unsupported embedding method/target")
    if method not in settings.EXECUTION_MODEL_VERSIONS:
        raise ValueError("Embedding model identity is not configured")
    mapping = data["seq_id_to_seq"]
    if not mapping or not all(sequence_id(sequence) == sid for sid, sequence in mapping.items()):
        raise ValueError("Sequence identity does not match the durable registry")
    spec = {
        "model_versions": {method: settings.EXECUTION_MODEL_VERSIONS[method]},
        "targets": [target],
        "methods": {target: method},
        "embedding_request": data,
    }
    request_id = data.get("request_id")
    if request_id is not None and (
        not isinstance(request_id, str) or not 1 <= len(request_id) <= 128
    ):
        raise ValueError("Invalid embedding submission identifier")
    key = digest({"embedding_request_id": request_id}) if request_id else digest(spec)
    existing = EmbeddingSubmission.objects.filter(request_key=key).first()
    if existing:
        if existing.payload != data:
            raise ValueError("Conflicting embedding submission identifier")
        return {"job_id": str(existing.pk), "status": status(existing)["status"]}
    execution = ExecutionSpec.objects.create(
        job=None,
        specification=spec,
        checksum=digest(spec),
        input_checksum=digest(mapping),
        paused=settings.EXECUTION_START_PAUSED,
    )
    if method.startswith("KinForm"):
        outputs = kinform_graph(execution, method, target, mapping)
    else:
        from api.execution.planning import PROTEIN_STEPS

        if method not in PROTEIN_STEPS:
            raise ValueError("Unsupported GPU embedding representation")
        expected = expected_paths_by_seq(
            method_key=method,
            target=target,
            seq_ids=list(mapping),
            media_path=Path("/app/media"),
            env={},
        )
        paths = {
            sid: {str(Path(path).relative_to("/app/media")) for path in values}
            for sid, values in expected.items()
        }
        if not paths:
            raise ValueError("Embedding output specification is incomplete")
        outputs = embedding_stage(
            execution, method, target, mapping, paths, {"step": PROTEIN_STEPS[method]}
        )
    # A shared producer can belong to another request, including one that has
    # already finished. Retain explicit dependencies for this submission too.
    barrier = WorkStage.objects.create(
        execution=execution, key="embedding-request", kind="barrier", method=method, target=target
    )
    for artifact, path, _ in outputs:
        attach(barrier, artifact, path)
    submission = EmbeddingSubmission.objects.create(
        request_key=key, payload=data, execution=execution
    )
    return {"job_id": str(submission.pk), "status": "queued"}


def status(submission):
    stages = list(submission.execution.stages.all())
    barrier = next((stage for stage in stages if stage.key == "embedding-request"), None)
    ready = (
        barrier is not None
        and not barrier.dependencies.exclude(state="completed").exists()
        and barrier.artifacts.count()
        == barrier.artifacts.filter(artifact__locations__ready=True).values("pk").distinct().count()
    )
    if ready and barrier.state != "completed":
        from django.utils import timezone

        WorkStage.objects.filter(pk=barrier.pk, state="ready").update(
            state="completed", completed_at=timezone.now()
        )
        barrier.state = "completed"
    state = (
        "done"
        if ready or (stages and all(stage.state == "completed" for stage in stages))
        else "running"
        if any(stage.state == "running" for stage in stages)
        else "queued"
    )
    return {
        "job_id": str(submission.pk),
        "status": state,
        "method_key": submission.payload["method_key"],
        "target": submission.payload["target"],
        "profile": submission.payload.get("profile"),
        "step_work": submission.payload.get("step_work", {}),
        "waiting_reason": next((s.waiting_reason for s in stages if s.waiting_reason), None),
        "error": None,
    }


def dispatch(operation, data):
    if operation == "embed-submit":
        return submit(data)
    if operation == "embed-health":
        for submission in EmbeddingSubmission.objects.filter(
            execution__stages__kind="barrier", execution__stages__state="ready"
        ).select_related("execution"):
            status(submission)
        active = (
            EmbeddingSubmission.objects.filter(execution__stages__state="running")
            .distinct()
            .count()
        )
        queued = (
            EmbeddingSubmission.objects.filter(execution__stages__state__in=["ready", "held"])
            .exclude(execution__stages__state="running")
            .distinct()
            .count()
        )
        return {"active_jobs": active, "queued_jobs": queued}
    submission = EmbeddingSubmission.objects.select_related("execution").get(pk=data["job_id"])
    if operation == "embed-status":
        return status(submission)
    if operation == "embed-logs":
        # Durable stage outcomes provide diagnostics without exposing the execution credentials.
        return {
            "job_id": str(submission.pk),
            "lines": [
                f"{stage.kind} {stage.state} {stage.waiting_reason}"
                for stage in submission.execution.stages.order_by("created_at")
            ],
        }
    raise ValueError("Unknown embedding operation")
