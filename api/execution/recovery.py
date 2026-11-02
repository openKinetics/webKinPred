"""Recover immutable artifacts in shared batches without reopening completed jobs."""

from collections import defaultdict
from copy import deepcopy
from pathlib import PurePosixPath

from django.utils import timezone

from api.execution.control import digest
from api.execution.models import Artifact, ExecutionAttempt, RecoveryBatch, WorkStage
from api.execution.planning import attach, graph_transaction
from api.execution.resources import LIVE_STATES

FINISHED = ("completed", "canceled")


def _live(stage):
    return ExecutionAttempt.objects.filter(stage=stage, state__in=LIVE_STATES).exists()


def _recoverable(stage):
    return stage and (
        stage.kind == "embedding"
        or stage.payload.get("step") in {"kinform_pool", "kinform_binding_merge"}
    )


def _origin(stage):
    origin = stage.payload.get("recovery_origin")
    return (
        WorkStage.objects.select_related("profile", "execution").get(pk=origin) if origin else stage
    )


def _context(source):
    payload = {
        k: v
        for k, v in source.payload.items()
        if k
        not in {
            "outputs",
            "output_artifacts",
            "recovery_origin",
            "recovery_group",
            "admission_wait_started_at",
        }
    }
    identity = (
        source.profile.settings.get("model_identity")
        if source.profile_id
        else (source.execution.specification.get("model_versions", {}))
    )
    signature = digest(
        {
            "kind": source.kind,
            "method": source.method,
            "target": source.target,
            "profile": source.profile_id,
            "models": identity,
            "payload": payload,
            "inputs": sorted(source.artifacts.values_list("artifact_id", "path")),
            "dependencies": sorted(
                str(pk) for pk in source.dependencies.values_list("pk", flat=True)
            ),
        }
    )
    return signature, payload


def _payload(template, outputs, source, signature):
    payload = deepcopy(template)
    payload.update(
        outputs=outputs,
        output_artifacts=[o["id"] for o in outputs],
        recovery_origin=str(source.pk),
        recovery_group=signature,
    )
    mapping = payload.get("seq_id_to_seq")
    wanted = {PurePosixPath(o["path"]).stem for o in outputs}
    if mapping and wanted <= mapping.keys():
        # Per-sequence proteins can be narrowed. Binding-table coverage and
        # molecular rows retain their complete legacy execution context.
        payload["seq_id_to_seq"] = {sid: seq for sid, seq in mapping.items() if sid in wanted}
        payload["shape"] = {
            **payload.get("shape", {}),
            "sequences": len(wanted),
            "max_length": max(len(mapping[sid]) for sid in wanted),
        }
    return payload


def _inputs(source, restored):
    mapping = source.payload.get("seq_id_to_seq", {})
    selected = restored.payload.get("seq_id_to_seq", {})
    for required in source.artifacts.select_related("artifact"):
        sid = PurePosixPath(required.path).stem
        if sid in mapping and sid not in selected:
            continue
        attach(restored, required.artifact, required.path)
    restored.dependencies.add(*source.dependencies.exclude(pk=restored.pk))


def _schedule(execution, entries):
    groups = defaultdict(list)
    for artifact, producer, output in entries:
        source = _origin(producer)
        signature, template = _context(source)
        groups[signature].append((artifact, source, output, template, producer.priority))
    replacements = defaultdict(set)
    for signature, group in groups.items():
        registry, _ = RecoveryBatch.objects.get_or_create(
            pk=signature, defaults={"source": group[0][1]}
        )
        registry = RecoveryBatch.objects.select_for_update().get(pk=signature)
        restored = (
            (
                WorkStage.objects.select_for_update(of=("self",))
                .select_related("execution__job")
                .get(pk=registry.stage_id)
            )
            if registry.stage_id
            else None
        )
        mutable = (
            restored is not None
            and restored.state == "ready"
            and not _live(restored)
            and not restored.execution.paused
            and (
                not restored.execution.job_id
                or restored.execution.job.status not in {"Completed", "Failed"}
            )
        )
        source = registry.source
        _, template = _context(source)
        outputs = {o["id"]: o for o in restored.payload["outputs"]} if mutable else {}
        outputs.update({a.pk: out for a, _, out, _, _ in group})
        payload = _payload(template, list(outputs.values()), source, signature)
        if not mutable:
            registry.generation += 1
            restored = WorkStage.objects.create(
                execution=execution,
                key=f"restore-batch:{signature}:{registry.generation}",
                kind=source.kind,
                method=source.method,
                target=source.target,
                profile=source.profile,
                payload=payload,
                priority=max(execution.priority, max(row[4] for row in group)),
            )
            registry.stage = restored
            registry.save(update_fields=["stage", "generation"])
        else:
            restored.payload = payload
            restored.profile = source.profile
            restored.save(update_fields=["payload", "profile"])
        _inputs(source, restored)
        for artifact, _, _, _, _ in group:
            old = artifact.producer_id
            artifact.producer = restored
            if artifact.checksum:
                artifact.metadata = {
                    **artifact.metadata,
                    "previous_checksums": [
                        *artifact.metadata.get("previous_checksums", []),
                        artifact.checksum,
                    ],
                }
                artifact.checksum, artifact.size = "", 0
            artifact.save(update_fields=["producer", "checksum", "size", "metadata"])
            for consumer in WorkStage.objects.filter(artifacts__artifact=artifact).exclude(
                state__in=FINISHED
            ):
                if consumer.pk != restored.pk:
                    consumer.dependencies.add(restored)
            replacements[old].add(restored.pk)
    return replacements


@graph_transaction
def missing_dependencies(stage_id, host_id):
    stage = WorkStage.objects.select_for_update().get(pk=stage_id)
    if stage.state != "recovering":
        return
    missing, unrecoverable = [], False
    for dependency in stage.artifacts.select_related(
        "artifact__producer__profile", "artifact__producer__execution"
    ):
        artifact = Artifact.objects.select_for_update().get(pk=dependency.artifact_id)
        original = dependency.artifact.producer
        # materialize() invalidates individual missing locations explicitly.
        # An unspecified runtime cache error must not invalidate all host copies.
        if artifact.locations.filter(ready=True).exists():
            continue
        if not artifact.regenerable or not _recoverable(original):
            unrecoverable = True
            continue
        if _live(original) or original.state == "held":
            stage.dependencies.add(original)
            continue
        output = next(
            (v for v in original.payload.get("outputs", []) if v["id"] == artifact.pk), None
        )
        if output is None:
            unrecoverable = True
            continue
        if original.state not in FINISHED and original.execution.job_id != stage.execution.job_id:
            owner = original.execution.job
            if owner is None or owner.status not in {"Completed", "Failed"}:
                stage.dependencies.add(original)
                continue
        missing.append((artifact, original, output))
    _schedule(stage.execution, missing)
    stage.state = "held" if unrecoverable else "ready"
    stage.waiting_reason = "artifact_requires_attention" if unrecoverable else "waiting_embeddings"
    stage.save(update_fields=["state", "waiting_reason"])


@graph_transaction
def reconcile_pending(limit=256):
    stages = list(
        WorkStage.objects.filter(key__startswith="restore:", state="ready")
        .exclude(execution__job__status__in=["Completed", "Failed"])
        .select_for_update(skip_locked=True, of=("self",))
        .select_related("execution__job", "profile")
        .order_by("created_at")[:limit]
    )
    grouped, retired, reused = defaultdict(list), [], []
    replacements = defaultdict(set)
    for stage in stages:
        if _live(stage):
            continue
        owned, needs = [], []
        for output in stage.payload.get("outputs", []):
            artifact = Artifact.objects.select_for_update().get(pk=output["id"])
            if artifact.producer_id != stage.pk:
                if artifact.producer_id:
                    replacements[stage.pk].add(artifact.producer_id)
                continue
            owned.append(artifact)
            if not artifact.locations.filter(ready=True).exists():
                needs.append((artifact, stage, output))
        if owned and not needs:
            stage.state, stage.completed_at = "completed", timezone.now()
            stage.checkpoint = {**stage.checkpoint, "reused_verified_outputs": True}
            stage.save(update_fields=["state", "completed_at", "checkpoint"])
            reused.append(str(stage.pk))
        else:
            grouped[stage.execution_id].extend(needs)
            retired.append(stage)
    for execution_id, entries in grouped.items():
        if entries:
            execution = next(s.execution for s in retired if s.execution_id == execution_id)
            for old, new in _schedule(execution, entries).items():
                replacements[old].update(new)
    for old in retired:
        if old.outputs.exists():
            raise RuntimeError("Cannot retire a recovery while it owns an artifact")
        for consumer in old.dependants.exclude(state__in=FINISHED):
            consumer.dependencies.remove(old)
            consumer.dependencies.add(*replacements[old.pk])
        old.state, old.waiting_reason = "canceled", "recovery_rebatched"
        old.save(update_fields=["state", "waiting_reason"])
    return {
        "examined": len(stages),
        "retired": len(retired),
        "reused": len(reused),
        "batch_stages": len(set().union(*replacements.values())) if replacements else 0,
    }
