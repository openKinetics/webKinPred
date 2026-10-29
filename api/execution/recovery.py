"""Rebuild lost embedding dependencies without reopening completed user jobs."""

from copy import deepcopy

from api.execution.models import Artifact, ExecutionAttempt, WorkStage
from api.execution.planning import attach, graph_transaction
from api.execution.resources import LIVE_STATES


@graph_transaction
def missing_dependencies(stage_id, host_id):
    stage = WorkStage.objects.select_for_update().get(pk=stage_id)
    if stage.state != "recovering":
        return
    unrecoverable = False
    for dependency in stage.artifacts.select_related("artifact__producer"):
        artifact = Artifact.objects.select_for_update().get(pk=dependency.artifact_id)
        original = artifact.producer
        if host_id is None and artifact.locations.filter(ready=True).exists():
            continue
        if (
            not artifact.regenerable
            or original is None
            or not (
                original.kind == "embedding"
                or original.payload.get("step") in {"kinform_pool", "kinform_binding_merge"}
            )
        ):
            if not artifact.locations.filter(ready=True).exists():
                unrecoverable = True
            continue
        # The supervisor reports its own invalid/missing private dependency. Another
        # verified host copy can satisfy the next attempt without recomputation.
        if host_id is not None:
            artifact.locations.filter(host_id=host_id).update(ready=False)
        if artifact.locations.filter(ready=True).exists():
            continue
        if ExecutionAttempt.objects.filter(stage=original, state__in=LIVE_STATES).exists():
            stage.dependencies.add(original)
            continue
        output = next(
            (value for value in original.payload.get("outputs", []) if value["id"] == artifact.pk),
            None,
        )
        if output is None:
            unrecoverable = True
            continue
        payload = deepcopy(original.payload)
        payload["outputs"] = [output]
        payload["output_artifacts"] = [artifact.pk]
        restored, _ = WorkStage.objects.get_or_create(
            execution=stage.execution,
            key="restore:" + artifact.pk,
            defaults={
                "kind": original.kind,
                "method": original.method,
                "target": original.target,
                "payload": payload,
                "profile": original.profile,
            },
        )
        if restored.pk != original.pk:
            restored.dependencies.add(*original.dependencies.all())
            for required in original.artifacts.select_related("artifact"):
                attach(restored, required.artifact, required.path)
        if restored.state == "completed":
            restored.state = "ready"
            restored.completed_at = None
            restored.save(update_fields=["state", "completed_at"])
        artifact.producer = restored
        # Every verified copy has been lost. Recomputing a floating-point
        # representation need not reproduce its previous serialization bytes.
        # Preserve the old checksum as evidence; new consumers wait for the new
        # producer and receive a newly verified immutable blob.
        if artifact.checksum:
            artifact.metadata = {
                **artifact.metadata,
                "previous_checksums": [
                    *artifact.metadata.get("previous_checksums", []),
                    artifact.checksum,
                ],
            }
            artifact.checksum = ""
            artifact.size = 0
        artifact.save(update_fields=["producer", "checksum", "size", "metadata"])
        for consumer in WorkStage.objects.filter(artifacts__artifact=artifact).exclude(
            state="completed"
        ):
            if consumer.pk != restored.pk:
                consumer.dependencies.add(restored)
    stage.state = "held" if unrecoverable else "ready"
    stage.waiting_reason = "artifact_requires_attention" if unrecoverable else "waiting_embeddings"
    stage.save(update_fields=["state", "waiting_reason"])
