"""Amortize model startup over measured, bounded protein work within one job.

Scientific artifact identities and consumer paths stay unchanged. Only untouched
producer stages are replaced; molecular batches and active/retried stages retain
all of their original boundaries.
"""

import logging
from collections import defaultdict
from copy import deepcopy
from datetime import timedelta
from pathlib import PurePosixPath

from django.db import connection, transaction
from django.db.models import Exists, OuterRef, Prefetch
from django.db.models.functions import Coalesce
from django.utils import timezone

from api.execution.control import digest
from api.execution.models import (
    Artifact,
    ArtifactDependency,
    ExecutionAttempt,
    ExecutorHost,
    ResourceProfile,
    WorkStage,
)

STEPS = frozenset(
    {
        "prot_t5_mean",
        "eitlem_esm1v",
        "omniesi_esm2",
        "catrange_esmc",
        "turnup_esm1b",
        "catpred_embed",
        "iecata_prot_t5_residues",
        "kinform_t5_residues",
        "kinform_esm2_residues",
        "kinform_esmc_residues",
        "kinform_pseq2sites",
    }
)
INPUT_STEPS = frozenset({"kinform_pseq2sites", "kinform_pool"})


def merged_payload(stages, profile):
    """Reject differing non-sequence options and every unmeasured shape."""
    payload = deepcopy(stages[0].payload)
    varying = {"seq_id_to_seq", "outputs", "output_artifacts", "shape", "admission_wait_started_at"}
    fixed = {k: v for k, v in payload.items() if k not in varying}
    sequences, outputs, paths, shape, inputs = {}, {}, {}, {}, {}
    for stage in stages:
        for dependency in getattr(stage, "cohort_inputs", []):
            if dependency.path in inputs and inputs[dependency.path] != dependency.artifact_id:
                return None
            inputs[dependency.path] = dependency.artifact_id
        if {k: v for k, v in stage.payload.items() if k not in varying} != fixed:
            return None
        for sid, sequence in stage.payload["seq_id_to_seq"].items():
            if sid in sequences and sequences[sid] != sequence:
                return None
            sequences[sid] = sequence
        for output in stage.payload["outputs"]:
            if (output["id"] in outputs and outputs[output["id"]] != output) or (
                output["path"] in paths and paths[output["path"]] != output["id"]
            ):
                return None
            outputs[output["id"]] = output
            paths[output["path"]] = output["id"]
        for key, value in stage.payload["shape"].items():
            shape[key] = max(shape.get(key, 0), value)
    shape["sequences"] = len(sequences)
    if not shape["sequences"] or any(
        shape.get(k, float("inf")) > v for k, v in profile.limits.items()
    ):
        return None
    payload.update(
        seq_id_to_seq=sequences,
        outputs=list(outputs.values()),
        output_artifacts=list(outputs),
        shape=shape,
    )
    return payload


@transaction.atomic
def coalesce(limit=16):
    """Bounded reconciliation. Empty until measured cohort profiles are enabled."""
    if connection.vendor != "postgresql":
        return []
    with connection.cursor() as cursor:
        cursor.execute("SET LOCAL statement_timeout = %s", ["3000ms"])
        cursor.execute("SELECT pg_try_advisory_xact_lock(%s)", [481726392])
        if not cursor.fetchone()[0]:
            return []
    for ident in WorkStage.objects.filter(
        state="ready",
        profile__suspended=True,
        profile__settings__cohort_sources__isnull=False,
    ).values_list("pk", flat=True)[:8]:
        split_cohort(ident)
    fingerprints = set()
    for capabilities in ExecutorHost.objects.filter(
        paused=False,
        reconciled=True,
        heartbeat_at__gte=timezone.now() - timedelta(seconds=10),
    ).values_list("capabilities", flat=True):
        fingerprints.update(capabilities.get("profiles", []))
    profiles = ResourceProfile.objects.filter(
        validated=True,
        suspended=False,
        fingerprint__in=fingerprints,
        settings__cohort_sources__isnull=False,
    )
    created = []
    for profile in profiles:
        gpu = (
            profile.stage_kind == "embedding"
            and profile.resources.get("vram")
            and profile.settings.get("step") in STEPS
        )
        pooling = (
            profile.stage_kind == "predict"
            and profile.method == "KinForm-pool"
            and not profile.resources.get("vram")
            and profile.settings.get("step") == "kinform_pool"
        )
        if not (gpu or pooling):
            continue
        sources = ResourceProfile.objects.filter(
            fingerprint__in=profile.settings["cohort_sources"],
            validated=True,
            suspended=False,
            stage_kind=profile.stage_kind,
            method=profile.method,
            image__in=[profile.image, profile.settings.get("cohort_parent_image", profile.image)],
            settings__model_identity=profile.settings.get("model_identity"),
            settings__step=profile.settings.get("step"),
        )
        blocked = WorkStage.dependencies.through.objects.filter(
            from_workstage_id=OuterRef("pk")
        ).exclude(to_workstage__state="completed")
        attempts = ExecutionAttempt.objects.filter(stage_id=OuterRef("pk"))
        query = (
            WorkStage.objects.select_for_update(skip_locked=True, of=("self",))
            .filter(
                state="ready",
                profile__in=sources,
                execution__paused=False,
                execution__job__status__in=["Pending", "Processing"],
                available_at__lte=timezone.now(),
                failures=0,
                target__in=profile.settings.get("targets", []),
            )
            .annotate(
                tried=Exists(attempts),
                blocked=Exists(blocked),
                has_inputs=Exists(ArtifactDependency.objects.filter(stage_id=OuterRef("pk"))),
            )
            .filter(tried=False, blocked=False)
            .order_by(
                "execution__background",
                "-execution__priority",
                Coalesce("execution__job__submission_time", "execution__created_at"),
                "created_at",
            )
        )
        if profile.settings.get("step") not in INPUT_STEPS:
            query = query.filter(has_inputs=False)
        # Lock/rank IDs before fetching manifests. Sorting all queued payloads
        # can spill gigabytes of immutable JSON for a sixteen-row planning pass.
        ids = list(query.values_list("pk", flat=True)[: max(2, min(limit, 512))])
        selected_rows = {
            stage.pk: stage
            for stage in WorkStage.objects.filter(pk__in=ids).prefetch_related(
                Prefetch(
                    "artifacts",
                    queryset=ArtifactDependency.objects.only("artifact_id", "path", "stage_id"),
                    to_attr="cohort_inputs",
                )
            )
        }
        candidates = [selected_rows[ident] for ident in ids]

        groups = defaultdict(list)
        for stage in candidates:
            if stage.payload.get("seq_id_to_seq") and stage.payload.get("outputs"):
                groups[(stage.execution_id, stage.target, stage.profile_id)].append(stage)
        for group in groups.values():
            selected = []
            for stage in group:
                trial = selected + [stage]
                if merged_payload(trial, profile) is None:
                    if len(selected) > 1:
                        created.append(replace(selected, profile))
                    selected = [stage] if merged_payload([stage], profile) else []
                else:
                    selected = trial
            if len(selected) > 1:
                created.append(replace(selected, profile))
    return [ident for ident in created if ident]


def replace(stages, profile):
    """Called with all source row locks; canceled rows cannot subsequently claim."""
    ids = [stage.pk for stage in stages]
    payload = merged_payload(stages, profile)
    artifacts = list(
        Artifact.objects.select_for_update()
        .filter(pk__in=payload["output_artifacts"])
        .order_by("pk")
    )
    if len(artifacts) != len(payload["output_artifacts"]) or any(
        a.producer_id not in ids for a in artifacts
    ):
        logging.getLogger(__name__).warning(
            "Protein cohort producer changed; retaining original stages",
            extra={"profile": profile.pk},
        )
        return None
    payload["cohort_source_stages"] = list(map(str, ids))
    stage = WorkStage.objects.create(
        execution_id=stages[0].execution_id,
        key="cohort:" + digest(sorted(map(str, ids))),
        kind=stages[0].kind,
        method=stages[0].method,
        target=stages[0].target,
        profile=profile,
        payload=payload,
        priority=max(s.priority for s in stages),
        created_at=min(s.created_at for s in stages),
    )
    inputs = {
        dependency.path: dependency.artifact_id
        for source in stages
        for dependency in getattr(source, "cohort_inputs", [])
    }
    ArtifactDependency.objects.bulk_create(
        [
            ArtifactDependency(stage=stage, artifact_id=ident, path=path)
            for path, ident in inputs.items()
        ]
    )
    edge = WorkStage.dependencies.through
    # Completed prerequisite links are retained for provenance and diagnostics.
    dependencies = set(
        edge.objects.filter(from_workstage_id__in=ids).values_list("to_workstage_id", flat=True)
    )
    stage.dependencies.add(*dependencies)
    consumers = set(
        edge.objects.filter(to_workstage_id__in=ids).values_list("from_workstage_id", flat=True)
    )
    edge.objects.bulk_create(
        [edge(from_workstage_id=consumer, to_workstage_id=stage.pk) for consumer in consumers],
        ignore_conflicts=True,
    )
    edge.objects.filter(to_workstage_id__in=ids).delete()
    Artifact.objects.filter(pk__in=payload["output_artifacts"], producer_id__in=ids).update(
        producer=stage
    )
    WorkStage.objects.filter(pk__in=ids, state="ready").update(
        state="canceled", waiting_reason="coalesced"
    )
    return str(stage.pk)


def split_cohort(ident):
    """Undo an enlarged unit after a confirmed OOM; retain committed checkpoints."""
    from api.execution.planning import graph_transaction
    from api.execution.resources import LIVE_STATES

    @graph_transaction
    def split():
        stage = (
            WorkStage.objects.select_for_update(of=("self",))
            .select_related("profile")
            .get(pk=ident)
        )
        if stage.state == "canceled":
            return False
        allowed = stage.state == "recovering" or (
            stage.state == "ready" and stage.profile.suspended
        )
        if not allowed or stage.attempts.filter(state__in=LIVE_STATES).exists():
            raise ValueError("Cohort must be stopped before splitting")
        ids = stage.payload["cohort_source_stages"]
        sources = list(WorkStage.objects.select_for_update().filter(pk__in=ids).order_by("pk"))
        if len(sources) != len(ids) or any(
            s.execution_id != stage.execution_id or s.state != "canceled" or s.attempts.exists()
            for s in sources
        ):
            # Recovery may be owned by a newer job after the original job
            # completed. Create small units in that job; never reopen historical
            # jobs or move their stopped attempts.
            parent = ResourceProfile.objects.filter(
                fingerprint__in=stage.profile.settings.get(
                    "cohort_sources", [s.profile.fingerprint for s in sources if s.profile_id]
                ),
                validated=True,
                stage_kind=stage.kind,
                method=stage.method,
                image=stage.profile.image,
                settings__model_identity=stage.profile.settings.get("model_identity"),
            ).first()
            if parent is None:
                stage.state = "held"
                stage.waiting_reason = "profiling_required"
                stage.save(update_fields=["state", "waiting_reason"])
                return False
            alternatives = ResourceProfile.objects.filter(
                fingerprint__in=stage.profile.settings.get("cohort_sources", []),
                validated=True,
                stage_kind=stage.kind,
                method=stage.method,
                settings__model_identity=stage.profile.settings.get("model_identity"),
            )
            parent = next(
                (
                    p
                    for p in alternatives
                    if all(
                        stage.payload["shape"].get(k, float("inf")) <= v
                        for k, v in p.limits.items()
                        if k != "sequences"
                    )
                ),
                parent,
            )
            mapping = list(stage.payload["seq_id_to_seq"].items())
            width = max(1, min(16, parent.limits["sequences"]))
            sources = []
            if stage.payload.get("step") == "kinform_pseq2sites":
                partitions = []
                for output in stage.payload["outputs"]:
                    members = output["identity"]["context"]["sequences"]
                    if (
                        not members
                        or len(members) > width
                        or len(set(members)) != len(members)
                        or not set(members) <= dict(mapping).keys()
                    ):
                        raise ValueError(
                            "Binding partition cannot fit its original measured profile"
                        )
                    partitions.append(({sid: dict(mapping)[sid] for sid in members}, [output]))
            else:
                partitions = []
                for offset in range(0, len(mapping), width):
                    subset = dict(mapping[offset : offset + width])
                    outputs = [
                        o
                        for o in stage.payload["outputs"]
                        if PurePosixPath(o["path"]).stem in subset
                    ]
                    partitions.append((subset, outputs))
            for ordinal, (subset, outputs) in enumerate(partitions):
                payload = deepcopy(stage.payload)
                for key in (
                    "cohort_source_stages",
                    "recovery_origin",
                    "recovery_group",
                    "admission_wait_started_at",
                ):
                    payload.pop(key, None)
                payload.update(
                    seq_id_to_seq=subset,
                    outputs=outputs,
                    output_artifacts=[o["id"] for o in outputs],
                    shape={
                        **payload["shape"],
                        "sequences": len(subset),
                        "max_length": max(map(len, subset.values())),
                    },
                )
                source = WorkStage.objects.create(
                    execution_id=stage.execution_id,
                    key=f"split:{stage.pk}:{ordinal}",
                    kind=stage.kind,
                    method=stage.method,
                    target=stage.target,
                    profile=parent,
                    payload=payload,
                    created_at=stage.created_at,
                )
                source.dependencies.add(*stage.dependencies.all())
                from api.execution.planning import attach

                for dependency in stage.artifacts.select_related("artifact"):
                    sid = PurePosixPath(dependency.path).stem
                    if sid not in stage.payload["seq_id_to_seq"] or sid in subset:
                        attach(source, dependency.artifact, dependency.path)
                sources.append(source)
            if {a for source in sources for a in source.payload["output_artifacts"]} != set(
                stage.payload["output_artifacts"]
            ):
                raise ValueError("Protein split cannot preserve output identities")
        for source in sources:
            Artifact.objects.filter(
                pk__in=source.payload["output_artifacts"], producer=stage
            ).update(producer=source)
            source.state = "ready"
            source.waiting_reason = "retrying"
            source.available_at = stage.available_at
            source.priority = max(source.priority, stage.priority)
            source.save(update_fields=["state", "waiting_reason", "available_at", "priority"])
        edge = WorkStage.dependencies.through
        consumers = list(stage.dependants.all())
        source_ids = {s.pk for s in sources}
        for consumer in consumers:
            required = set(
                consumer.artifacts.filter(artifact__producer_id__in=source_ids).values_list(
                    "artifact__producer_id", flat=True
                )
            )
            consumer.dependencies.add(*(required or source_ids))
        edge.objects.filter(to_workstage_id=stage.pk).delete()
        stage.state = "canceled"
        stage.waiting_reason = "cohort_split"
        stage.save(update_fields=["state", "waiting_reason"])
        from api.execution.progress import refresh

        refresh(stage.execution)
        return True

    return split()
