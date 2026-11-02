"""Amortize model startup over measured, bounded protein work within one job.

Scientific artifact identities and consumer paths stay unchanged. Only untouched
producer stages are replaced; molecular batches and active/retried stages retain
all of their original boundaries.
"""

import logging
from collections import defaultdict
from copy import deepcopy
from datetime import timedelta

from django.db import connection, transaction
from django.db.models import Exists, OuterRef
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
    }
)


def merged_payload(stages, profile):
    """Reject differing non-sequence options and every unmeasured shape."""
    payload = deepcopy(stages[0].payload)
    varying = {"seq_id_to_seq", "outputs", "output_artifacts", "shape", "admission_wait_started_at"}
    fixed = {k: v for k, v in payload.items() if k not in varying}
    sequences, outputs, paths, shape = {}, {}, {}, {}
    for stage in stages:
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
def coalesce(limit=128):
    """Bounded reconciliation. Empty until measured cohort profiles are enabled."""
    if connection.vendor != "postgresql":
        return []
    with connection.cursor() as cursor:
        cursor.execute("SET LOCAL statement_timeout = %s", ["3000ms"])
        cursor.execute("SELECT pg_try_advisory_xact_lock(%s)", [481726392])
        if not cursor.fetchone()[0]:
            return []
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
        if (
            profile.stage_kind != "embedding"
            or not profile.resources.get("vram")
            or profile.settings.get("step") not in STEPS
        ):
            continue
        sources = ResourceProfile.objects.filter(
            fingerprint__in=profile.settings["cohort_sources"],
            validated=True,
            suspended=False,
            stage_kind=profile.stage_kind,
            method=profile.method,
            image=profile.image,
            settings__model_identity=profile.settings.get("model_identity"),
            settings__step=profile.settings.get("step"),
        )
        blocked = WorkStage.dependencies.through.objects.filter(
            from_workstage_id=OuterRef("pk")
        ).exclude(to_workstage__state="completed")
        attempts = ExecutionAttempt.objects.filter(stage_id=OuterRef("pk"))
        candidates = list(
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
            .filter(tried=False, blocked=False, has_inputs=False)
            .order_by(
                "execution__background",
                "-execution__priority",
                Coalesce("execution__job__submission_time", "execution__created_at"),
                "created_at",
            )[: max(2, min(limit, 512))]
        )
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
    stage = WorkStage.objects.create(
        execution_id=stages[0].execution_id,
        key="cohort:" + digest(sorted(map(str, ids))),
        kind="embedding",
        method=stages[0].method,
        target=stages[0].target,
        profile=profile,
        payload=payload,
        priority=max(s.priority for s in stages),
        created_at=min(s.created_at for s in stages),
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
