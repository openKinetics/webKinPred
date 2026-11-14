"""Incremental deterministic graphs; no model is loaded in the control plane."""

from __future__ import annotations

import hashlib
import shutil
import time
from datetime import timedelta
from functools import wraps
from pathlib import Path

from django.db import IntegrityError, connection, transaction
from django.db.models import Exists, OuterRef, Q
from django.utils import timezone

from api.execution.artifacts import ArtifactStore, sha256_file
from api.execution.control import bind_profiles, digest
from api.execution.models import (
    Artifact,
    ArtifactDependency,
    ArtifactLocation,
    SequenceIdentity,
    WorkStage,
)
from api.execution.shapes import prediction_shape
from api.methods.registry import get as get_method
from api.services.embedding_plan_service import (
    expected_paths_by_seq,
    normalise_sequences_for_method,
)
from api.services.prediction_batch_service import build_sequence_batch_plan, build_target_batch_plan
from api.utils.convert_to_mol import convert_to_mol, substrate_as_smiles
from api.utils.substrate_expansion import split_substrate_list

MOLECULAR = {
    "UniKP": "UniKP",
    "KinForm-H": "KinForm",
    "KinForm-L": "KinForm",
    "CataPro": "CataPro",
    "CatRange": "CatRange",
}
EMBEDDING_STAGE_SEQUENCES = 16
PROTEIN_STEPS = {
    "UniKP": "prot_t5_mean",
    "CataPro": "prot_t5_mean",
    "EITLEM": "eitlem_esm1v",
    "TurNup": "turnup_esm1b",
    "CatPred": "catpred_embed",
    "OmniESI": "omniesi_esm2",
    "OmniESI-O2DENet": "omniesi_esm2",
    "IECata": "iecata_prot_t5_residues",
    "CatRange": "catrange_esmc",
}


def graph_transaction(function):
    @wraps(function)
    def wrapped(*args, **kwargs):
        with transaction.atomic():
            if connection.vendor == "postgresql":
                # Serialize producer publication across preparation and compatibility APIs.
                # Never hold this lock during model execution or artifact transfer.
                with connection.cursor() as cursor:
                    cursor.execute("SELECT pg_advisory_xact_lock(%s)", [481726392])
            return function(*args, **kwargs)

    return wrapped


def sequence_id(sequence):
    sha = hashlib.sha256(sequence.encode()).hexdigest()
    existing = SequenceIdentity.objects.filter(sha256=sha).first()
    if existing:
        if existing.seq != sequence:
            raise ValueError("Sequence checksum collision")
        return existing.pk
    suffix = 0
    while True:
        ident = sha[:12] + (f"_{suffix}" if suffix else "")
        try:
            with transaction.atomic():
                return SequenceIdentity.objects.create(
                    id=ident, sha256=sha, seq=sequence, length=len(sequence)
                ).pk
        except IntegrityError:
            existing = SequenceIdentity.objects.filter(sha256=sha).first()
            if existing:
                return existing.pk
            suffix += 1


def artifact_identity(spec, method, target, path, *, context=None):
    versions = spec["model_versions"]
    if method not in versions:
        raise ValueError("Missing pinned model identity for " + method)
    version = versions[method]
    family = str(Path(path).parent)
    # Sharing across methods requires an explicit pinned encoder representation.
    # A prediction-head checksum alone does not establish encoder compatibility.
    representation = version.get("representations", {}).get(family)
    return {
        "representation_version": "execution-v2",
        "method": None if representation is not None else method,
        # Only CatPred pools with target/checkpoint-specific learned weights.
        "target": target if method == "CatPred" else None,
        "models": representation if representation is not None else version,
        "path": path,
        "context": context,
    }


def declare(identity):
    from api.execution.artifact_policy import retention_for
    from api.execution.representation_aliases import alternate_identity

    alternate = alternate_identity(identity)
    if alternate is not None:
        primary_id, alternate_id = digest(identity), digest(alternate)
        live = (
            WorkStage.objects.filter(
                pk=OuterRef("producer_id"),
                state__in=["ready", "running"],
                execution__paused=False,
            )
            .exclude(execution__job__status__in=["Completed", "Failed"])
            .filter(Q(profile__isnull=True) | Q(profile__suspended=False))
        )
        ready = ArtifactLocation.objects.filter(artifact_id=OuterRef("pk"), ready=True)
        candidates = {
            a.pk: a
            for a in Artifact.objects.filter(pk__in=[primary_id, alternate_id]).annotate(
                alias_ready=Exists(ready), alias_live=Exists(live)
            )
        }
        # Existing manifests retain their exact artifact and producer identities.
        if primary_id in candidates:
            return candidates[primary_id]
        shared = candidates.get(alternate_id)
        if shared and ((shared.alias_ready and shared.checksum) or shared.alias_live):
            return shared
        try:
            with transaction.atomic():
                return Artifact.objects.create(
                    id=primary_id,
                    identity=identity,
                    checksum="",
                    size=0,
                    retention=retention_for(identity),
                )
        except IntegrityError:
            return Artifact.objects.get(pk=primary_id)
    artifact, _ = Artifact.objects.get_or_create(
        id=digest(identity),
        defaults={
            "identity": identity,
            "checksum": "",
            "size": 0,
            "retention": retention_for(identity),
        },
    )
    return artifact


def attach(stage, artifact, path):
    ArtifactDependency.objects.get_or_create(
        stage=stage, path=path, defaults={"artifact": artifact}
    )
    if artifact.producer_id and artifact.producer_id != stage.pk:
        stage.dependencies.add(artifact.producer_id)


def attach_many(stage, entries):
    entries = {path: artifact for artifact, path in entries}
    existing = dict(stage.artifacts.values_list("path", "artifact_id"))
    if any(
        path in existing and existing[path] != artifact.pk for path, artifact in entries.items()
    ):
        raise ValueError("Immutable stage input path changed")
    ArtifactDependency.objects.bulk_create(
        [
            ArtifactDependency(stage=stage, artifact=artifact, path=path)
            for path, artifact in entries.items()
            if path not in existing
        ],
        batch_size=256,
        ignore_conflicts=True,
    )
    stage.dependencies.add(
        *{
            artifact.producer_id
            for artifact in entries.values()
            if artifact.producer_id and artifact.producer_id != stage.pk
        }
    )


def embedding_sequences(descriptor, batch):
    """Avoid loading GPU models for rows the CPU adapter will reject before inference."""
    valid = []
    for index, sequence in enumerate(batch.sequences):
        if (
            not isinstance(sequence, str)
            or not sequence
            or any(c not in "ACDEFGHIKLMNPQRSTVWY" for c in sequence)
        ):
            continue
        accepted = True
        for column, kwarg in descriptor.col_to_kwarg.items():
            values = batch.call_kwargs.get(kwarg)
            if not isinstance(values, (list, tuple)) or index >= len(values):
                continue
            value = values[index]
            tokens = (
                value
                if isinstance(value, (list, tuple))
                else split_substrate_list(value)
                if column in {"Substrates", "Products"}
                else [value]
            )
            if not tokens or any(convert_to_mol(token) is None for token in tokens):
                accepted = False
                break
        if accepted:
            valid.append(sequence)
    return valid


def validated_prediction_frame(frame, descriptor):
    """Match single-molecule planning at the legacy runner boundary.

    Legacy generic runners split semicolons during chemistry validation. A
    single-molecule encoder cannot consume that text. Only private execution
    copies blank unsupported cells so those runners emit their normal invalid
    row result. Originals and valid chemistry (including dot components) remain
    unchanged; plural Substrates expansion retains its existing semantics.
    """
    cfg = descriptor.subprocess
    if not (
        cfg
        and cfg.validate_chemistry
        and descriptor.input_format == "single"
        and "Substrate" in descriptor.col_to_kwarg
        and "Substrate" in frame.columns
    ):
        return frame, []
    invalid = [i for i, value in enumerate(frame["Substrate"]) if convert_to_mol(value) is None]
    if not invalid:
        return frame, []
    copied = frame.copy()
    copied.loc[invalid, "Substrate"] = ""
    return copied, invalid


def attach_prediction_input(stage, artifact, path, *, repair=False):
    from api.execution.resources import LIVE_STATES

    existing = stage.artifacts.select_for_update().filter(path=path).first()
    if existing and existing.artifact_id != artifact.pk:
        if (
            not repair
            or stage.state != "held"
            or stage.attempts.filter(state__in=LIVE_STATES).exists()
        ):
            raise ValueError("Cannot change an existing prediction input without a stopped repair")
        existing.artifact = artifact
        existing.save(update_fields=["artifact"])
    else:
        attach(stage, artifact, path)


CLEANUP_WAIT_SECONDS = 120
CLEANUP_PENDING = "[Errno 11] Artifact cleanup is in progress"


CLEANUP_YIELD_SECONDS = 20
CLEANUP_STARVED_SECONDS = 60


def _yield_to_cleanup():
    """Leave the graph lock free between chunks while a host's deletions starve.

    A chunk holds the lock for its whole transaction and the collector waits at
    most 2.5 s for it. The 5,000-row requests of 14 November held it for 30-45 s
    per chunk with gaps of about a second, so production created no cleanup for
    over half an hour while 896 consumed residue matrices (1.2 GB) filled its
    disk. Supervisors ask for cleanups about every 16 s; wait until each host
    with pending hints has had one recently, or for at most 20 s.
    """
    from api.execution.models import ArtifactCleanup, ArtifactSweep

    deadline = time.monotonic() + CLEANUP_YIELD_SECONDS
    while True:
        recent = timezone.now() - timedelta(seconds=CLEANUP_STARVED_SECONDS)
        starved = [
            host
            for host, pending in ArtifactSweep.objects.values_list("host_id", "pending")
            if pending
            and not ArtifactCleanup.objects.filter(host_id=host)
            .filter(Q(created_at__gte=recent) | Q(completed_at__isnull=True))
            .exists()
        ]
        if not starved or time.monotonic() >= deadline:
            return
        time.sleep(1)


def _publish_after_cleanup(path, identity, host_id, store):
    """Publish a chunk once no artifact deletion is pending on ``host_id``.

    Publication refuses to run beside a pending deletion. Failing the chunk held
    the whole request for a minute and replanned its committed chunks. Production
    deletions are claimed and finished within a second, so retry briefly. The
    refusal rolls back publication's savepoint, which releases the graph lock,
    and nothing else in the chunk has run yet.
    """
    deadline = time.monotonic() + CLEANUP_WAIT_SECONDS
    while True:
        try:
            return publish_file(path, identity, host_id, store)
        except BlockingIOError as error:
            if str(error) != CLEANUP_PENDING or time.monotonic() >= deadline:
                raise
            time.sleep(1)


@graph_transaction
def publish_file(path, identity, host_id, store):
    from django.conf import settings

    from api.execution.models import ArtifactCleanup

    if ArtifactCleanup.objects.filter(host_id=host_id, completed_at__isnull=True).exists():
        raise BlockingIOError(11, "Artifact cleanup is in progress")

    checksum = sha256_file(path)
    with path.open("rb") as stream:
        store.publish(
            stream,
            checksum,
            path.stat().st_size,
            minimum_free=settings.EXECUTION_HOSTS.get(host_id, {})
            .get("budgets", {})
            .get("disk_headroom", 0),
        )
    artifact, _ = Artifact.objects.get_or_create(
        id=digest(identity),
        defaults={
            "identity": identity,
            "checksum": checksum,
            "size": path.stat().st_size,
            "regenerable": False,
            "retention": "permanent",
        },
    )
    if artifact.checksum != checksum:
        raise ValueError("Immutable input changed")
    ArtifactLocation.objects.update_or_create(
        artifact=artifact, host_id=host_id, defaults={"ready": True, "verified_at": timezone.now()}
    )
    return artifact


@graph_transaction
def embedding_stage(execution, method, target, seq_map, paths_by_seq, extra=None):
    """Attach existing producers and generate one bounded request for remaining outputs."""
    needed = []
    for sid, paths in paths_by_seq.items():
        for path in paths:
            artifact = declare(
                artifact_identity(
                    execution.specification,
                    method,
                    target,
                    path,
                    context=(extra or {}).get("contexts", {}).get(sid),
                )
            )
            needed.append((artifact, path, sid))
    missing = [
        (a, p, sid)
        for a, p, sid in needed
        if not a.producer_id and not a.locations.filter(ready=True).exists()
    ]
    ids = list(dict.fromkeys(sid for _, _, sid in missing))
    for offset in range(0, len(ids), EMBEDDING_STAGE_SEQUENCES):
        missing_ids = set(ids[offset : offset + EMBEDDING_STAGE_SEQUENCES])
        partition = [(a, p, sid) for a, p, sid in missing if sid in missing_ids]
        pending_sequences = {sid: seq for sid, seq in seq_map.items() if sid in missing_ids}
        key = "embedding:" + digest(sorted(a.pk for a, _, _ in partition))
        stage, _ = WorkStage.objects.get_or_create(
            execution=execution,
            key=key,
            defaults={
                "kind": "embedding",
                "method": method,
                "target": target,
                "payload": {
                    "seq_id_to_seq": pending_sequences,
                    "outputs": [
                        {"id": a.pk, "identity": a.identity, "path": p} for a, p, _ in partition
                    ],
                    "output_artifacts": [a.pk for a, _, _ in partition],
                    "shape": {
                        "sequences": len(pending_sequences),
                        "max_length": max(map(len, pending_sequences.values()), default=0),
                    },
                    **(extra or {}),
                },
            },
        )
        changed = Artifact.objects.filter(
            pk__in=[a.pk for a, _, _ in partition], producer__isnull=True
        ).update(producer=stage)
        if changed != len(partition):
            raise IntegrityError("Embedding producer changed during planning")
        for artifact, _, _ in partition:
            artifact.producer = stage
    return needed


@graph_transaction
def kinform_graph(execution, method, target, seq_map):
    from api.execution.representation_aliases import alternate_identity

    roots = {
        "kinform_t5_residues": ["prot_t5_layer_19", "prot_t5_last"],
        "kinform_esm2_residues": ["esm2_layer_26", "esm2_layer_29"],
        "kinform_esmc_residues": ["esmc_layer_24", "esmc_layer_32"],
    }
    binding_path = "pseq2sites/binding_sites_all.tsv"
    ids = sorted(seq_map)
    paths = {
        sid: {
            f"sequence_info/{root}/{kind}_vecs/{sid}.npy"
            for names in roots.values()
            for root in names
            for kind in ["mean", "weighted"]
        }
        for sid in seq_map
    }
    wanted = [
        (artifact_identity(execution.specification, method, target, path), path, sid)
        for sid, names in paths.items()
        for path in names
    ]
    binding_identity = artifact_identity(
        execution.specification, method, target, binding_path, context={"sequences": ids}
    )
    wanted.append((binding_identity, binding_path, ""))
    alternatives = {digest(identity): alternate_identity(identity) for identity, _, _ in wanted}
    all_ids = set(alternatives) | {
        digest(other) for other in alternatives.values() if other is not None
    }
    ready = ArtifactLocation.objects.filter(artifact_id=OuterRef("pk"), ready=True)
    known = {
        a.pk: a
        for a in Artifact.objects.filter(pk__in=all_ids).annotate(catalog_ready=Exists(ready))
    }
    cached = []
    for identity, path, sid in wanted:
        primary = digest(identity)
        artifact = known.get(primary)
        if artifact is None and alternatives[primary] is not None:
            artifact = known.get(digest(alternatives[primary]))
        if artifact is None or not artifact.catalog_ready or not artifact.checksum:
            break
        cached.append((artifact, path, sid))
    if len(cached) == len(wanted):
        return cached

    # Resolve pooled outputs first. An existing producer already owns all of its
    # dependencies; rebuilding expired residue inputs would create unused work.
    outputs = embedding_stage(execution, method, target, seq_map, paths, {"step": "kinform_pool"})
    pools = list(
        WorkStage.objects.filter(
            pk__in={a.producer_id for a, _, _ in outputs if a.producer_id},
            execution=execution,
            kind="embedding",
            method=method,
            target=target,
            state="ready",
            payload__step="kinform_pool",
            attempts__isnull=True,
        )
    )
    pool_needs = {}
    for producer in pools:
        for output in producer.payload["outputs"]:
            path = Path(output["path"])
            pool_needs.setdefault(path.stem, set()).add(path.parts[1])
    binding = declare(binding_identity)
    build_binding = not binding.producer_id and not binding.locations.filter(ready=True).exists()
    pieces, pending = [], []
    if build_binding:
        for offset in range(0, len(ids), EMBEDDING_STAGE_SEQUENCES):
            part = {sid: seq_map[sid] for sid in ids[offset : offset + EMBEDDING_STAGE_SEQUENCES]}
            context = {"sequences": sorted(part)}
            path = "pseq2sites/parts/" + digest(context) + ".tsv"
            artifact = declare(
                artifact_identity(execution.specification, method, target, path, context=context)
            )
            if not artifact.producer_id and not artifact.locations.filter(ready=True).exists():
                stage, _ = WorkStage.objects.get_or_create(
                    execution=execution,
                    key="binding:" + artifact.pk,
                    defaults={
                        "kind": "embedding",
                        "method": method,
                        "target": target,
                        "payload": {
                            "step": "kinform_pseq2sites",
                            "seq_id_to_seq": part,
                            "shape": {
                                "sequences": len(part),
                                "max_length": max(map(len, part.values()), default=0),
                            },
                            "outputs": [
                                {"id": artifact.pk, "identity": artifact.identity, "path": path}
                            ],
                            "output_artifacts": [artifact.pk],
                        },
                    },
                )
                artifact.producer = stage
                artifact.save(update_fields=["producer"])
                pending.append(stage)
            pieces.append((artifact, path))
    binding_ids = {sid for stage in pending for sid in stage.payload["seq_id_to_seq"]}
    residues = []
    for step, names in roots.items():
        members = {sid for sid, needed in pool_needs.items() if needed.intersection(names)}
        if step == "kinform_t5_residues":
            members |= binding_ids
        residue_paths = {
            sid: {
                f"sequence_info/{root}/residue_vecs/{sid}.npy"
                for root in names
                if root in pool_needs.get(sid, set())
                or (root == "prot_t5_last" and sid in binding_ids)
            }
            for sid in members
        }
        if residue_paths:
            residues.extend(
                embedding_stage(
                    execution,
                    method,
                    target,
                    {sid: seq_map[sid] for sid in members},
                    residue_paths,
                    {"step": step},
                )
            )
    for stage in pending:
        attach_many(
            stage,
            [
                (artifact, path)
                for artifact, path, sid in residues
                if sid in stage.payload["seq_id_to_seq"] and "/prot_t5_last/" in path
            ],
        )
    if build_binding:
        stage, _ = WorkStage.objects.get_or_create(
            execution=execution,
            key="binding-merge:" + binding.pk,
            defaults={
                "kind": "predict",
                "method": "KinForm-binding",
                "target": target,
                "payload": {
                    "step": "kinform_binding_merge",
                    "seq_id_to_seq": seq_map,
                    "parts": [path for _, path in pieces],
                    "shape": {
                        "sequences": len(seq_map),
                        "max_length": max(map(len, seq_map.values()), default=0),
                    },
                    "outputs": [
                        {"id": binding.pk, "identity": binding.identity, "path": binding_path}
                    ],
                    "output_artifacts": [binding.pk],
                },
            },
        )
        attach_many(stage, pieces)
        binding.producer = stage
        binding.save(update_fields=["producer"])
    for producer in pools:
        attach_many(
            producer,
            [
                (artifact, path)
                for artifact, path, sid in residues
                if sid in producer.payload["seq_id_to_seq"]
            ]
            + [(binding, binding_path)],
        )
        producer.kind, producer.method = "predict", "KinForm-pool"
        producer.save(update_fields=["kind", "method"])
    return [(binding, binding_path, "")] + outputs


@graph_transaction
def molecular_graph(execution, descriptor, target, batch, ordinal):
    method = MOLECULAR[descriptor.key]
    rows = []
    substrates = batch.call_kwargs.get("substrates", [])
    for index, (sequence, substrate) in enumerate(zip(batch.sequences, substrates, strict=False)):
        if (
            not all(c in set("ACDEFGHIKLMNPQRSTVWY") for c in sequence)
            or convert_to_mol(substrate) is None
        ):
            continue
        text = substrate_as_smiles(
            substrate,
            canonicalize=execution.specification["canonicalize_substrates"],
            preserve_raw_smiles_when_possible=True,
        )
        position = (ordinal + len(rows)) % 128 % 100
        context = (
            {"input": text, "legacy_position": position}
            if method in {"KinForm", "UniKP"}
            else {"input": text}
        )
        path = f"sequence_info/molecular/{method}/{digest(context)}.npy"
        rows.append(
            {
                "input": text,
                "legacy_position": position,
                "path": path,
                "context": context,
                "unit_index": index,
            }
        )
    outputs = []
    for row in rows:
        artifact = declare(
            artifact_identity(
                execution.specification, descriptor.key, target, row["path"], context=row["context"]
            )
        )
        outputs.append((artifact, row["path"], ""))
    missing = [
        (artifact, row)
        for (artifact, _, _), row in zip(outputs, rows, strict=False)
        if not artifact.producer_id and not artifact.locations.filter(ready=True).exists()
    ]
    unique_rows = list({artifact.pk: (artifact, row) for artifact, row in missing}.values())
    for offset in range(0, len(unique_rows), EMBEDDING_STAGE_SEQUENCES):
        unique = {
            artifact.pk: (artifact, row)
            for artifact, row in unique_rows[offset : offset + EMBEDDING_STAGE_SEQUENCES]
        }
        key = "molecular:" + digest(sorted(unique))
        stage, _ = WorkStage.objects.get_or_create(
            execution=execution,
            key=key,
            defaults={
                "kind": "embedding",
                "method": "molecular-" + method,
                "target": target,
                "payload": {
                    "step": "molecular",
                    "molecular_method": method,
                    "rows": [row for _, row in unique.values()],
                    "shape": {
                        "sequences": len(unique),
                        "max_length": max(
                            (len(row["input"]) for _, row in unique.values()), default=0
                        ),
                    },
                    "outputs": [
                        {"id": a.pk, "identity": a.identity, "path": row["path"]}
                        for a, row in unique.values()
                    ],
                    "output_artifacts": list(unique),
                },
            },
        )
        for artifact, _ in unique.values():
            artifact.producer = stage
            artifact.save(update_fields=["producer"])
    return outputs, rows


def prepare(execution, media: Path, blob_root: Path, *, repair_validation=False):
    """Resume by deterministic keys. Each chunk is committed independently."""
    spec = execution.specification
    source = media / "jobs" / execution.job.public_id / "input.csv"
    if sha256_file(source) != execution.input_checksum:
        raise ValueError("Original input checksum mismatch")
    descriptors = {t: get_method(spec["methods"][t]) for t in spec["targets"]}
    store = ArtifactStore(blob_root)
    store.root.mkdir(parents=True, exist_ok=True)
    prep = execution.stages.get(key="prepare")
    ordinal = {t: 0 for t in spec["targets"]}
    prediction_stages = []
    original_chunks = {}
    offset = 0
    chunk_dir = blob_root.parent / "planning" / str(execution.pk)
    from api.execution.bounded import frames

    for chunk_index, frame in enumerate(frames(source)):
        from api.execution.background import higher_preparation

        if not repair_validation and higher_preparation(execution).exists():
            # Completed chunks keep their deterministic keys. Resume this
            # preparation after the urgent request has been prepared.
            return
        frame = frame.reset_index(drop=True)
        plan = build_sequence_batch_plan(frame, descriptors.values(), spec["handle_long_sequences"])
        mapping = {}
        for sequence in plan.expansion.children:
            text = sequence.processed_sequence
            if isinstance(text, str) and text:
                mapping[text] = sequence_id(text)
        chunk_dir.mkdir(parents=True, exist_ok=True)
        chunk_path = chunk_dir / f"{chunk_index}.csv"
        frame.to_csv(chunk_path, index=False)
        with transaction.atomic():
            chunk = _publish_after_cleanup(
                chunk_path,
                {"job_input": execution.input_checksum, "chunk": chunk_index},
                "production",
                store,
            )
            original_chunks[chunk_index] = chunk
            for target, descriptor in descriptors.items():
                batch = build_target_batch_plan(descriptor, target, frame, plan)
                sequences = normalise_sequences_for_method(
                    descriptor.key, embedding_sequences(descriptor, batch)
                )
                for sequence in sequences:
                    mapping.setdefault(sequence, sequence_id(sequence))
                seq_map = {mapping[seq]: seq for seq in sequences}
                prediction, _ = WorkStage.objects.get_or_create(
                    execution=execution,
                    key=f"predict:{target}:{chunk_index}",
                    defaults={
                        "kind": "predict",
                        "method": descriptor.key,
                        "target": target,
                        "payload": {
                            "offset": offset,
                            "rows": len(frame),
                            "seqmap": mapping,
                            "legacy_ordinal": ordinal[target],
                            "shape": prediction_shape(
                                frame, batch, chunk_path.stat().st_size, descriptor.key
                            ),
                            "experimental_results": spec.get("experimental_results", {}).get(
                                target, []
                            ),
                            "chunk": chunk_index,
                        },
                    },
                )
                runner_frame, invalid_cells = validated_prediction_frame(frame, descriptor)
                runner_chunk = chunk
                if invalid_cells:
                    runner_path = chunk_dir / f"{chunk_index}.single-molecule.csv"
                    runner_frame.to_csv(runner_path, index=False)
                    runner_chunk = publish_file(
                        runner_path,
                        {
                            "job_input": execution.input_checksum,
                            "chunk": chunk_index,
                            "validation": "single-molecule-v1",
                        },
                        "production",
                        store,
                    )
                    prediction.payload["input_validation"] = {
                        "policy": "single-molecule-v1",
                        "invalid_cells": invalid_cells,
                        "original_artifact": chunk.pk,
                    }
                attach_prediction_input(
                    prediction,
                    runner_chunk,
                    f"jobs/{execution.job.public_id}/input.csv",
                    repair=repair_validation,
                )
                prediction.dependencies.add(prep)
                if descriptor.key in PROTEIN_STEPS:
                    expected = expected_paths_by_seq(
                        method_key=descriptor.key,
                        target=target,
                        seq_ids=list(seq_map),
                        media_path=Path("/app/media"),
                        env=spec.get("embedding_env", {}),
                    )
                    relative = {
                        sid: {str(Path(path).relative_to("/app/media")) for path in paths}
                        for sid, paths in expected.items()
                    }
                    if seq_map and not relative:
                        raise ValueError("Embedding profile could not declare required artifacts")
                    for artifact, path, _ in embedding_stage(
                        execution,
                        descriptor.key,
                        target,
                        seq_map,
                        relative,
                        {"step": PROTEIN_STEPS[descriptor.key]},
                    ):
                        attach(prediction, artifact, path)
                if descriptor.key.startswith("KinForm") and seq_map:
                    for artifact, path, _ in kinform_graph(
                        execution, descriptor.key, target, seq_map
                    ):
                        attach(prediction, artifact, path)
                if descriptor.key in MOLECULAR:
                    molecular, rows = molecular_graph(
                        execution, descriptor, target, batch, ordinal[target]
                    )
                    for artifact, path, _ in molecular:
                        attach(prediction, artifact, path)
                    prediction.payload["molecular"] = {MOLECULAR[descriptor.key]: rows}
                    prediction.payload["unit_context"] = {
                        str(row["unit_index"]): row["context"] for row in rows
                    }
                    ordinal[target] += len(rows)
                else:
                    ordinal[target] += len(batch.sequences)
                if spec.get("recon_xkg"):
                    from api.execution.context import StageContext, stage_context
                    from api.services.prediction_store import (
                        build_unit_keys,
                        coerce_value,
                        get_many,
                    )

                    namespace = digest(spec["model_versions"][descriptor.key])
                    prediction.payload["cache_namespace"] = namespace
                    with stage_context(
                        StageContext(
                            execution.job.public_id,
                            cache_namespace=namespace,
                            unit_context=prediction.payload.get("unit_context", {}),
                        )
                    ):
                        keys, _, _ = build_unit_keys(
                            descriptor,
                            target,
                            batch.sequences,
                            batch.call_kwargs,
                            spec["canonicalize_substrates"],
                        )
                    prediction.payload["cache_snapshot"] = {
                        key: value
                        for key, value in get_many(keys).items()
                        if coerce_value(value) is not None
                    }
                result_path = f"execution/results/{prediction.pk}.json"
                result = declare(
                    {
                        "job": execution.job.public_id,
                        "stage": str(prediction.pk),
                        "specification": execution.checksum,
                    }
                )
                result.producer = prediction
                result.save(update_fields=["producer"])
                prediction.payload.update(
                    outputs=[{"id": result.pk, "identity": result.identity, "path": result_path}],
                    output_artifacts=[result.pk],
                )
                prediction.save(update_fields=["payload"])
                prediction_stages.append(prediction)
        offset += len(frame)
        # Committed chunks are complete; make them claimable while later chunks
        # are planned. Binding otherwise waited for the whole request, and the
        # controller does nothing else while a large request is prepared.
        bind_profiles()
        _yield_to_cleanup()
    with transaction.atomic():
        assembly, _ = WorkStage.objects.get_or_create(
            execution=execution,
            key="assemble",
            defaults={"kind": "assemble", "payload": {"rows": offset, "shape": {"rows": offset}}},
        )
        assembly.dependencies.add(*prediction_stages)
        for prediction in prediction_stages:
            output = prediction.payload["outputs"][0]
            attach(assembly, Artifact.objects.get(pk=output["id"]), output["path"])
            # Inputs are needed for final original-row serialization, without loading whole jobs.
            attach(
                assembly,
                original_chunks[prediction.payload["chunk"]],
                f"execution/inputs/{prediction.payload['chunk']}.csv",
            )
        identity = {
            "job": execution.job.public_id,
            "specification": execution.checksum,
            "result": "assembled",
        }
        result = declare(identity)
        if not (spec.get("include_similarity_columns", True) and "kcat" in spec["targets"]):
            result.regenerable = False
            result.save(update_fields=["regenerable"])
        result.producer = assembly
        result.save(update_fields=["producer"])
        assembly.payload.update(
            outputs=[
                {
                    "id": result.pk,
                    "identity": identity,
                    "path": f"jobs/{execution.job.public_id}/output.csv",
                }
            ],
            output_artifacts=[result.pk],
            phase="preliminary"
            if spec.get("include_similarity_columns", True) and "kcat" in spec["targets"]
            else "final",
        )
        assembly.save(update_fields=["payload"])
        if assembly.payload["phase"] == "preliminary":
            enrich_identity = {
                "job": execution.job.public_id,
                "specification": execution.checksum,
                "result": "similarity",
            }
            enriched = declare(enrich_identity)
            path = f"jobs/{execution.job.public_id}/output.csv"
            similarity, _ = WorkStage.objects.get_or_create(
                execution=execution,
                key="similarity",
                defaults={
                    "kind": "similarity",
                    "method": spec["methods"]["kcat"],
                    "target": "kcat",
                    "payload": {
                        "rows": offset,
                        "shape": {"rows": offset},
                        "outputs": [{"id": enriched.pk, "identity": enrich_identity, "path": path}],
                        "output_artifacts": [enriched.pk],
                    },
                },
            )
            enriched.producer = similarity
            enriched.save(update_fields=["producer"])
            attach(similarity, result, path)
            result_chunks = []
            for prediction in prediction_stages:
                if prediction.target == "kcat":
                    output = prediction.payload["outputs"][0]
                    attach(similarity, Artifact.objects.get(pk=output["id"]), output["path"])
                    result_chunks.append(
                        {"path": output["path"], "offset": prediction.payload["offset"]}
                    )
            similarity.payload["result_chunks"] = sorted(
                result_chunks, key=lambda entry: entry["offset"]
            )
            similarity.save(update_fields=["payload"])
            identity = {
                "job": execution.job.public_id,
                "specification": execution.checksum,
                "result": "final",
            }
            final_artifact = declare(identity)
            final_artifact.regenerable = False
            final_artifact.save(update_fields=["regenerable"])
            final, _ = WorkStage.objects.get_or_create(
                execution=execution,
                key="publish",
                defaults={
                    "kind": "assemble",
                    "payload": {
                        "phase": "publish",
                        "rows": offset,
                        "shape": {"rows": offset},
                        "outputs": [{"id": final_artifact.pk, "identity": identity, "path": path}],
                        "output_artifacts": [final_artifact.pk],
                    },
                },
            )
            final_artifact.producer = final
            final_artifact.save(update_fields=["producer"])
            attach(final, enriched, path)
        prep.state = "completed"
        prep.completed_at = timezone.now()
        prep.save(update_fields=["state", "completed_at"])
    # Published chunks are immutable blobs; these are only private staging copies.
    # Failed/interrupted planning retains its directory until a successful retry.
    shutil.rmtree(chunk_dir, ignore_errors=True)
