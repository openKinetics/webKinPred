"""Verify deferred requests without creating sequence, artifact, or quota rows."""

from pathlib import Path

from api.execution.artifacts import sha256_file
from api.execution.bounded import frames
from api.execution.models import ResourceProfile
from api.execution.planning import (
    EMBEDDING_STAGE_SEQUENCES,
    MOLECULAR,
    PROTEIN_STEPS,
    embedding_sequences,
)
from api.execution.shapes import prediction_shape
from api.methods.registry import get
from api.services.embedding_plan_service import normalise_sequences_for_method
from api.services.prediction_batch_service import build_sequence_batch_plan, build_target_batch_plan
from api.utils.convert_to_mol import convert_to_mol, substrate_as_smiles


class Coverage:
    def __init__(self, profiles=None):
        self.profiles = (
            list(profiles)
            if profiles is not None
            else list(
                ResourceProfile.objects.filter(validated=True, suspended=False).order_by("id")
            )
        )
        self.used = set()

    def require(self, kind, method, target, identity, shape, step=None):
        for profile in self.profiles:
            if (
                profile.stage_kind == kind
                and profile.method == method
                and profile.settings.get("step") == step
                and profile.settings.get("model_identity") == identity
                and target in profile.settings.get("targets", [target])
                and all(
                    shape.get(key, float("inf")) <= limit for key, limit in profile.limits.items()
                )
                and (kind != "assemble" or "production" in profile.evidence["runtime"]["hosts"])
            ):
                self.used.add(profile.fingerprint)
                return
        raise ValueError(f"Missing measured profile: {kind}/{method}/{target}/{step}: {shape}")

    def execution(self, execution, media):
        spec = execution.specification
        source = Path(media) / "jobs" / execution.job.public_id / "input.csv"
        if sha256_file(source) != execution.input_checksum:
            raise ValueError("Preserved input checksum changed")
        descriptors = {target: get(spec["methods"][target]) for target in spec["targets"]}
        count = 0
        for frame in frames(source):
            plan = build_sequence_batch_plan(
                frame, descriptors.values(), spec["handle_long_sequences"]
            )
            size = len(frame.to_csv(index=False).encode())
            for target, descriptor in descriptors.items():
                method = descriptor.key
                identity = spec["model_versions"][method]
                batch = build_target_batch_plan(descriptor, target, frame, plan)
                self.require(
                    "predict",
                    method,
                    target,
                    identity,
                    prediction_shape(frame, batch, size, method),
                )
                sequences = normalise_sequences_for_method(
                    method, embedding_sequences(descriptor, batch)
                )
                shape = {
                    "sequences": min(EMBEDDING_STAGE_SEQUENCES, len(sequences)),
                    "max_length": max(map(len, sequences), default=0),
                }
                if sequences and method in PROTEIN_STEPS:
                    self.require(
                        "embedding", method, target, identity, shape, PROTEIN_STEPS[method]
                    )
                if sequences and method.startswith("KinForm"):
                    for step in [
                        "kinform_t5_residues",
                        "kinform_esm2_residues",
                        "kinform_esmc_residues",
                        "kinform_pseq2sites",
                    ]:
                        self.require("embedding", method, target, identity, shape, step)
                    self.require("predict", "KinForm-pool", target, identity, shape, "kinform_pool")
                    self.require(
                        "predict",
                        "KinForm-binding",
                        target,
                        identity,
                        {**shape, "sequences": len(sequences)},
                        "kinform_binding_merge",
                    )
                if method in MOLECULAR:
                    molecules = []
                    for sequence, substrate in zip(
                        batch.sequences, batch.call_kwargs.get("substrates", []), strict=False
                    ):
                        if (
                            all(c in "ACDEFGHIKLMNPQRSTVWY" for c in sequence)
                            and convert_to_mol(substrate) is not None
                        ):
                            molecules.append(
                                substrate_as_smiles(
                                    substrate,
                                    canonicalize=spec["canonicalize_substrates"],
                                    preserve_raw_smiles_when_possible=True,
                                )
                            )
                    if molecules:
                        # This upper bound also covers positional contexts that
                        # prevent deduplication of UniKP/KinForm representations.
                        self.require(
                            "embedding",
                            "molecular-" + MOLECULAR[method],
                            target,
                            identity,
                            {
                                "sequences": min(EMBEDDING_STAGE_SEQUENCES, len(molecules)),
                                "max_length": max(map(len, molecules)),
                            },
                            "molecular",
                        )
            count += len(frame)
        if count == 0:
            raise ValueError("The preserved input has no rows")
        versions = spec.get("execution_versions", {})
        self.require(
            "assemble", "", "", versions.get("assemble", spec["model_versions"]), {"rows": count}
        )
        if spec.get("include_similarity_columns", True) and "kcat" in spec["targets"]:
            self.require(
                "similarity",
                spec["methods"]["kcat"],
                "kcat",
                versions.get("similarity", spec["model_versions"]),
                {"rows": count},
            )
        return {
            "specification": execution.checksum,
            "input": execution.input_checksum,
            "rows": count,
        }
