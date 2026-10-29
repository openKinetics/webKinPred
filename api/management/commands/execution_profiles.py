import json
import re
from pathlib import Path

from api.execution.control import digest
from api.execution.models import ResourceProfile
from api.execution.validation import parity_accepted
from django.core.management.base import BaseCommand, CommandError
from django.db import transaction


class Command(BaseCommand):
    help = "Import profiles only with complete measurements, immutable images, and parity evidence."

    def add_arguments(self, parser):
        parser.add_argument("registry", type=Path)

    @transaction.atomic
    def handle(self, *args, **options):
        for profile in json.loads(options["registry"].read_text()):
            evidence = profile.get("evidence", {})
            runtime = evidence.get("runtime", {})
            required = {
                "source_commit",
                "source_digest",
                "model_checksums",
                "tokenizer_checksums",
                "dependencies",
                "hardware",
                "driver",
                "precision",
                "batch_context",
                "hosts",
            }
            if not required <= runtime.keys():
                raise CommandError("Profile runtime identity is incomplete")
            if (
                evidence.get("fresh_runs", 0) < 3
                or evidence.get("warm_runs", 0) < 5
                or not parity_accepted(evidence, gpu=profile["resources"].get("vram", 0) > 0)
                or not evidence.get("covered_limits")
                or not evidence.get("dataset_checksum")
                or not evidence.get("runtime")
                or not evidence.get("runtime", {}).get("hosts")
                or not profile.get("settings", {}).get("model_identity")
            ):
                raise CommandError(
                    "Profile lacks required profiling and scientific parity evidence"
                )
            if not re.fullmatch(r"(?:.+@)?sha256:[0-9a-f]{64}", profile["image"]):
                raise CommandError("Image must be immutable")
            if profile["stage_kind"] == "embedding" and profile["resources"].get("vram", 0) <= 0:
                raise CommandError("Embedding profiles require a positive GPU reservation")
            if profile["stage_kind"] in {"embedding", "predict", "similarity"}:
                targets = profile.get("settings", {}).get("targets")
                if (
                    not isinstance(targets, list)
                    or not targets
                    or any(target not in {"kcat", "Km", "kcat/Km"} for target in targets)
                ):
                    raise CommandError("Profiles must declare their measured target coverage")
            if any(profile["resources"].get(key, 0) <= 0 for key in ("ram", "cpu", "disk", "pids")):
                raise CommandError("Profiles require complete positive host resource envelopes")
            expected = digest(
                {
                    key: profile[key]
                    for key in [
                        "image",
                        "stage_kind",
                        "method",
                        "settings",
                        "resources",
                        "limits",
                        "evidence",
                    ]
                }
            )
            if expected != profile["fingerprint"]:
                raise CommandError("Profile fingerprint does not match evidence and configuration")
            existing = ResourceProfile.objects.filter(pk=profile["id"]).first()
            if existing and existing.fingerprint != expected:
                raise CommandError("Create a new profile ID for changed evidence")
            ResourceProfile.objects.update_or_create(
                pk=profile["id"],
                defaults={
                    **{
                        k: profile[k]
                        for k in [
                            "image",
                            "stage_kind",
                            "method",
                            "settings",
                            "resources",
                            "limits",
                            "evidence",
                        ]
                    },
                    "fingerprint": expected,
                    "validated": True,
                },
            )
