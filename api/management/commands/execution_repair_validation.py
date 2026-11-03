"""Repair stopped single-molecule validation mismatches without resubmitting jobs."""

import json
from pathlib import Path

from api.execution.artifacts import sha256_file
from api.execution.bounded import frames
from api.execution.models import ExecutionAttempt, ExecutionSpec
from api.execution.planning import prepare, validated_prediction_frame
from api.execution.resources import LIVE_STATES
from api.methods.registry import get
from django.conf import settings
from django.core.management.base import BaseCommand, CommandError
from django.db import transaction
from django.utils import timezone


class Command(BaseCommand):
    help = "Repair selected inactive jobs held by a single-molecule validation mismatch."

    def add_arguments(self, parser):
        parser.add_argument("jobs", nargs="+")
        parser.add_argument("--dry-run", action="store_true")

    def handle(self, *args, **options):
        for ident in dict.fromkeys(options["jobs"]):
            with transaction.atomic():
                execution = ExecutionSpec.objects.select_for_update().get(job__public_id=ident)
                held = list(execution.stages.filter(kind="predict", state="held"))
                if not held:
                    self.stdout.write(json.dumps({"job": ident, "repaired_stages": 0}))
                    continue
                if any(s.waiting_reason != "dependency_plan_requires_attention" for s in held):
                    raise CommandError("Job has an unrelated hold; review it before repair")
                if ExecutionAttempt.objects.filter(
                    stage__execution=execution, state__in=LIVE_STATES
                ).exists():
                    raise CommandError("Job has live attempts; let them finish before repair")
                source = Path(settings.MEDIA_ROOT) / "jobs" / ident / "input.csv"
                if sha256_file(source) != execution.input_checksum:
                    raise CommandError("Original input checksum mismatch")
                counts = {s.method: 0 for s in held}
                for frame in frames(source):
                    for method in counts:
                        _, invalid = validated_prediction_frame(frame, get(method))
                        counts[method] += len(invalid)
                if not all(counts.values()):
                    raise CommandError(
                        "Hold is not explained by unsupported single-molecule inputs"
                    )
                if options["dry_run"]:
                    self.stdout.write(
                        json.dumps({"job": ident, "dry_run": True, "invalid_cells": counts})
                    )
                    continue
                paused = execution.paused
                execution.paused = True
                execution.save(update_fields=["paused"])
                prepare(
                    execution,
                    Path(settings.MEDIA_ROOT),
                    Path(settings.EXECUTION_BLOB_ROOT),
                    repair_validation=True,
                )
                for stage in held:
                    stage.refresh_from_db()
                    if (
                        stage.payload.get("input_validation", {}).get("policy")
                        != "single-molecule-v1"
                    ):
                        raise CommandError("Repair did not produce the required validated input")
                    stage.state, stage.waiting_reason, stage.available_at = (
                        "ready",
                        "",
                        timezone.now(),
                    )
                    stage.save(update_fields=["state", "waiting_reason", "available_at"])
                execution.paused = paused
                execution.save(update_fields=["paused"])
                self.stdout.write(
                    json.dumps(
                        {"job": ident, "repaired_stages": len(held), "invalid_cells": counts}
                    )
                )
