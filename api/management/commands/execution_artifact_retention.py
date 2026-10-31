"""Classify only managed executor artifacts; never traverses legacy cache trees."""

import json
from collections import Counter

from api.execution.artifact_policy import retention_for
from api.execution.models import Artifact, ArtifactLocation
from django.core.management.base import BaseCommand
from django.db import connection


class Command(BaseCommand):
    help = "Classify verified executor artifacts for retention; --apply enables updates."

    def add_arguments(self, parser):
        parser.add_argument("--apply", action="store_true")

    def handle(self, *args, **options):
        counts = Counter()
        pending = []
        with connection.cursor() as cursor:
            cursor.execute("SET jit = off")
            cursor.execute("SET max_parallel_workers_per_gather = 0")
        try:
            rows = (
                Artifact.objects.filter(
                    retention="protected",
                    pk__in=ArtifactLocation.objects.filter(ready=True).values("artifact_id"),
                )
                .select_related("producer__execution")
                .only(
                    "id",
                    "identity",
                    "retention",
                    "regenerable",
                    "producer_id",
                    "producer__execution_id",
                    "producer__execution__job_id",
                )
                .iterator(chunk_size=256)
            )
            for artifact in rows:
                policy = retention_for(artifact.identity)
                if not artifact.regenerable:
                    policy = "permanent"
                if policy == "transient" and (
                    not artifact.producer_id or artifact.producer.execution.job_id is None
                ):
                    policy = "protected"
                counts[policy] += 1
                if options["apply"] and policy != "protected":
                    artifact.retention = policy
                    pending.append(artifact)
                    if len(pending) >= 256:
                        Artifact.objects.bulk_update(pending, ["retention"], batch_size=256)
                        pending.clear()
            if pending:
                Artifact.objects.bulk_update(pending, ["retention"], batch_size=256)
        finally:
            with connection.cursor() as cursor:
                cursor.execute("RESET jit")
                cursor.execute("RESET max_parallel_workers_per_gather")
        self.stdout.write(json.dumps({"applied": options["apply"], "classified": counts}))
