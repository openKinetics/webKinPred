"""Classify only managed executor artifacts; never traverses legacy cache trees."""

import json
from collections import Counter, defaultdict

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
        pending = defaultdict(list)

        def flush():
            # A constant policy per group avoids quadratic CASE evaluation for
            # hundreds of thousands of verified vectors. Updates are resumable.
            for policy, ids in pending.items():
                Artifact.objects.filter(pk__in=ids, retention="protected").update(retention=policy)
            pending.clear()

        with connection.cursor() as cursor:
            cursor.execute("SET jit = off")
            cursor.execute("SET max_parallel_workers_per_gather = 0")
        try:
            last_location = 0
            seen_protected = set()
            while True:
                page = list(
                    ArtifactLocation.objects.filter(ready=True, pk__gt=last_location)
                    .order_by("pk")
                    .values_list("pk", "artifact_id")[:512]
                )
                if not page:
                    break
                last_location = page[-1][0]
                # Keyset pages avoid PostgreSQL materializing a WITH HOLD cursor
                # containing the entire resident catalogue on the production disk.
                rows = (
                    Artifact.objects.filter(
                        retention="protected", pk__in=[ident for _, ident in page]
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
                )
                for artifact in rows:
                    if artifact.pk in seen_protected:
                        continue
                    policy = retention_for(artifact.identity)
                    if not artifact.regenerable:
                        policy = "permanent"
                    if policy == "transient" and (
                        not artifact.producer_id or artifact.producer.execution.job_id is None
                    ):
                        policy = "protected"
                    counts[policy] += 1
                    if options["apply"] and policy != "protected":
                        pending[policy].append(artifact.pk)
                        if sum(map(len, pending.values())) >= 256:
                            flush()
                    else:
                        seen_protected.add(artifact.pk)
                if pending:
                    flush()
            if pending:
                flush()
        finally:
            with connection.cursor() as cursor:
                cursor.execute("RESET jit")
                cursor.execute("RESET max_parallel_workers_per_gather")
        self.stdout.write(json.dumps({"applied": options["apply"], "classified": counts}))
