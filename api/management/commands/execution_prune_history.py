"""Delete bookkeeping that finished requests no longer use; dry run unless --apply.

For requests completed at least --older-than-hours ago, without a pending progress
projection, this removes:

* the input links (ArtifactDependency rows) of their finished prediction, similarity and
  assembly stages. Embedding stages and the KinForm pool and binding steps keep theirs:
  recovery regenerates lost outputs from those producers' inputs and payloads.
* catalogue rows of residue matrices (transient artifacts) that have no ready copy, no
  remaining input link and a finished producer of such a request (or of a shared batch),
  with their location rows. A later request needing the same matrix plans it normally.

Job records and results, stage and attempt records, stage payloads, pooled vectors and
anything an unfinished request uses are kept. Deleted rows leave reusable space in the
tables; the database files do not shrink.
"""

import json
import time
from datetime import timedelta

from api.execution.models import (
    Artifact,
    ArtifactDependency,
    ArtifactLocation,
    ExecutionOutbox,
    ExecutionSpec,
    WorkStage,
)
from django.core.management.base import BaseCommand
from django.db import OperationalError, connection, transaction
from django.db.models import Q
from django.utils import timezone

FINISHED = ["completed", "canceled"]
# recovery._recoverable(): these producers are regenerated from their own inputs.
RECOVERABLE_METHODS = ["KinForm-pool", "KinForm-binding"]
GRAPH_LOCK = 481726392


class Command(BaseCommand):
    help = "Delete finished requests' unused input links and deleted residue catalogue rows."

    def add_arguments(self, parser):
        parser.add_argument("--apply", action="store_true")
        parser.add_argument("--older-than-hours", type=float, default=24)
        parser.add_argument("--batch", type=int, default=5000)
        parser.add_argument("--pause", type=float, default=0.2)
        parser.add_argument("--max-seconds", type=float, default=0)

    def handle(self, *args, **options):
        self.apply, self.batch, self.pause = options["apply"], options["batch"], options["pause"]
        self.deadline = (
            time.monotonic() + options["max_seconds"] if options["max_seconds"] else None
        )
        self.cutoff = timezone.now() - timedelta(hours=options["older_than_hours"])
        with connection.cursor() as cursor:
            cursor.execute("SET jit = off")
            cursor.execute("SET max_parallel_workers_per_gather = 0")
        summary = {"apply": self.apply, "cutoff": self.cutoff.isoformat()}
        summary["input_links"] = self.prune_links()
        summary.update(self.prune_residues())
        self.stdout.write(json.dumps(summary))

    def expired(self):
        return self.deadline is not None and time.monotonic() >= self.deadline

    def finished_requests(self):
        pending = set()
        for owners in ExecutionOutbox.objects.filter(
            published_at__isnull=True, event__has_key="progress_execution_ids"
        ).values_list("event__progress_execution_ids", flat=True):
            pending.update(str(owner) for owner in owners or [])
        return [
            pk
            for pk in ExecutionSpec.objects.filter(
                job__status="Completed", job__completion_time__lt=self.cutoff
            )
            .order_by("pk")
            .values_list("pk", flat=True)
            if str(pk) not in pending
        ]

    def prune_links(self):
        executions, deleted = self.finished_requests(), 0
        for offset in range(0, len(executions), 100):
            stages = list(
                WorkStage.objects.filter(
                    execution_id__in=executions[offset : offset + 100], state__in=FINISHED
                )
                .exclude(kind="embedding")
                .exclude(method__in=RECOVERABLE_METHODS)
                .values_list("pk", flat=True)
            )
            for start in range(0, len(stages), 500):
                links = ArtifactDependency.objects.filter(stage_id__in=stages[start : start + 500])
                if not self.apply:
                    deleted += links.count()
                    continue
                while True:
                    if self.expired():
                        return deleted
                    ids = list(links.values_list("pk", flat=True)[: self.batch])
                    if not ids:
                        break
                    deleted += len(ids)
                    with transaction.atomic():
                        with connection.cursor() as cursor:
                            cursor.execute("SET LOCAL statement_timeout = 30000")
                        ArtifactDependency.objects.filter(pk__in=ids).delete()
                    time.sleep(self.pause)
        return deleted

    def eligible_residues(self, page):
        ids = [pk for pk, _ in page]
        ready = set(
            ArtifactLocation.objects.filter(artifact_id__in=ids, ready=True).values_list(
                "artifact_id", flat=True
            )
        )
        linked = set(
            ArtifactDependency.objects.filter(artifact_id__in=ids).values_list(
                "artifact_id", flat=True
            )
        )
        finished = set(
            WorkStage.objects.filter(
                pk__in={producer for _, producer in page if producer}, state__in=FINISHED
            )
            .filter(
                Q(execution__job__isnull=True)
                | Q(
                    execution__job__status="Completed",
                    execution__job__completion_time__lt=self.cutoff,
                )
            )
            .values_list("pk", flat=True)
        )
        return [
            pk
            for pk, producer in page
            if pk not in ready and pk not in linked and producer in finished
        ]

    def prune_residues(self):
        last, artifacts, locations = "", 0, 0
        while not self.expired():
            page = list(
                Artifact.objects.filter(retention="transient", pk__gt=last)
                .order_by("pk")
                .values_list("pk", "producer_id")[: self.batch]
            )
            if not page:
                break
            last = page[-1][0]
            eligible = self.eligible_residues(page)
            if not eligible:
                continue
            if not self.apply:
                artifacts += len(eligible)
                locations += ArtifactLocation.objects.filter(artifact_id__in=eligible).count()
                continue
            try:
                with transaction.atomic():
                    with connection.cursor() as cursor:
                        # Planning attaches inputs to existing artifacts under this
                        # lock; holding it means no new link can appear meanwhile.
                        cursor.execute("SET LOCAL lock_timeout = 5000")
                        cursor.execute("SELECT pg_advisory_xact_lock(%s)", [GRAPH_LOCK])
                        cursor.execute("SET LOCAL lock_timeout = 0")
                        cursor.execute("SET LOCAL statement_timeout = 30000")
                    eligible = self.eligible_residues(
                        [row for row in page if row[0] in set(eligible)]
                    )
                    locations += ArtifactLocation.objects.filter(artifact_id__in=eligible).delete()[
                        0
                    ]
                    artifacts += Artifact.objects.filter(pk__in=eligible).delete()[0]
            except OperationalError:
                last = page[0][0][:-1]  # Retry this page after the lock is free.
            time.sleep(self.pause)
        return {"residue_artifacts": artifacts, "residue_locations": locations}
