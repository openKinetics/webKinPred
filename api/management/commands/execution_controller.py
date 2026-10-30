import time
from datetime import timedelta
from pathlib import Path

from api.execution.control import bind_profiles, ensure_hosts, require_postgres
from api.execution.models import ExecutionOutbox, ExecutionSpec
from api.execution.planning import prepare
from django.conf import settings
from django.core.management.base import BaseCommand
from django.db import connection
from django.db.models.functions import Coalesce
from django.utils import timezone


class Command(BaseCommand):
    help = "Reconcile ready work from PostgreSQL independently of broker notifications."

    def add_arguments(self, parser):
        parser.add_argument("--once", action="store_true")

    def handle(self, *args, **options):
        require_postgres()
        ensure_hosts()
        # Session-level singleton; no database transaction is held while preparing files.
        with connection.cursor() as cursor:
            cursor.execute("SELECT pg_try_advisory_lock(%s)", [481726391])
            if not cursor.fetchone()[0]:
                self.stderr.write("Another execution controller owns the preparation lock")
                return
        try:
            last_priority_refresh = 0
            while True:
                if settings.EXECUTION_V2_ENABLED:
                    if time.monotonic() - last_priority_refresh >= 5:
                        from api.execution.priority import refresh_stage_priorities

                        refresh_stage_priorities()
                        last_priority_refresh = time.monotonic()
                    for execution in (
                        ExecutionSpec.objects.filter(
                            paused=False,
                            stages__key="prepare",
                            stages__state="ready",
                            stages__available_at__lte=timezone.now(),
                        )
                        .select_related("job")
                        .distinct()
                        .order_by(
                            "-priority", Coalesce("job__submission_time", "created_at"), "pk"
                        )[:4]
                    ):
                        # A priority request arriving during this batch should
                        # not wait behind the other prefetched normal requests.
                        if ExecutionSpec.objects.filter(
                            paused=False,
                            priority__gt=execution.priority,
                            stages__key="prepare",
                            stages__state="ready",
                            stages__available_at__lte=timezone.now(),
                        ).exists():
                            break
                        try:
                            prepare(
                                execution,
                                Path(settings.MEDIA_ROOT),
                                Path(settings.EXECUTION_BLOB_ROOT),
                            )
                            execution.refresh_from_db(fields=["priority"])
                            if execution.priority:
                                from api.execution.priority import refresh_stage_priorities

                                refresh_stage_priorities()
                                bind_profiles()
                        except Exception as error:
                            # Keep inputs/specifications and hold for controller recovery.
                            from api.execution.errors import classify

                            category = classify(error)
                            execution.stages.filter(key="prepare").update(
                                state="ready"
                                if category in {"oom", "disk", "network", "cpu", "storage"}
                                else "held",
                                waiting_reason=category
                                if category != "unknown"
                                else "preparation_requires_attention",
                                available_at=timezone.now() + timedelta(seconds=60),
                            )
                            import logging

                            logging.getLogger(__name__).exception(
                                "Preparation held",
                                extra={"job_public_id": execution.job.public_id},
                            )
                    bind_profiles()
                    from api.services.prediction_store import upsert_many, upsert_similarity_many

                    for event in ExecutionOutbox.objects.filter(published_at__isnull=True).order_by(
                        "id"
                    )[:100]:
                        if event.event.get("recover_stage"):
                            from api.execution.recovery import missing_dependencies

                            missing_dependencies(
                                event.event["recover_stage"], event.event["host_id"]
                            )
                        rows = event.event.get("cache_writes", [])
                        if rows and upsert_many(rows) != len({row["lookup_key"] for row in rows}):
                            continue  # Durable retry after cache/database outages.
                        from collections import defaultdict

                        groups = defaultdict(list)
                        for dataset, *values in event.event.get("similarity_writes", []):
                            groups[dataset].append(tuple(values))
                        if any(
                            upsert_similarity_many(values, dataset) != len(values)
                            for dataset, values in groups.items()
                        ):
                            continue
                        event.published_at = timezone.now()
                        event.save(update_fields=["published_at"])
                if options["once"]:
                    return
                time.sleep(1)
        finally:
            with connection.cursor() as cursor:
                cursor.execute("SELECT pg_advisory_unlock(%s)", [481726391])
