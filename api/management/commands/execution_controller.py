import time
from datetime import timedelta
from pathlib import Path

from api.execution.background import higher_preparation
from api.execution.control import bind_profiles, ensure_hosts, require_postgres
from api.execution.models import ExecutionOutbox, ExecutionSpec
from api.execution.planning import prepare
from django.conf import settings
from django.core.management.base import BaseCommand
from django.db import OperationalError, connection
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
            last_cohort_refresh = 0
            last_dependency_order_refresh = 0
            dependency_order_cursor = 0
            while True:
                if settings.EXECUTION_V2_ENABLED:
                    if time.monotonic() - last_priority_refresh >= 5:
                        from api.execution.priority import refresh_stage_priorities

                        refresh_stage_priorities()
                        last_priority_refresh = time.monotonic()
                    if time.monotonic() - last_dependency_order_refresh >= 5:
                        from api.execution.dependency_order import refresh_dependency_order

                        try:
                            deadline = time.monotonic() + 1
                            for _ in range(8):
                                dependency_order_cursor, _ = refresh_dependency_order(
                                    dependency_order_cursor, limit=32
                                )
                                if not dependency_order_cursor or time.monotonic() >= deadline:
                                    break
                        except OperationalError:
                            import logging

                            logging.getLogger(__name__).warning(
                                "Shared dependency queue-age refresh deferred"
                            )
                        last_dependency_order_refresh = time.monotonic()
                    from api.execution.preparation import candidate_rules

                    preparation_rules = candidate_rules()
                    for execution in (
                        ExecutionSpec.objects.filter(
                            paused=False,
                            stages__key="prepare",
                            stages__state="ready",
                            stages__available_at__lte=timezone.now(),
                        )
                        .filter(preparation_rules)
                        .select_related("job")
                        .distinct()
                        .order_by(
                            "background",
                            "-priority",
                            Coalesce("job__submission_time", "created_at"),
                            "pk",
                        )[:4]
                    ):
                        # A priority request arriving during this batch should
                        # not wait behind the other prefetched normal requests.
                        if higher_preparation(execution).filter(preparation_rules).exists():
                            break
                        from api.execution.preparation import waiting_reason

                        reason = waiting_reason(execution, settings.EXECUTION_BLOB_ROOT)
                        if reason:
                            execution.stages.filter(key="prepare").update(
                                waiting_reason=reason,
                                available_at=timezone.now() + timedelta(seconds=60),
                            )
                            continue
                        try:
                            prepare(
                                execution,
                                Path(settings.MEDIA_ROOT),
                                Path(settings.EXECUTION_BLOB_ROOT),
                            )
                            from api.execution.dependency_order import refresh_dependency_order

                            # Newly prepared old requests must not wait for a
                            # full metadata sweep before their shared inputs.
                            try:
                                refresh_dependency_order(execution_id=execution.pk)
                            except OperationalError:
                                pass  # The bounded periodic sweep retries this metadata.
                            execution.refresh_from_db(fields=["priority"])
                            if execution.priority or not execution.background:
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
                    if time.monotonic() - last_cohort_refresh >= 10:
                        from api.execution.cohorts import coalesce

                        try:
                            coalesce()
                        except OperationalError:
                            import logging

                            logging.getLogger(__name__).warning(
                                "Protein cohort planning deferred; existing stages remain eligible"
                            )
                        last_cohort_refresh = time.monotonic()
                    from api.services.prediction_store import upsert_many, upsert_similarity_many

                    for event in (
                        ExecutionOutbox.objects.filter(published_at__isnull=True)
                        .exclude(event__has_key="progress_execution_ids")
                        .order_by("id")[:100]
                    ):
                        if event.event.get("split_cohort"):
                            from api.execution.cohorts import split_cohort

                            split_cohort(event.event["split_cohort"])
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
                    from api.execution.progress import project_pending

                    project_pending()
                if options["once"]:
                    return
                time.sleep(1)
        finally:
            with connection.cursor() as cursor:
                cursor.execute("SELECT pg_advisory_unlock(%s)", [481726391])
