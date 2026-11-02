"""Reconcile completed result summaries without using the prediction queue."""

import json
import time

from api.services.about_stats_service import refresh_about_stats_cache, sync_completed_jobs
from django.core.management.base import BaseCommand
from django.db import connection

# 391 controller, 392 graph/cleanup, 393 priority, 394 policy, 395 metrics.
ABOUT_STATS_WORKER_LOCK = 481726396


class Command(BaseCommand):
    help = "Incrementally backfill/refresh About statistics; singleton on PostgreSQL."

    def add_arguments(self, parser):
        parser.add_argument("--once", action="store_true")
        parser.add_argument("--batch-size", type=int, default=20)
        parser.add_argument("--interval", type=float, default=30)

    def handle(self, *args, **options):
        locked = False
        if connection.vendor == "postgresql":
            with connection.cursor() as cursor:
                cursor.execute("SELECT pg_try_advisory_lock(%s)", [ABOUT_STATS_WORKER_LOCK])
                locked = cursor.fetchone()[0]
                cursor.execute("SET statement_timeout = 30000")
                cursor.execute("SET max_parallel_workers_per_gather = 0")
                cursor.execute("SET work_mem = '16MB'")
            if not locked:
                self.stdout.write("Another About statistics worker owns reconciliation")
                return
        try:
            last_log = 0
            while True:
                count = sync_completed_jobs(limit=options["batch_size"])
                payload = refresh_about_stats_cache(force=options["once"])
                if time.monotonic() - last_log >= 30:
                    self.stdout.write(
                        json.dumps(
                            {"processed_batch": count, "snapshot_ready": payload is not None}
                        )
                    )
                    self.stdout.flush()
                    last_log = time.monotonic()
                if options["once"]:
                    self.stdout.write(
                        str({"processed": count, "snapshot_ready": payload is not None})
                    )
                    return
                time.sleep(0.1 if count else max(1, options["interval"]))
        finally:
            if locked:
                with connection.cursor() as cursor:
                    cursor.execute("SELECT pg_advisory_unlock(%s)", [ABOUT_STATS_WORKER_LOCK])
