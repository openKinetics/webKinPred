import json
import os
from pathlib import Path

from api.execution.artifacts import atomic_json
from api.models import Job
from django.conf import settings
from django.core.management.base import BaseCommand, CommandError
from tools.executor.migration import export_broker, normalize_request, reconcile
from webKinPred.celery import app


class Command(BaseCommand):
    help = "Export and reconcile the legacy queue without deleting messages or job rows."

    def add_arguments(self, parser):
        parser.add_argument("directory", type=Path)
        parser.add_argument("--workers-from", type=Path)
        parser.add_argument("--writers-quiesced", action="store_true")

    def handle(self, *args, **options):
        import redis

        destination = options["directory"].resolve()
        destination.mkdir(parents=True, exist_ok=False)
        os.chmod(destination, 0o700)
        previous = options.get("workers_from")
        if previous:
            if not options.get("writers_quiesced"):
                raise CommandError("Reusing captured worker requests requires --writers-quiesced")
            buckets = json.loads(previous.read_text())
            if set(buckets) != {"active", "reserved", "scheduled"}:
                raise CommandError("Incomplete captured worker inventory")
            from api.execution.artifacts import sha256_file

            atomic_json(
                destination / "worker-capture-provenance.json",
                {
                    "source": str(previous),
                    "sha256": sha256_file(previous),
                    "writers_quiesced": True,
                },
            )
        else:
            inspector = app.control.inspect(timeout=10)
            buckets = {
                "active": inspector.active(),
                "reserved": inspector.reserved(),
                "scheduled": inspector.scheduled(),
            }
        if any(value is None for value in buckets.values()):
            raise CommandError(
                "Worker inspection did not reply; stop and reconcile manually before cutover"
            )
        atomic_json(destination / "workers.json", buckets)
        requests = []
        for bucket in buckets.values():
            for worker_tasks in bucket.values():
                for task in worker_tasks:
                    task = task.get("request", task)
                    requests.append(
                        normalize_request(
                            task["name"],
                            task.get("args", []),
                            task.get("kwargs", {}),
                            task.get("id"),
                        )
                    )
        records = export_broker(
            redis.Redis.from_url(settings.CELERY_BROKER_URL), destination / "broker"
        )
        from django_redis import get_redis_connection

        quotas = get_redis_connection("default")
        atomic_json(
            destination / "quota.json",
            [
                {"key": key.decode(), "value": int(quotas.get(key)), "ttl_ms": quotas.pttl(key)}
                for key in quotas.scan_iter(match="quota:*")
                if quotas.get(key) is not None
            ],
        )
        requests.extend(record["request"] for record in records if "request" in record)
        jobs = list(Job.objects.values())
        # Preserve identifiers/timestamps; the database backup remains authoritative.
        atomic_json(destination / "jobs.json", json.loads(json.dumps(jobs, default=str)))
        manifest = reconcile(jobs, requests)
        atomic_json(destination / "reconciliation.json", manifest)
        self.stdout.write(
            f"Exported {len(jobs)} jobs; {manifest['unfinished_count']} unfinished; "
            f"ready={manifest['ready']}"
        )
        if not manifest["ready"]:
            raise CommandError("Reconciliation is incomplete; original queue has been preserved")
