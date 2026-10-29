import json
from pathlib import Path

from api.execution.dispatch import submit
from api.execution.models import ExecutionSpec
from api.models import Job
from django.core.management.base import BaseCommand, CommandError
from django.db import transaction


class Command(BaseCommand):
    help = "Idempotently import reconciled requests, preserving jobs and leaving execution paused."

    def add_arguments(self, parser):
        parser.add_argument("manifest", type=Path)
        parser.add_argument("--quota-snapshot", type=Path, required=True)

    @transaction.atomic
    def handle(self, *args, **options):
        manifest = json.loads(options["manifest"].read_text())
        if not manifest.get("ready") or manifest.get("errors"):
            raise CommandError("Migration reconciliation must pass before import")
        if len(manifest["specifications"]) != manifest["unfinished_count"]:
            raise CommandError("Incomplete specification manifest")
        ids = set(manifest["specifications"])
        unfinished = set(
            Job.objects.filter(status__in=["Pending", "Processing"]).values_list(
                "public_id", flat=True
            )
        )
        if ids != unfinished:
            raise CommandError("Manifest does not match the frozen unfinished-job inventory")
        from datetime import date

        from api.execution.control import digest
        from api.execution.models import QuotaDay, QuotaEntry

        for record in json.loads(options["quota_snapshot"].read_text()):
            _, day, subject = record["key"].split(":", 2)
            key = "opening:" + digest(record["key"])
            account, created = QuotaDay.objects.get_or_create(
                subject=subject, day=date.fromisoformat(day), defaults={"used": record["value"]}
            )
            if created:
                QuotaEntry.objects.create(key=key, account=account, amount=record["value"])
            elif not QuotaEntry.objects.filter(key=key).exists():
                raise CommandError("Existing quota bucket has no matching opening balance")
        for ident, request in manifest["specifications"].items():
            request = dict(request)
            request.pop("public_id", None)
            request.pop("task_id", None)
            execution = submit(ident, **request)
            execution.paused = True
            execution.save(update_fields=["paused"])
            job = execution.job
            account, _ = QuotaDay.objects.get_or_create(
                subject=job.quota_subject or job.ip_address, day=job.submission_time.date()
            )
            QuotaEntry.objects.get_or_create(
                key="legacy-reservation:" + str(job.pk),
                defaults={
                    "account": account,
                    "job": job,
                    "amount": 0,
                    "reserved_amount": job.requested_rows,
                },
            )
        if ExecutionSpec.objects.filter(job__public_id__in=ids).count() != len(ids):
            raise CommandError("Imported execution count mismatch")
        self.stdout.write(f"Preserved and imported {len(ids)} jobs; admissions remain paused")
