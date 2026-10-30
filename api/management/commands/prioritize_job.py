"""Raise one durable request and its prerequisites without recreating the job."""

from api.execution.models import ExecutionSpec
from api.execution.priority import MAX_PRIORITY, lock_priorities, refresh_stage_priorities
from api.models import Job
from django.core.management.base import BaseCommand, CommandError
from django.db import transaction


class Command(BaseCommand):
    help = (
        "Prioritize a job and shared prerequisites. Running attempts are not preempted; "
        "resource limits, holds and retry backoff still apply."
    )

    def add_arguments(self, parser):
        parser.add_argument("public_id")
        group = parser.add_mutually_exclusive_group()
        group.add_argument(
            "--priority",
            type=int,
            default=100,
            help="Higher numbers run first (default: 100; normal jobs: 0)",
        )
        group.add_argument(
            "--clear", action="store_true", help="Restore normal submission-order priority"
        )
        parser.add_argument(
            "--resume", action="store_true", help="Also resume this job if individually paused"
        )

    @transaction.atomic
    def handle(self, *args, **options):
        value = 0 if options["clear"] else options["priority"]
        if not 0 <= value <= MAX_PRIORITY:
            raise CommandError(f"Priority must be between 0 and {MAX_PRIORITY}")
        lock_priorities()
        try:
            job = Job.objects.get(public_id=options["public_id"])
        except Job.DoesNotExist as exc:
            raise CommandError("Job does not exist") from exc
        if job.status == "Completed":
            self.stdout.write(
                f"{job.public_id} is already completed; nothing was requeued or recomputed."
            )
            return
        if job.status == "Failed":
            raise CommandError(
                "Job is Failed; diagnose/recover it before changing scheduling priority"
            )
        try:
            execution = ExecutionSpec.objects.get(job=job)
        except ExecutionSpec.DoesNotExist as exc:
            raise CommandError(
                "Job has no durable execution specification; original options must be recovered"
            ) from exc
        if execution.paused and not options["resume"] and value:
            raise CommandError(
                "Job is individually paused; use --resume to resume this job as well"
            )
        fields = {"priority": value}
        if options["resume"]:
            fields["paused"] = False
        ExecutionSpec.objects.filter(pk=execution.pk).update(**fields)
        changed = refresh_stage_priorities()
        self.stdout.write(
            f"{job.public_id}: priority {value}; {changed} stage priorities updated. "
            "Resource limits and running attempts are unchanged."
        )
        if execution.background:
            self.stdout.write(
                "Background class still applies; normal requests always take precedence."
            )
        held = execution.stages.filter(state="held").count()
        if held:
            self.stdout.write(f"{held} held stages still require operator attention.")
