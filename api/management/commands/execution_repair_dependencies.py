"""Recover pre-execution cache misses without disabling healthy execution profiles."""

from api.execution.models import ExecutionOutbox, ResourceProfile, WorkStage
from django.core.management.base import BaseCommand
from django.db import transaction
from django.utils import timezone


class Command(BaseCommand):
    help = "Repair held missing-artifact work and profiles suspended only by dependency failures."

    def add_arguments(self, parser):
        parser.add_argument("--dry-run", action="store_true")
        parser.add_argument("--limit", type=int, default=128)

    @transaction.atomic
    def handle(self, *args, **options):
        restored, stages = [], []
        for profile in ResourceProfile.objects.select_for_update().filter(
            suspended=True, validated=True
        ):
            held = profile.workstage_set.filter(state="held")
            if not held.exists() or held.exclude(waiting_reason="missing_embedding").exists():
                continue
            if not profile.workstage_set.filter(
                attempts__state="completed", attempts__profile_fingerprint=profile.fingerprint
            ).exists():
                continue
            profile.suspended = False
            profile.save(update_fields=["suspended"])
            profile.workstage_set.filter(state="ready", waiting_reason="profiling_required").update(
                waiting_reason=""
            )
            restored.append(profile.pk)
        candidates = (
            WorkStage.objects.select_for_update(skip_locked=True, of=("self",))
            .filter(state="held", waiting_reason="missing_embedding")
            .exclude(execution__job__status__in=["Completed", "Failed"])[
                : max(1, min(options["limit"], 512))
            ]
        )
        for stage in candidates:
            if stage.attempts.filter(state__in=["leased", "running", "uncertain"]).exists():
                continue
            stage.state = "recovering"
            stage.available_at = timezone.now()
            stage.save(update_fields=["state", "available_at"])
            ExecutionOutbox.objects.get_or_create(
                key="dependency-repair:" + str(stage.pk),
                defaults={"event": {"recover_stage": str(stage.pk), "host_id": None}},
            )
            stages.append(str(stage.pk))
        result = {
            "restored_profiles": restored,
            "recovering_stages": stages,
            "dry_run": options["dry_run"],
        }
        if options["dry_run"]:
            transaction.set_rollback(True)
        self.stdout.write(str(result))
