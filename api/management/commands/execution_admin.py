import json

from api.execution.control import bind_profiles, require_postgres
from api.execution.models import ExecutionSpec, ExecutorHost, ResourceProfile, WorkStage
from django.core.management.base import BaseCommand, CommandError
from django.db import transaction
from django.utils import timezone


class Command(BaseCommand):
    help = "Inspect, pause, resume and recover durable execution; never purges broker queues."

    def add_arguments(self, parser):
        parser.add_argument(
            "action",
            choices=[
                "status",
                "pause",
                "resume",
                "retry",
                "disable-profile",
                "bind-profiles",
                "drain",
                "drain-gpu",
                "resume-gpu",
                "stop",
                "explain",
                "pause-submissions",
                "resume-submissions",
            ],
        )
        parser.add_argument("--host")
        parser.add_argument("--stage")
        parser.add_argument("--profile")

    @transaction.atomic
    def handle(self, *args, **options):
        action = options["action"]
        if action in {"pause-submissions", "resume-submissions"}:
            from pathlib import Path

            from django.conf import settings

            path = Path(settings.EXECUTION_MAINTENANCE_FILE)
            if action == "pause-submissions":
                path.parent.mkdir(parents=True, exist_ok=True)
                path.touch()
            else:
                path.unlink(missing_ok=True)
        elif action in {"drain-gpu", "resume-gpu"}:
            if not options["host"]:
                raise CommandError("--host required")
            host = ExecutorHost.objects.select_for_update().get(pk=options["host"])
            host.gpu_paused = action == "drain-gpu"
            host.save(update_fields=["gpu_paused"])
            self.stdout.write(
                "GPU admissions paused; CPU work continues"
                if host.gpu_paused
                else "GPU admissions resumed"
            )
        elif action in {"drain", "stop"}:
            if not options["host"]:
                raise CommandError("--host required")
            host = ExecutorHost.objects.select_for_update().get(pk=options["host"])
            host.paused = True
            host.save(update_fields=["paused"])
            if action == "stop":
                host.attempts.filter(state__in=["leased", "running", "uncertain"]).update(
                    stop_requested=True
                )
            count = host.attempts.filter(state__in=["leased", "running", "uncertain"]).count()
            self.stdout.write(
                f"Admissions paused; {count} attempts retain reservations until stopped"
            )
        elif action == "explain":
            stage = WorkStage.objects.get(pk=options["stage"])
            self.stdout.write(
                json.dumps(
                    {
                        "state": stage.state,
                        "waiting_reason": stage.waiting_reason,
                        "paused": stage.execution.paused,
                        "profile": stage.profile_id,
                        "dependencies": list(
                            stage.dependencies.exclude(state="completed").values(
                                "id", "state", "waiting_reason"
                            )
                        ),
                        "unavailable_artifacts": list(
                            stage.artifacts.exclude(artifact__locations__ready=True).values_list(
                                "artifact_id", flat=True
                            )
                        ),
                        "attempts": list(
                            stage.attempts.values(
                                "id", "host_id", "state", "resources", "lease_until"
                            )
                        ),
                    },
                    default=str,
                )
            )
        elif action == "status":
            self.stdout.write(
                json.dumps(
                    {
                        "hosts": list(
                            ExecutorHost.objects.values(
                                "id", "paused", "reconciled", "heartbeat_at"
                            )
                        ),
                        "stages": list(
                            WorkStage.objects.values(
                                "id",
                                "execution__job__public_id",
                                "state",
                                "waiting_reason",
                                "profile_id",
                                "failures",
                            )
                        ),
                    },
                    default=str,
                )
            )
        elif action in {"pause", "resume"}:
            if action == "resume":
                require_postgres()
            if options["host"]:
                if not ExecutorHost.objects.filter(pk=options["host"]).update(
                    paused=action == "pause"
                ):
                    raise CommandError("Unknown host")
            else:
                ExecutionSpec.objects.exclude(job__status__in=["Completed", "Failed"]).update(
                    paused=action == "pause"
                )
        elif action == "retry":
            if not options["stage"]:
                raise CommandError("--stage required")
            stage = WorkStage.objects.select_for_update().get(pk=options["stage"])
            if (
                stage.state != "held"
                or stage.attempts.filter(state__in=["running", "leased", "uncertain"]).exists()
            ):
                raise CommandError("Only held stages with no live attempt may be retried")
            stage.state = "ready"
            stage.available_at = timezone.now()
            stage.failures = 0
            stage.save(update_fields=["state", "available_at", "failures"])
        elif action == "disable-profile":
            if not options["profile"]:
                raise CommandError("--profile required")
            ResourceProfile.objects.filter(pk=options["profile"]).update(suspended=True)
        elif action == "bind-profiles":
            bind_profiles()
