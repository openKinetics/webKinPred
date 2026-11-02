import json

from api.execution.control import bind_profiles
from api.execution.recovery import reconcile_pending
from django.core.management.base import BaseCommand
from django.db import transaction


class Command(BaseCommand):
    help = "Batch pending legacy recoveries without stopping running attempts."

    def add_arguments(self, parser):
        parser.add_argument("--limit", type=int, default=256)
        parser.add_argument("--dry-run", action="store_true")

    def handle(self, *args, **options):
        with transaction.atomic():
            report = reconcile_pending(max(1, min(options["limit"], 512)))
            if options["dry_run"]:
                transaction.set_rollback(True)
        if not options["dry_run"]:
            bind_profiles()
        self.stdout.write(json.dumps({**report, "dry_run": options["dry_run"]}))
