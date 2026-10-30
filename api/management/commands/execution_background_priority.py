import ipaddress
import json
import re

from api.execution.background import apply_background_rules, lock_policy
from api.execution.models import SchedulingRule
from django.core.management.base import BaseCommand, CommandError
from django.db import transaction


class Command(BaseCommand):
    help = "Persist lowest-priority IP/key-prefix selectors and classify all unfinished jobs."

    def add_arguments(self, parser):
        parser.add_argument("--ip", action="append", default=[])
        parser.add_argument("--api-key-prefix", action="append", default=[])
        parser.add_argument("--clear", action="store_true", help="Disable the specified selectors")

    @transaction.atomic
    def handle(self, *args, **options):
        rules = []
        for value in options["ip"]:
            try:
                rules.append(("ip", str(ipaddress.ip_address(value))))
            except ValueError as error:
                raise CommandError("Invalid IP address") from error
        for value in options["api_key_prefix"]:
            if not re.fullmatch(r"ak_[0-9a-f]{1,64}", value):
                raise CommandError("Invalid API key prefix")
            rules.append(("api_key_prefix", value))
        if options["clear"] and not rules:
            raise CommandError("Specify the selectors to clear")
        lock_policy()
        for kind, value in rules:
            SchedulingRule.objects.update_or_create(
                kind=kind, value=value, defaults={"enabled": not options["clear"]}
            )
        self.stdout.write(json.dumps(apply_background_rules()))
