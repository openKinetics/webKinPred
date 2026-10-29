"""Check the full frozen backlog without eagerly creating all artifact graphs."""

import json
from pathlib import Path

from api.execution.artifacts import atomic_json, sha256_file
from api.execution.coverage import Coverage
from api.execution.models import ExecutionSpec, ResourceProfile
from django.conf import settings
from django.core.management.base import BaseCommand, CommandError


class Command(BaseCommand):
    help = "Read-only request/input/profile audit for deferred stage preparation."

    def add_arguments(self, parser):
        parser.add_argument("manifest", type=Path)
        parser.add_argument("destination", type=Path)
        parser.add_argument("--registry", type=Path, action="append")

    def handle(self, *args, **options):
        manifest = json.loads(options["manifest"].read_text())
        if not manifest.get("ready") or manifest.get("errors"):
            raise CommandError("The source queue must reconcile before coverage auditing")
        expected = set(manifest["specifications"])
        fingerprints = {
            profile["fingerprint"]
            for path in options.get("registry") or []
            for profile in json.loads(path.read_text())
        }
        profiles = ResourceProfile.objects.filter(validated=True, suspended=False).order_by("id")
        if options.get("registry"):
            profiles = profiles.filter(fingerprint__in=fingerprints)
        audit = Coverage(profiles)
        verified, errors = {}, {}
        for execution in (
            ExecutionSpec.objects.filter(job__public_id__in=expected)
            .select_related("job")
            .order_by("pk")
            .iterator(chunk_size=64)
        ):
            try:
                verified[execution.job.public_id] = audit.execution(execution, settings.MEDIA_ROOT)
            except (ValueError, OSError) as error:
                errors[execution.job.public_id] = str(error)
            done = len(verified) + len(errors)
            if done % 100 == 0:
                self.stdout.write(f"Checked {done} requests; {len(errors)} require attention")
        for missing in expected - verified.keys() - errors.keys():
            errors[missing] = "The preserved request has no imported execution specification"
        result = {
            "ready": not errors and set(verified) == expected,
            "manifest_checksum": sha256_file(options["manifest"]),
            "jobs": verified,
            "errors": errors,
            "profile_fingerprints": sorted(audit.used),
        }
        atomic_json(options["destination"], result)
        if not result["ready"]:
            raise CommandError(f"Coverage audit preserved {len(errors)} explicit holds for review")
        self.stdout.write(f"Verified all {len(verified)} preserved requests without creating work")
