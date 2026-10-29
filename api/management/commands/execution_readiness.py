"""Fail closed before canary admission or reopening submissions."""

import json
from datetime import datetime, timedelta
from pathlib import Path

from api.execution.control import require_postgres
from api.execution.models import ExecutionAttempt, ExecutionSpec, ExecutorHost, WorkStage
from api.execution.validation import gpu_numerical_waiver
from api.models import Job
from django.core.management.base import BaseCommand, CommandError
from django.utils import timezone


def canary_timestamps(evidence, hosts):
    """Count canary completions from admission, observation from latest recovery."""

    def parse(value):
        parsed = datetime.fromisoformat(value)
        if timezone.is_naive(parsed):
            raise CommandError("Canary timestamps must include their UTC offset")
        return parsed

    start = parse(evidence["canary_started_at"])
    recovered = [start]
    if evidence.get("canary_healthy_since"):
        recovered.append(parse(evidence["canary_healthy_since"]))
    for host in hosts:
        if host.telemetry.get("control_recovered_at"):
            recovered.append(parse(host.telemetry["control_recovered_at"]))
    return start, max(recovered)


class Command(BaseCommand):
    help = "Validate migration, profiles, hosts and canary evidence before resuming."

    def add_arguments(self, parser):
        parser.add_argument("evidence", type=Path)
        parser.add_argument("--phase", choices=["canary", "reopen"], default="canary")

    def handle(self, *args, **options):
        require_postgres()
        evidence = json.loads(options["evidence"].read_text())
        errors = []
        for key in [
            "source_inventories_verified",
            "queue_reconciliation_verified",
            "backups_verified",
            "restoration_rehearsed",
            "database_import_verified",
            "quota_reconciliation_verified",
            "compatibility_recovery_suite_passed",
            "legacy_gpu_requests_drained",
            "legacy_gpu_proxy_active",
        ]:
            if evidence.get(key) is not True:
                errors.append("Missing acceptance evidence: " + key)
        if evidence.get("scientific_parity_passed") is not True and not (
            gpu_numerical_waiver(evidence)
            and evidence.get("cpu_prediction_compatibility_passed") is True
            and evidence.get("output_structure_passed") is True
        ):
            errors.append(
                "Scientific compatibility or an explicit GPU numerical waiver is required"
            )
        manifest = json.loads(Path(evidence["migration_manifest"]).read_text())
        if not manifest.get("ready") or manifest.get("errors"):
            errors.append("Source queue reconciliation is incomplete")
        expected = set(manifest["specifications"])
        imported = set(
            ExecutionSpec.objects.filter(job__public_id__in=expected).values_list(
                "job__public_id", flat=True
            )
        )
        if expected != imported:
            errors.append("An unfinished source job has no imported specification")
        for public_id in expected:
            execution = ExecutionSpec.objects.get(job__public_id=public_id)
            if not execution.specification.get("model_versions"):
                errors.append("Missing model identity: " + public_id)
        stages = WorkStage.objects.exclude(execution__job__status__in=["Completed", "Failed"])
        if stages.filter(state="held").exists():
            errors.append("Unfinished stages require operator attention")
        if (
            stages.filter(state="ready")
            .exclude(kind__in=["prepare", "barrier"])
            .filter(profile__isnull=True)
            .exists()
        ):
            errors.append("Unprofiled stages remain")
        if stages.filter(kind="prepare", state="ready").exists():
            from api.execution.artifacts import sha256_file
            from api.execution.models import ResourceProfile

            coverage_path = evidence.get("coverage_report")
            coverage = json.loads(Path(coverage_path).read_text()) if coverage_path else {}
            if (
                not coverage.get("ready")
                or coverage.get("errors")
                or coverage.get("manifest_checksum")
                != sha256_file(Path(evidence["migration_manifest"]))
                or set(coverage.get("jobs", {})) != expected
            ):
                errors.append(
                    "Deferred preparation requires a complete frozen-request coverage audit"
                )
            else:
                for execution in ExecutionSpec.objects.filter(
                    job__public_id__in=expected
                ).select_related("job"):
                    entry = coverage["jobs"][execution.job.public_id]
                    if (
                        entry.get("specification") != execution.checksum
                        or entry.get("input") != execution.input_checksum
                    ):
                        errors.append("Coverage identity changed: " + execution.job.public_id)
                fingerprints = set(coverage.get("profile_fingerprints", []))
                available = set(
                    ResourceProfile.objects.filter(
                        fingerprint__in=fingerprints, validated=True, suspended=False
                    ).values_list("fingerprint", flat=True)
                )
                advertised = {
                    fingerprint
                    for host in ExecutorHost.objects.all()
                    for fingerprint in host.capabilities.get("profiles", [])
                }
                if fingerprints != available or not fingerprints <= advertised:
                    errors.append(
                        "A coverage profile is suspended, changed, or unavailable on the executors"
                    )
                cohort = evidence.get("canary_jobs", [])
                if (
                    not isinstance(cohort, list)
                    or len(set(cohort)) < 100
                    or not set(cohort) <= expected
                ):
                    errors.append("Select at least 100 preserved jobs for canary execution")
                elif stages.filter(
                    execution__job__public_id__in=cohort, kind="prepare", state="ready"
                ).exists():
                    errors.append("Canary requests must be prepared before admission")
        for host_id in ["production", "gpu"]:
            host = ExecutorHost.objects.filter(pk=host_id).first()
            if (
                not host
                or not host.reconciled
                or not host.heartbeat_at
                or host.heartbeat_at < timezone.now() - timedelta(seconds=10)
            ):
                errors.append("Executor unavailable or unreconciled: " + host_id)
            elif host_id == "gpu" and not host.telemetry.get("gpu_uuid"):
                errors.append("GPU telemetry unavailable")
            if host and host.telemetry.get("legacy_consumers"):
                errors.append("Legacy consumers remain active: " + host_id)
        if ExecutionAttempt.objects.filter(state="uncertain").exists():
            errors.append("Uncertain containers retain reservations")
        if options["phase"] == "reopen":
            start, observation = canary_timestamps(evidence, ExecutorHost.objects.all())
            if timezone.now() - observation < timedelta(hours=1):
                errors.append("Healthy canary observation is shorter than one hour")
            completed = Job.objects.filter(
                execution__isnull=False,
                completion_time__gte=start,
                status="Completed",
                public_id__in=expected,
            ).count()
            if completed < 100:
                errors.append(
                    f"Canary completed only {completed} imported jobs; at least 100 required"
                )
            if evidence.get("canary_method_and_shape_coverage_passed") is not True:
                errors.append("Canary method/shape/interrupted-job coverage is incomplete")
        if errors:
            raise CommandError("\n".join(errors))
        self.stdout.write(
            "Acceptance gates passed; this command does not resume admissions or submissions"
        )
