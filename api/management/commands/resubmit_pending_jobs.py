"""
Flush all Pending jobs and resubmit them as fresh tasks.

Strategy
--------
1. Snapshot every Pending job from the DB.
2. Purge the entire Celery queue (removes all enqueued but not-yet-started tasks).
3. Revoke any active/reserved/scheduled tasks whose first arg matches a pending job ID.
4. Delete the stale Job rows.
5. Recreate each Job with a clean state and fire run_multi_prediction.delay().

Run with:
    python manage.py resubmit_pending_jobs
    python manage.py resubmit_pending_jobs --dry-run
    python manage.py resubmit_pending_jobs --ids qew3T0D 9IitFoz
"""

import ast

from django.conf import settings
from django.core.management.base import BaseCommand, CommandError
from django.utils import timezone

from api.models import Job
from api.tasks import run_multi_prediction
from api.utils.job_utils import canonicalise_targets
from webKinPred.celery import app


def _parse_task_args(raw):
    if isinstance(raw, (list, tuple)):
        return list(raw)
    if isinstance(raw, str):
        try:
            v = ast.literal_eval(raw)
            return list(v) if isinstance(v, (list, tuple)) else []
        except Exception:
            return []
    return []


class Command(BaseCommand):
    help = "Flush all Pending jobs and resubmit them as fresh Celery tasks"

    def add_arguments(self, parser):
        parser.add_argument(
            "--ids",
            nargs="+",
            metavar="PUBLIC_ID",
            help="Only resubmit specific job IDs (default: all Pending jobs)",
        )
        parser.add_argument(
            "--dry-run",
            action="store_true",
            help="Print what would happen without making any changes",
        )

    def handle(self, *args, **options):
        if settings.EXECUTION_V2_ENABLED:
            raise CommandError(
                "Durable execution preserves job identities and request options. "
                "Use execution_admin retry/resume; legacy queue purge is disabled."
            )
        dry_run = options["dry_run"]
        specific_ids = options.get("ids")

        qs = Job.objects.filter(status="Pending")
        if specific_ids:
            qs = qs.filter(public_id__in=specific_ids)

        jobs = list(qs.select_related("user"))

        if not jobs:
            self.stdout.write(self.style.WARNING("No Pending jobs found — nothing to do."))
            return

        self.stdout.write(f"Found {len(jobs)} Pending job(s):")
        for j in jobs:
            self.stdout.write(f"  {j.public_id}  {j.prediction_type}  rows={j.requested_rows}")

        if dry_run:
            self.stdout.write(self.style.WARNING("Dry-run mode — no changes made."))
            return

        # ------------------------------------------------------------------
        # Step 1: snapshot
        # ------------------------------------------------------------------
        snapshots = []
        for job in jobs:
            snapshots.append(
                {
                    "public_id": job.public_id,
                    "prediction_type": job.prediction_type,
                    "ip_address": job.ip_address,
                    "quota_subject": job.quota_subject,
                    "requested_rows": job.requested_rows,
                    "kcat_method": job.kcat_method,
                    "km_method": job.km_method,
                    "kcat_km_method": job.kcat_km_method,
                    "handle_long_sequences": job.handle_long_sequences,
                    "canonicalize_substrates": job.canonicalize_substrates,
                    "recon_xkg": job.recon_xkg,
                    "user_id": job.user_id,
                    "submission_time": job.submission_time,
                }
            )

        pending_ids = {s["public_id"] for s in snapshots}

        # ------------------------------------------------------------------
        # Step 2: purge the Celery queue entirely (removes all enqueued msgs)
        # ------------------------------------------------------------------
        self.stdout.write("Purging Celery queue…")
        try:
            purged = app.control.purge()
            self.stdout.write(f"  Purged {purged} message(s) from the broker queue.")
        except Exception as exc:
            self.stdout.write(self.style.WARNING(f"  Queue purge failed (non-fatal): {exc}"))

        # ------------------------------------------------------------------
        # Step 3: revoke tasks that are already being worked on by a worker
        # ------------------------------------------------------------------
        self.stdout.write("Inspecting active/reserved/scheduled Celery tasks…")
        try:
            insp = app.control.inspect(timeout=5)
            active = insp.active() or {}
            reserved = insp.reserved() or {}
            scheduled = insp.scheduled() or {}
        except Exception as exc:
            self.stdout.write(self.style.WARNING(f"  Inspect failed (non-fatal): {exc}"))
            active = reserved = scheduled = {}

        revoked = []
        for bucket_name, bucket in [("active", active), ("reserved", reserved)]:
            for _, tasks in bucket.items():
                for t in tasks:
                    args = _parse_task_args(t.get("args", []))
                    if args and str(args[0]) in pending_ids:
                        tid = t.get("id")
                        if tid:
                            app.control.revoke(tid, terminate=True, signal="SIGTERM")
                            revoked.append((bucket_name, tid, str(args[0])))

        for _, tasks in scheduled.items():
            for t in tasks:
                req = t.get("request", {}) if isinstance(t, dict) else {}
                args = _parse_task_args(req.get("args", []))
                if args and str(args[0]) in pending_ids:
                    tid = req.get("id")
                    if tid:
                        app.control.revoke(tid, terminate=True, signal="SIGTERM")
                        revoked.append(("scheduled", tid, str(args[0])))

        if revoked:
            for bucket, tid, pid in revoked:
                self.stdout.write(f"  Revoked [{bucket}] task {tid} for job {pid}")
        else:
            self.stdout.write("  No active/reserved/scheduled tasks found for these jobs.")

        # ------------------------------------------------------------------
        # Step 4: delete stale DB rows
        # ------------------------------------------------------------------
        self.stdout.write("Deleting stale Pending job rows…")
        Job.objects.filter(public_id__in=pending_ids).delete()
        self.stdout.write(f"  Deleted {len(snapshots)} row(s).")

        # ------------------------------------------------------------------
        # Step 5: recreate + resubmit
        # ------------------------------------------------------------------
        self.stdout.write("Recreating jobs and resubmitting…")
        for snap in snapshots:
            new_job = Job(
                public_id=snap["public_id"],
                prediction_type=snap["prediction_type"],
                ip_address=snap["ip_address"],
                quota_subject=snap["quota_subject"],
                requested_rows=snap["requested_rows"],
                kcat_method=snap["kcat_method"],
                km_method=snap["km_method"],
                kcat_km_method=snap["kcat_km_method"],
                handle_long_sequences=snap["handle_long_sequences"],
                canonicalize_substrates=snap["canonicalize_substrates"],
                recon_xkg=snap.get("recon_xkg", False),
                user_id=snap["user_id"],
                submission_time=snap["submission_time"],
                status="Pending",
                start_time=None,
                completion_time=None,
                error_message=None,
                total_molecules=0,
                molecules_processed=0,
                invalid_rows=0,
                total_predictions=0,
                predictions_made=0,
            )
            new_job.save()

            targets = []
            methods = {}
            if snap["kcat_method"]:
                targets.append("kcat")
                methods["kcat"] = snap["kcat_method"]
            if snap["km_method"]:
                targets.append("Km")
                methods["Km"] = snap["km_method"]
            if snap["kcat_km_method"]:
                targets.append("kcat/Km")
                methods["kcat/Km"] = snap["kcat_km_method"]

            ordered = canonicalise_targets(targets)

            run_multi_prediction.delay(
                snap["public_id"],
                ordered,
                methods,
                {},
                bool(snap["canonicalize_substrates"]),
                True,
                False,
                bool(snap.get("recon_xkg", False)),
            )

            self.stdout.write(
                self.style.SUCCESS(
                    f"  Requeued {snap['public_id']}  targets={ordered}  methods={methods}"
                )
            )

        self.stdout.write(self.style.SUCCESS(f"\nDone — {len(snapshots)} job(s) resubmitted."))
