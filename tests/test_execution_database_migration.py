"""Rehearse old SQLite schema import into disposable, migrated PostgreSQL."""

import os
import sqlite3
import subprocess
import sys
from io import StringIO
from pathlib import Path

import pytest
from api.models import ApiKey, ApiUser, Job
from django.core.management import call_command
from django.db import connection


@pytest.mark.django_db(transaction=True)
def test_sqlite_import_preserves_identity_relationships_and_repeats_safely(tmp_path):
    root = Path(__file__).resolve().parents[1]
    source = tmp_path / "legacy.sqlite3"
    (tmp_path / "legacy_migration_settings.py").write_text(
        "from webKinPred.settings import *\n"
        f"DATABASES={{'default':{{'ENGINE':'django.db.backends.sqlite3','NAME':{str(source)!r}}}}}\n"
        "DATABASE_ROUTERS=[]\n"
    )
    environment = {
        **os.environ,
        "DJANGO_SETTINGS_MODULE": "legacy_migration_settings",
        "PYTHONPATH": str(tmp_path) + os.pathsep + str(root),
        "DJANGO_SECRET_KEY": "isolated-migration-only",
    }
    subprocess.run(
        [sys.executable, "manage.py", "migrate", "api", "0022", "--noinput"],
        cwd=root,
        env=environment,
        check=True,
        capture_output=True,
    )
    subprocess.run(
        [
            sys.executable,
            "-c",
            "import django; django.setup(); "
            "from api.models import Job,ApiUser,ApiKey; "
            "u=ApiUser.objects.create(id=804,ip_address='192.0.2.4',"
            "custom_daily_limit=9999999999999); "
            "ApiKey.objects.create(user=u,custom_daily_limit=9999999999999); "
            "Job.objects.create(job_id=1204,public_id='oldjob1',user=u,status='Processing',"
            "prediction_type='kcat',requested_rows=12,quota_subject='legacy:804')",
        ],
        cwd=root,
        env=environment,
        check=True,
        capture_output=True,
    )
    with sqlite3.connect(source) as source_db:
        original_time = source_db.execute(
            "SELECT submission_time FROM api_job WHERE job_id=1204"
        ).fetchone()[0]
    try:
        call_command("execution_migrate_database", source, stdout=StringIO())
        job = Job.objects.get(public_id="oldjob1")
        assert job.pk == 1204 and job.user_id == 804
        assert job.status == "Processing" and job.requested_rows == 12
        assert job.quota_subject == "legacy:804"
        assert job.submission_time.replace(tzinfo=None).isoformat(sep=" ") == original_time
        assert ApiUser.objects.get(pk=804).ip_address == "192.0.2.4"
        assert ApiUser.objects.get(pk=804).custom_daily_limit == 9999999999999
        assert ApiKey.objects.get(user_id=804).custom_daily_limit == 9999999999999
        call_command("execution_migrate_database", source, stdout=StringIO())
        assert Job.objects.filter(public_id="oldjob1").count() == 1
        assert Job.objects.create(public_id="nextjob", prediction_type="kcat").pk > 1204
    finally:
        with connection.cursor() as cursor:
            cursor.execute("DROP TABLE IF EXISTS execution_import_audit")


@pytest.mark.django_db(transaction=True)
def test_sequence_import_preserves_long_sequences_and_collision_suffixes(tmp_path):
    import hashlib
    import random

    from api.execution.models import SequenceIdentity

    sequence = "".join(random.Random(42).choices("ACDEFGHIKLMNPQRSTVWY", k=15000))
    checksum = hashlib.sha256(sequence.encode()).hexdigest()
    ident = checksum[:12] + "_17"
    source = tmp_path / "sequence-backup.sqlite3"
    with sqlite3.connect(source) as db:
        db.execute(
            "CREATE TABLE sequences (id text primary key, sha256 text unique, seq text unique, "
            "len integer, uses_count integer, created_at text, last_seen_at text)"
        )
        db.execute(
            "INSERT INTO sequences VALUES (?,?,?,?,?,?,?)",
            [ident, checksum, sequence, len(sequence), 23, "2025-01-01", "2026-10-28"],
        )
    try:
        call_command("execution_migrate_database", source, stdout=StringIO())
        record = SequenceIdentity.objects.get(pk=ident)
        assert record.seq == sequence and record.length == 15000
        assert record.uses_count == 23 and record.sha256 == checksum
    finally:
        with connection.cursor() as cursor:
            cursor.execute("DROP TABLE IF EXISTS execution_import_audit")
