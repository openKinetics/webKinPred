"""Copy a quiescent SQLite backup into a newly migrated, dedicated PostgreSQL database."""

import sqlite3
from pathlib import Path

from api.execution.artifacts import sha256_file
from django.core.management.base import BaseCommand, CommandError
from django.db import connections, transaction
from psycopg2 import sql


class Command(BaseCommand):
    help = "Import a verified SQLite backup into an empty PostgreSQL database; preserves IDs."

    def add_arguments(self, parser):
        parser.add_argument("source", type=Path)
        parser.add_argument(
            "--database", default="default", choices=["default", "seqmap", "prediction_store"]
        )

    def handle(self, *args, **options):
        source = options["source"].resolve()
        target = connections[options["database"]]
        if target.vendor != "postgresql":
            raise CommandError("Target must be a dedicated PostgreSQL database")
        checksum = sha256_file(source)
        incoming = sqlite3.connect(f"file:{source}?mode=ro", uri=True)
        try:
            if incoming.execute("PRAGMA integrity_check").fetchone()[0] != "ok":
                raise CommandError("SQLite source failed integrity check")
            tables = [
                r[0]
                for r in incoming.execute(
                    "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"
                )
                if not r[0].startswith("sqlite_") and r[0] != "django_migrations"
            ]
            with transaction.atomic(using=options["database"]), target.cursor() as cursor:
                cursor.execute(
                    "CREATE TABLE IF NOT EXISTS execution_import_audit "
                    "(source_sha256 text PRIMARY KEY, "
                    "imported_at timestamptz NOT NULL DEFAULT now())"
                )
                cursor.execute(
                    "SELECT 1 FROM execution_import_audit WHERE source_sha256=%s", [checksum]
                )
                if cursor.fetchone():
                    self.stdout.write("Identical backup already imported; no changes")
                    return
                cursor.execute("SELECT count(*) FROM execution_import_audit")
                if cursor.fetchone()[0]:
                    raise CommandError("A different backup was already imported")
                available = set(target.introspection.table_names(cursor))
                if set(tables) - available:
                    raise CommandError(
                        "Target schema is missing source tables: " + repr(set(tables) - available)
                    )
                for table in tables:
                    if table not in {"django_content_type", "auth_permission"}:
                        cursor.execute(
                            sql.SQL("SELECT count(*) FROM {}").format(sql.Identifier(table))
                        )
                        if cursor.fetchone()[0]:
                            raise CommandError(
                                "Target contains data; refusing replacement: " + table
                            )
                cursor.execute("SET CONSTRAINTS ALL DEFERRED")
                for table in ["auth_permission", "django_content_type"]:
                    if table in tables:
                        cursor.execute(sql.SQL("DELETE FROM {}").format(sql.Identifier(table)))
                for table in tables:
                    columns = [
                        r[1]
                        for r in incoming.execute(
                            'PRAGMA table_info("' + table.replace('"', '""') + '")'
                        )
                    ]
                    cursor.execute(
                        "SELECT column_name,data_type FROM information_schema.columns "
                        "WHERE table_schema=current_schema() AND table_name=%s",
                        [table],
                    )
                    types = dict(cursor.fetchall())
                    if set(columns) - types.keys():
                        raise CommandError("Target columns missing in " + table)
                    query = sql.SQL("INSERT INTO {} ({}) VALUES ({})").format(
                        sql.Identifier(table),
                        sql.SQL(",").join(map(sql.Identifier, columns)),
                        sql.SQL(",").join(sql.Placeholder() for _ in columns),
                    )
                    rows = incoming.execute('SELECT * FROM "' + table.replace('"', '""') + '"')
                    count = 0
                    while batch := rows.fetchmany(1000):
                        values = [
                            tuple(
                                bool(v) if v is not None and types[c] == "boolean" else v
                                for c, v in zip(columns, row, strict=True)
                            )
                            for row in batch
                        ]
                        cursor.executemany(query, values)
                        count += len(values)
                    cursor.execute(sql.SQL("SELECT count(*) FROM {}").format(sql.Identifier(table)))
                    if cursor.fetchone()[0] != count:
                        raise CommandError("Row count mismatch in " + table)
                    for column in columns:
                        cursor.execute("SELECT pg_get_serial_sequence(%s,%s)", [table, column])
                        sequence = cursor.fetchone()[0]
                        if sequence:
                            cursor.execute(
                                sql.SQL("SELECT max({}) FROM {}").format(
                                    sql.Identifier(column), sql.Identifier(table)
                                )
                            )
                            maximum = cursor.fetchone()[0]
                            cursor.execute(
                                "SELECT setval(%s,%s,%s)",
                                [sequence, maximum or 1, maximum is not None],
                            )
                    self.stdout.write(f"{table}: {count} rows")
                cursor.execute("SET CONSTRAINTS ALL IMMEDIATE")
                cursor.execute(
                    "INSERT INTO execution_import_audit(source_sha256) VALUES (%s)", [checksum]
                )
            if options["database"] == "default":
                from django.core.management.sql import emit_post_migrate_signal

                emit_post_migrate_signal(verbosity=0, interactive=False, db="default")
        finally:
            incoming.close()
