from django.db import migrations, models

# Each index duplicates another one: Django adds a varchar_pattern_ops copy of every
# indexed character column for LIKE prefix searches, which nothing runs on these
# columns, and location lookups by artifact are served by artifact_host_location.
# On production they held 3.0 GB and were written on every catalogue insert.
INDEXES = {
    "api_artifact_id_40399df2_like": "api_artifact (id varchar_pattern_ops)",
    "api_artifact_checksum_f27f023d_like": "api_artifact (checksum varchar_pattern_ops)",
    "api_artifactdependency_artifact_id_7f8c32d7_like": (
        "api_artifactdependency (artifact_id varchar_pattern_ops)"
    ),
    "api_artifactlocation_artifact_id_f9123442_like": (
        "api_artifactlocation (artifact_id varchar_pattern_ops)"
    ),
    "api_artifactlocation_artifact_id_f9123442": "api_artifactlocation (artifact_id)",
    "api_artifactlocation_host_id_87eb319a_like": (
        "api_artifactlocation (host_id varchar_pattern_ops)"
    ),
}


def drop_indexes(apps, schema_editor):
    if schema_editor.connection.vendor == "postgresql":
        for name in INDEXES:
            schema_editor.execute(f"DROP INDEX CONCURRENTLY IF EXISTS {name}")


def create_indexes(apps, schema_editor):
    if schema_editor.connection.vendor == "postgresql":
        for name, columns in INDEXES.items():
            schema_editor.execute(f"CREATE INDEX CONCURRENTLY IF NOT EXISTS {name} ON {columns}")


class Migration(migrations.Migration):
    atomic = False
    dependencies = [("api", "0044_stage_dependency_index")]
    operations = [
        migrations.SeparateDatabaseAndState(
            database_operations=[migrations.RunPython(drop_indexes, create_indexes)],
            state_operations=[
                migrations.AlterField(
                    model_name="artifactlocation",
                    name="artifact",
                    field=models.ForeignKey(
                        db_index=False,
                        on_delete=models.deletion.PROTECT,
                        related_name="locations",
                        to="api.artifact",
                    ),
                )
            ],
        )
    ]
