from django.contrib.postgres.operations import AddIndexConcurrently
from django.db import migrations, models


class Migration(migrations.Migration):
    atomic = False
    dependencies = [("api", "0036_background_scheduling")]

    operations = [
        migrations.AddField(
            model_name="executionattempt",
            name="scratch_released_at",
            field=models.DateTimeField(null=True),
        ),
        migrations.AddField(
            model_name="artifactcleanup",
            name="attempt_ids",
            field=models.JSONField(default=list, db_default=[]),
        ),
        migrations.AddField(
            model_name="artifact",
            name="retention",
            field=models.CharField(max_length=16, default="protected", db_default="protected"),
        ),
        AddIndexConcurrently(
            model_name="artifact",
            index=models.Index(
                fields=["retention", "checksum"],
                condition=models.Q(retention__in=["transient", "cache"]),
                name="artifact_collectible_policy",
            ),
        ),
    ]
