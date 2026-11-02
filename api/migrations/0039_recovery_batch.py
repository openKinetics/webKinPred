import django.db.models.deletion
from django.db import migrations, models


class Migration(migrations.Migration):
    dependencies = [("api", "0038_artifact_sweep")]

    operations = [
        migrations.CreateModel(
            name="RecoveryBatch",
            fields=[
                ("id", models.CharField(max_length=64, primary_key=True, serialize=False)),
                ("generation", models.PositiveIntegerField(default=0)),
                (
                    "source",
                    models.ForeignKey(
                        on_delete=django.db.models.deletion.PROTECT,
                        related_name="recovery_batches",
                        to="api.workstage",
                    ),
                ),
                (
                    "stage",
                    models.OneToOneField(
                        null=True, on_delete=django.db.models.deletion.PROTECT, to="api.workstage"
                    ),
                ),
            ],
        ),
    ]
