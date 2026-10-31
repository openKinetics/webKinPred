import django.db.models.deletion
from django.db import migrations, models


class Migration(migrations.Migration):
    dependencies = [("api", "0037_artifact_retention")]
    operations = [
        migrations.CreateModel(
            name="ArtifactSweep",
            fields=[
                (
                    "host",
                    models.OneToOneField(
                        on_delete=django.db.models.deletion.PROTECT,
                        primary_key=True,
                        serialize=False,
                        to="api.executorhost",
                    ),
                ),
                ("cursor", models.PositiveBigIntegerField(default=0)),
                ("pending", models.JSONField(default=list)),
            ],
        ),
    ]
