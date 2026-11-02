from django.db import migrations, models


class Migration(migrations.Migration):
    dependencies = [("api", "0041_about_job_stats")]
    operations = [
        migrations.AddField(
            model_name="executorhost",
            name="gpu_paused",
            field=models.BooleanField(default=False, db_default=False),
        )
    ]
