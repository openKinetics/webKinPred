from django.db import migrations, models


class Migration(migrations.Migration):
    dependencies = [("api", "0042_executor_gpu_pause")]
    operations = [
        migrations.AddField(
            model_name="workstage",
            name="oldest_consumer_at",
            field=models.DateTimeField(null=True),
        )
    ]
