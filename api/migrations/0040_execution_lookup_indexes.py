from django.contrib.postgres.operations import AddIndexConcurrently
from django.db import migrations, models


class Migration(migrations.Migration):
    atomic = False
    dependencies = [("api", "0039_recovery_batch")]

    operations = [
        AddIndexConcurrently(
            "workstage",
            models.Index(
                fields=["profile", "available_at"],
                condition=models.Q(state="ready"),
                name="stage_ready_profile_time",
            ),
        ),
        AddIndexConcurrently(
            "workstage", models.Index(fields=["completed_at"], name="stage_completed_time")
        ),
        AddIndexConcurrently(
            "executionattempt",
            models.Index(models.F("outcome__category"), name="attempt_outcome_category"),
        ),
        AddIndexConcurrently(
            "executionattempt", models.Index(fields=["finished_at"], name="attempt_finished_time")
        ),
    ]
