from django.db import migrations, models

# Each cleanup claim selects a host's finished attempts whose scratch is not yet released.
# On 21 November only 7 of 206,490 attempts qualified, but the query read every attempt
# of the host (155,000 on the GPU host, about 4 times a minute).


def change_index(apps, schema_editor, *, remove=False):
    model = apps.get_model("api", "ExecutionAttempt")
    index = models.Index(
        fields=["host", "started_at"],
        condition=models.Q(scratch_released_at__isnull=True),
        name="attempt_unreleased_scratch",
    )
    method = schema_editor.remove_index if remove else schema_editor.add_index
    options = {"concurrently": True} if schema_editor.connection.vendor == "postgresql" else {}
    method(model, index, **options)


def add_index(apps, schema_editor):
    change_index(apps, schema_editor)


def remove_index(apps, schema_editor):
    change_index(apps, schema_editor, remove=True)


class Migration(migrations.Migration):
    atomic = False
    dependencies = [("api", "0046_outbox_unpublished_index")]
    operations = [
        migrations.SeparateDatabaseAndState(
            database_operations=[
                migrations.RunPython(
                    add_index, remove_index, hints={"model_name": "executionattempt"}
                )
            ],
            state_operations=[
                migrations.AddIndex(
                    model_name="executionattempt",
                    index=models.Index(
                        fields=["host", "started_at"],
                        condition=models.Q(scratch_released_at__isnull=True),
                        name="attempt_unreleased_scratch",
                    ),
                )
            ],
        )
    ]
