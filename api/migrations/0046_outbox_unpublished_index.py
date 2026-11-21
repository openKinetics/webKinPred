from django.db import migrations, models

# The progress projector asks for the oldest unpublished event several times a second.
# With no index on published_at, each poll scanned the whole outbox: on 21 November,
# 227,357 rows (47 MB), almost all of them published, about 5.5 times a second.


def change_index(apps, schema_editor, *, remove=False):
    model = apps.get_model("api", "ExecutionOutbox")
    index = models.Index(
        fields=["id"],
        condition=models.Q(published_at__isnull=True),
        name="outbox_unpublished_id",
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
    dependencies = [("api", "0045_drop_duplicate_artifact_indexes")]
    operations = [
        migrations.SeparateDatabaseAndState(
            database_operations=[
                migrations.RunPython(
                    add_index, remove_index, hints={"model_name": "executionoutbox"}
                )
            ],
            state_operations=[
                migrations.AddIndex(
                    model_name="executionoutbox",
                    index=models.Index(
                        fields=["id"],
                        condition=models.Q(published_at__isnull=True),
                        name="outbox_unpublished_id",
                    ),
                )
            ],
        )
    ]
