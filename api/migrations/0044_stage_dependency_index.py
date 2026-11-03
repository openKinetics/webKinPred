from django.db import migrations, models


def change_index(apps, schema_editor, *, remove=False):
    model = apps.get_model("api", "WorkStage")
    index = models.Index(
        fields=["id"], condition=~models.Q(state="completed"), name="stage_unfinished_id"
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
    dependencies = [("api", "0043_dependency_queue_age")]
    operations = [
        migrations.SeparateDatabaseAndState(
            database_operations=[
                migrations.RunPython(add_index, remove_index, hints={"model_name": "workstage"})
            ],
            state_operations=[
                migrations.AddIndex(
                    model_name="workstage",
                    index=models.Index(
                        fields=["id"],
                        condition=~models.Q(state="completed"),
                        name="stage_unfinished_id",
                    ),
                )
            ],
        )
    ]
