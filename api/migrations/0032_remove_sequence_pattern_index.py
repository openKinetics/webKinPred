from django.db import migrations


def remove_sequence_index(apps, schema_editor):
    connection = schema_editor.connection
    if connection.vendor != "postgresql":
        return
    # Older PostgreSQL/Django migrations left the auxiliary LIKE index behind
    # when removing the TextField's uniqueness. It has the same long-key limit.
    with connection.cursor() as cursor:
        constraints = connection.introspection.get_constraints(cursor, "sequences")
    for name, definition in constraints.items():
        if definition["columns"] == ["seq"] and definition["index"] and not definition["unique"]:
            schema_editor.execute("DROP INDEX " + schema_editor.quote_name(name))


class Migration(migrations.Migration):
    dependencies = [("api", "0031_sequence_hash_uniqueness")]
    operations = [
        migrations.RunPython(
            remove_sequence_index,
            migrations.RunPython.noop,
            hints={"model_name": "sequenceidentity"},
        )
    ]
