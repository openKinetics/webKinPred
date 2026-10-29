from django.db import migrations


def install(apps, schema_editor):
    if schema_editor.connection.vendor != "postgresql":
        return
    schema_editor.execute("""
        CREATE FUNCTION execution_specification_immutable() RETURNS trigger
        LANGUAGE plpgsql AS $$
        BEGIN
          IF NEW.specification IS DISTINCT FROM OLD.specification
            OR NEW.checksum IS DISTINCT FROM OLD.checksum
            OR NEW.input_checksum IS DISTINCT FROM OLD.input_checksum
            OR NEW.job_id IS DISTINCT FROM OLD.job_id
            OR NEW.policy_version IS DISTINCT FROM OLD.policy_version
            OR NEW.created_at IS DISTINCT FROM OLD.created_at THEN
              RAISE EXCEPTION 'Execution specification is immutable' USING ERRCODE='23514';
          END IF;
          RETURN NEW;
        END;
        $$;
        CREATE TRIGGER execution_specification_immutable
          BEFORE UPDATE ON api_executionspec FOR EACH ROW
          EXECUTE FUNCTION execution_specification_immutable();
    """)


def remove(apps, schema_editor):
    if schema_editor.connection.vendor == "postgresql":
        schema_editor.execute(
            "DROP TRIGGER IF EXISTS execution_specification_immutable ON api_executionspec"
        )
        schema_editor.execute("DROP FUNCTION IF EXISTS execution_specification_immutable()")


class Migration(migrations.Migration):
    dependencies = [("api", "0028_execution_attempt_stop_and_sequence_ids")]
    operations = [migrations.RunPython(install, remove)]
