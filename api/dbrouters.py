# api/dbrouters.py
class SeqMapRouter:
    def db_for_read(self, model, **hints):
        return "seqmap" if getattr(model, "seqmap_db", False) else None

    def db_for_write(self, model, **hints):
        return None          # read-only

    def allow_migrate(self, db, app_label, model_name=None, **hints):
        return False         # no migrations on any DB