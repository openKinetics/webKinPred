# api/dbrouters.py
class SeqMapRouter:
    def db_for_read(self, model, **hints):
        return "seqmap" if getattr(model, "seqmap_db", False) else None

    def db_for_write(self, model, **hints):
        return None          # read-only

    def allow_migrate(self, db, app_label, model_name=None, **hints):
        """
        Prevent migrations for the 'seqmap' app and prevent any app
        from migrating to the 'seqmap' database.
        """
        if app_label == 'seqmap':
            return False
        if db == 'seqmap':
            return False
        return None