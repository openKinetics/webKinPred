# api/dbrouters.py
class SeqMapRouter:
    def db_for_read(self, model, **hints):
        if model._meta.label_lower == "api.sequenceidentity":
            return "seqmap"
        return "seqmap" if getattr(model, "seqmap_db", False) else None

    def db_for_write(self, model, **hints):
        if model._meta.label_lower == "api.sequenceidentity":
            return "seqmap"
        return None  # read-only

    def allow_migrate(self, db, app_label, model_name=None, **hints):
        """
        Prevent migrations for the 'seqmap' app and prevent any app
        from migrating to the 'seqmap' database.
        """
        if app_label == "api" and model_name == "sequenceidentity":
            return db == "seqmap"
        if app_label == "seqmap":
            return False
        if db == "seqmap":
            return False
        return None


# Model names (lowercased) that live in the dedicated ReconXKG memoization
# database. They are regular models in the ``api`` app but are routed away from
# the default DB so cache traffic never contends with application tables.
PREDICTION_STORE_MODELS = {"predictionstore", "similaritystore"}


class PredictionStoreRouter:
    """Route the ReconXKG cache models to the ``prediction_store`` database."""

    def _is_store_model(self, model) -> bool:
        return getattr(model, "prediction_store_db", False)

    def db_for_read(self, model, **hints):
        return "prediction_store" if self._is_store_model(model) else None

    def db_for_write(self, model, **hints):
        return "prediction_store" if self._is_store_model(model) else None

    def allow_relation(self, obj1, obj2, **hints):
        # Cache rows are self-contained; only relate store rows to store rows.
        store1 = getattr(obj1, "prediction_store_db", False)
        store2 = getattr(obj2, "prediction_store_db", False)
        if store1 or store2:
            return store1 and store2
        return None

    def allow_migrate(self, db, app_label, model_name=None, **hints):
        is_store = (model_name or "").lower() in PREDICTION_STORE_MODELS
        if db == "prediction_store":
            return is_store
        if is_store:
            return False
        return None
