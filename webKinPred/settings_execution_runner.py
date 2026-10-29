"""No network databases in isolated model containers; progress is emitted as events."""
from .settings import *  # noqa: F403

DATABASES = {"default": {"ENGINE": "django.db.backends.sqlite3", "NAME": ":memory:"}}
DATABASE_ROUTERS = []
EXECUTION_V2_ENABLED = False
