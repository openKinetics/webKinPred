from webKinPred.settings import *  # noqa: F403

DATABASES = {
    "default": {
        "ENGINE": "django.db.backends.postgresql",
        "NAME": "webkinpred_test",
        "USER": "postgres",
        "PASSWORD": "execution-test-only",
        "HOST": "127.0.0.1",
        "PORT": 55439,
    },
}
DATABASE_ROUTERS = []
EXECUTION_V2_ENABLED = False
EXECUTION_HOSTS = {"test": {"token": "test-agent-only", "budgets": {}}}
