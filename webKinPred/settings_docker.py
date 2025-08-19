"""
Docker-specific Django settings for webKinPred project.
"""

from .settings import *
import os

# Override settings for Docker environment
DEBUG = os.getenv('DEBUG', 'False').lower() == 'true'

# WhiteNoise: serve collected static from Gunicorn in production
# Insert after SecurityMiddleware
MIDDLEWARE.insert(1, 'whitenoise.middleware.WhiteNoiseMiddleware')

STATIC_URL = '/django_static/'
STATIC_ROOT = '/app/staticfiles'
STATICFILES_STORAGE = 'whitenoise.storage.CompressedManifestStaticFilesStorage'
WHITENOISE_MAX_AGE = 31536000

# Docker-specific allowed hosts
ALLOWED_HOSTS = [
    "kineticxpredictor.humanmetabolism.org", 
    "www.kineticxpredictor.humanmetabolism.org",  # Add www subdomain
    "127.0.0.1", 
    "localhost",
    "backend",  # Docker service name
    "0.0.0.0",
    "webkinpred-backend-1",  # Docker compose service name
]

# Update CORS settings for Docker
CORS_ALLOWED_ORIGINS = [
    "http://localhost:5173",
    "http://127.0.0.1:5173",
    "http://localhost:3000",
    "http://127.0.0.1:3000",
    "https://kineticxpredictor.humanmetabolism.org"
]

CSRF_TRUSTED_ORIGINS = [
    "http://localhost:5173",
    "http://127.0.0.1:5173",
    "http://localhost:3000", 
    "http://127.0.0.1:3000",
    "https://kineticxpredictor.humanmetabolism.org",
]

# Redis configuration for Docker
REDIS_HOST = os.getenv('REDIS_HOST', 'redis')
REDIS_PORT = os.getenv('REDIS_PORT', '6379')

CELERY_BROKER_URL = f'redis://{REDIS_HOST}:{REDIS_PORT}/0'
CELERY_RESULT_BACKEND = f'redis://{REDIS_HOST}:{REDIS_PORT}/0'

CACHES = {
    "default": {
        "BACKEND": "django_redis.cache.RedisCache",
        "LOCATION": f"redis://{REDIS_HOST}:{REDIS_PORT}/1",
        "OPTIONS": {
            "CLIENT_CLASS": "django_redis.client.DefaultClient",
        },
        "TIMEOUT": None, 
    }
}

LOGGING_REDIS_URL = f"redis://{REDIS_HOST}:{REDIS_PORT}/2"

# Database configuration
DATABASES = {
    'default': {
        'ENGINE': 'django.db.backends.sqlite3',
        'NAME': '/app/db.sqlite3',  # Use absolute path
    },
    'seqmap': {
        'ENGINE': 'django.db.backends.sqlite3',
        'NAME': '/app/media/sequence_info/seqmap.sqlite3',  # Use absolute path
    }
}

# Security settings for production
if not DEBUG:
    SECURE_BROWSER_XSS_FILTER = True
    SECURE_CONTENT_TYPE_NOSNIFF = True
    X_FRAME_OPTIONS = 'DENY'
    SECURE_HSTS_SECONDS = 31536000
    SECURE_HSTS_INCLUDE_SUBDOMAINS = True
    SECURE_HSTS_PRELOAD = True

# Celery task routing
CELERY_TASK_ROUTES = {
    'api.tasks.*': {'queue': 'webkinpred'},
}

# Ensure all tasks use the webkinpred queue
CELERY_TASK_DEFAULT_QUEUE = 'webkinpred'
CELERY_TASK_DEFAULT_EXCHANGE = 'webkinpred'
CELERY_TASK_DEFAULT_ROUTING_KEY = 'webkinpred'
