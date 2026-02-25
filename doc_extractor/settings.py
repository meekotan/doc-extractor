import os
from pathlib import Path
from dotenv import load_dotenv
import dj_database_url

load_dotenv(override=True)

BASE_DIR = Path(__file__).resolve().parent.parent

SECRET_KEY = os.environ.get("DJANGO_SECRET_KEY", "dev-secret-key-change-in-production")

DEBUG = os.environ.get("DEBUG", "True") == "True"

ALLOWED_HOSTS = os.environ.get("ALLOWED_HOSTS", "127.0.0.1,localhost").split(",")


INSTALLED_APPS = [
    "django.contrib.contenttypes",
    "django.contrib.auth",
    "rest_framework",
    "extractor",
]

MIDDLEWARE = [
    "django.middleware.security.SecurityMiddleware",
    "django.middleware.common.CommonMiddleware",
]

TEMPLATES = [
    {
        "BACKEND": "django.template.backends.django.DjangoTemplates",
        "DIRS": [],
        "APP_DIRS": True,
        "OPTIONS": {
            "context_processors": [
                "django.template.context_processors.request",
            ],
        },
    },
]

ROOT_URLCONF = "doc_extractor.urls"

# ---------------------------------------------------------------------------
# Database
# Set DATABASE_URL in .env to use PostgreSQL:
#   DATABASE_URL=postgresql://doc_user:yourpassword@localhost:5432/doc_extractor
# Without DATABASE_URL, falls back to SQLite (dev/single-worker only).
# ---------------------------------------------------------------------------
_DATABASE_URL = os.environ.get("DATABASE_URL", "")
if _DATABASE_URL:
    DATABASES = {"default": dj_database_url.config(default=_DATABASE_URL, conn_max_age=600)}
else:
    DATABASES = {
        "default": {
            "ENGINE": "django.db.backends.sqlite3",
            "NAME": BASE_DIR / "db.sqlite3",
        }
    }

# LangExtract / LLM configuration
LANGEXTRACT_API_KEY = os.environ.get("LANGEXTRACT_API_KEY", "")   # Gemini / default
FIREWORKS_API_KEY   = os.environ.get("FIREWORKS_API_KEY",   "")   # Fireworks AI
LLM_MODEL_PRIMARY  = os.environ.get("LLM_MODEL_PRIMARY",  "gemini-2.0-flash")
LLM_MODEL_FALLBACK = os.environ.get("LLM_MODEL_FALLBACK", "gemini-2.0-flash")

# Currency database — JSON string loaded from env (matches CURRENCY_DB_JSON in original workflow)
CURRENCY_DB_JSON = os.environ.get("CURRENCY_DB_JSON", "[]")

DEFAULT_AUTO_FIELD = "django.db.models.BigAutoField"

TIME_ZONE = "UTC"
USE_TZ = True

LOGGING = {
    "version": 1,
    "disable_existing_loggers": False,
    "handlers": {
        "console": {
            "class": "logging.StreamHandler",
        },
    },
    "root": {
        "handlers": ["console"],
        "level": "WARNING",
    },
    "loggers": {
        "extractor": {
            "handlers": ["console"],
            "level": "DEBUG",
            "propagate": False,
        },
    },
}

REST_FRAMEWORK = {
    "DEFAULT_RENDERER_CLASSES": [
        "rest_framework.renderers.JSONRenderer",
        "rest_framework.renderers.BrowsableAPIRenderer",
    ],
}


