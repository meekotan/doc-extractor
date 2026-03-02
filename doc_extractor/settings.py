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
LANGEXTRACT_API_KEY = os.environ.get("LANGEXTRACT_API_KEY", "")   # Gemini API key
LLM_MODEL_PRIMARY  = os.environ.get("LLM_MODEL_PRIMARY",  "gpt-oss-120b")
LLM_MODEL_FALLBACK = os.environ.get("LLM_MODEL_FALLBACK", "gemini-2.5-flash")

# Cerebras cloud — GPT-OSS 120B (alias: "cerebras", raw: "gpt-oss-120b")
# ~3000 tok/s, 131K context window. Get key at: cerebras.ai/openai
CEREBRAS_BASE_URL            = os.environ.get("CEREBRAS_BASE_URL",            "https://api.cerebras.ai/v1")
CEREBRAS_API_KEY             = os.environ.get("CEREBRAS_API_KEY",             "")
LLM_MAX_WORKERS_CEREBRAS     = int(os.environ.get("LLM_MAX_WORKERS_CEREBRAS",     "20"))
LLM_MAX_CHAR_BUFFER_CEREBRAS = int(os.environ.get("LLM_MAX_CHAR_BUFFER_CEREBRAS", "5000"))

# Gemini performance tuning (fallback model)
# max_char_buffer: chars per chunk — smaller keeps Gemini within rate limits
# max_workers_gemini: parallel chunk workers
LLM_MAX_CHAR_BUFFER    = int(os.environ.get("LLM_MAX_CHAR_BUFFER",    "4000"))
LLM_MAX_WORKERS_GEMINI = int(os.environ.get("LLM_MAX_WORKERS_GEMINI", "10"))

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


