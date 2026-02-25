import logging

from django.conf import settings
from django.db import connections, OperationalError

logger = logging.getLogger(__name__)


def check_database() -> dict:
    """Verify that the default database connection is alive."""
    try:
        conn = connections["default"]
        conn.ensure_connection()
        return {"status": "ok"}
    except OperationalError as e:
        logger.warning("Health check — DB error: %s", e)
        return {"status": "error", "detail": str(e)}
    except Exception as e:
        logger.warning("Health check — DB unexpected error: %s", e)
        return {"status": "error", "detail": str(e)}


def check_llm_api() -> dict:
    """Check that a LangExtract API key is configured (non-empty, plausible length)."""
    key = getattr(settings, "LANGEXTRACT_API_KEY", "")
    model = getattr(settings, "LLM_MODEL_PRIMARY", "unknown")
    if key and len(key) > 10:
        return {"status": "ok", "model": model}
    return {"status": "error", "detail": "LANGEXTRACT_API_KEY not set or too short"}
