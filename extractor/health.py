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
    """Check that the active primary model's API key is configured."""
    from .extraction import MODEL_PROFILES
    primary_spec  = getattr(settings, "LLM_MODEL_PRIMARY", "")
    primary_model = MODEL_PROFILES.get(primary_spec, primary_spec)

    if primary_model.startswith("gpt-oss"):
        key      = getattr(settings, "CEREBRAS_API_KEY", "")
        provider = "Cerebras"
    else:
        key      = getattr(settings, "LANGEXTRACT_API_KEY", "")
        provider = "Gemini"

    if key and len(key) > 10:
        return {"status": "ok", "model": primary_model, "provider": provider}
    return {"status": "error", "detail": f"{provider} API key not configured", "model": primary_model}
