# doc_extractor

Django REST API for structured invoice data extraction.
Translates the `document_extractor_optimized.yml` Dify workflow (branch `document_code == 04021`) into a standalone Python service powered by [LangExtract](https://github.com/google/langextract).

---

## Features

- **5-step extraction pipeline**: clean text → LangExtract LLM → validate/repair JSON → fallback model → finalize
- **Per-request model selection**: pass `"model": "fast"` in the POST body to switch models on the fly
- **Bilingual invoice support**: prefers Russian descriptions and longer HS codes from RU section
- **Smart header injection**: document metadata + column headers passed as `additional_context` to every LangExtract chunk — no item-row leakage, no duplicate extractions
- **Full job history**: every request stored in DB with timing metrics, model used, and LangExtract visualizer data
- **Zero queues**: fully synchronous, no Celery/Redis required

---

## Quick Start

### 1. Clone and set up environment

```bash
git clone <repo-url>
cd doc_extractor

python -m venv venv
# Windows:
venv\Scripts\activate
# Linux/macOS:
source venv/bin/activate

pip install -r requirements.txt
```

### 2. Configure environment variables

Copy `.env.example` to `.env` and fill in the values:

```bash
cp .env.example .env
```

```dotenv
# .env

# Primary: GPT-OSS 120B via Cerebras (~3 000 tok/s, 131K context)
LLM_MODEL_PRIMARY=cerebras
LLM_MODEL_FALLBACK=gemini

# Cerebras cloud — get key at cerebras.ai/openai
CEREBRAS_BASE_URL=https://api.cerebras.ai/v1
CEREBRAS_API_KEY=your_cerebras_api_key_here
LLM_MAX_WORKERS_CEREBRAS=1
LLM_MAX_CHAR_BUFFER_CEREBRAS=100000

# Ollama cloud — get key at ollama.com/settings/keys
OLLAMA_BASE_URL=https://ollama.com/v1
OLLAMA_API_KEY=your_ollama_api_key_here
LLM_MAX_WORKERS_OLLAMA=1
LLM_MAX_CHAR_BUFFER_OLLAMA=30000

# Gemini 2.5 Flash — fallback — get key at aistudio.google.com/apikey
LANGEXTRACT_API_KEY=your_gemini_api_key_here
LLM_MAX_WORKERS_GEMINI=10
LLM_MAX_CHAR_BUFFER=4000

CURRENCY_DB_JSON=[{"code":"EUR","name":"Euro"},{"code":"USD","name":"US Dollar"}]

# Optional — omit to use SQLite (development only)
DATABASE_URL=postgresql://user:pass@localhost:5432/doc_extractor

DJANGO_SECRET_KEY=change-me-in-production
DEBUG=True
```

### 3. Run migrations and start

```bash
python manage.py migrate
python manage.py runserver
```

API is available at `http://127.0.0.1:8000/api/`.

---

## API Reference

### `POST /api/extract/`

Run the extraction pipeline on an invoice OCR text.

**Request body:**

```json
{
  "document_code": "04021",
  "ocr_draft": "<raw OCR text of the invoice>",
  "model": "fast"
}
```

| Field | Type | Required | Description |
|---|---|---|---|
| `document_code` | string | ✅ | Must be `"04021"` |
| `ocr_draft` | string | ✅ | Raw OCR text from the invoice |
| `model` | string | ❌ | Model profile or raw model ID (see [Model Selection](#model-selection)) |

**Response `200 OK`:**

```json
{
  "id": 42,
  "document_code": "04021",
  "model_id": "gemini-2.0-flash",
  "status": "success",
  "result": {
    "items": [
      {
        "description": "Труба карбоновая диаметр 16мм",
        "hs_code": "7304390009",
        "quantity": 10.0,
        "unit": "pcs",
        "cost": 15.5,
        "price": 155.0,
        "currency_code": "EUR",
        "currency_name": "Euro",
        "document_date": "15/01/2025",
        "document_number": "INV-206447",
        "country_origin": "DE",
        "country_sender": "Germany"
      }
    ],
    "count": 1
  },
  "error": "",
  "duration_ms": 8430,
  "metrics": {
    "t_clean_s": 0.001,
    "t_primary_llm_s": 8.2,
    "t_validate_s": 0.003,
    "t_finalize_s": 0.012,
    "t_total_s": 8.431,
    "items_extracted": 1,
    "fallback_used": false,
    "primary_valid": true
  },
  "created_at": "2025-02-25T10:00:00Z",
  "updated_at": "2025-02-25T10:00:08Z"
}
```

---

### `GET /api/jobs/<id>/`

Retrieve a stored extraction job by ID.

```bash
curl http://127.0.0.1:8000/api/jobs/42/
```

---

### `GET /api/jobs/<id>/visualize/`

Returns a standalone interactive HTML page powered by the LangExtract visualizer — shows exactly which text spans were matched for each extracted item.

```bash
# Open in browser:
open http://127.0.0.1:8000/api/jobs/42/visualize/
```

---

### `GET /api/metrics/`

Aggregate stats across all stored jobs.

```json
{
  "total_jobs": 50,
  "success_count": 47,
  "failed_count": 3,
  "success_rate": 0.94,
  "fallback_used_count": 8,
  "fallback_rate": 0.16,
  "avg_duration_ms": 9200.5,
  "avg_items_extracted": 24.3,
  "avg_step_times_s": {
    "t_primary_llm_s": 8.1,
    "t_validate_s": 0.004,
    "t_total_s": 8.9
  },
  "avg_field_fill_rates": {
    "hs_code": 0.87,
    "cost": 0.95,
    "country_origin": 0.78
  }
}
```

---

### `GET /api/health/`

Liveness check — returns `200` when DB and LLM API are reachable, `503` otherwise.

```json
{
  "status": "ok",
  "database": {"status": "ok"},
  "llm_api": {"status": "ok"}
}
```

---

## Model Selection

Pass an optional `"model"` field in the POST body to override the default primary model.

### Built-in profiles

| `"model"` value | Resolves to | Notes |
|---|---|---|
| _(omitted)_ | `LLM_MODEL_PRIMARY` from `.env` | Default |
| `"cerebras"` | `gpt-oss-120b` | GPT-OSS 120B via Cerebras (~3 000 tok/s) |
| `"gpt-oss"` | `gpt-oss:120b` | GPT-OSS 120B via Ollama cloud |
| `"gemini"` | `gemini-2.5-flash` | Google Gemini 2.5 Flash |
| any raw model ID | used verbatim | e.g. `"gemini-2.5-flash"` |

### Fallback model

The `"model"` override applies to the **primary call only**.
If the primary fails validation, the service automatically retries with `LLM_MODEL_FALLBACK` from `.env`.

### Adding a new profile

Edit `MODEL_PROFILES` in `extractor/extraction.py` — one line, no migrations:

```python
MODEL_PROFILES["my-model"] = "actual-model-id"
```

---

## Pipeline Architecture

```
POST /api/extract/
        │
        ▼
1. clean_text()
   Strip \r, collapse whitespace, prepend CURRENCY DATABASE block
        │
        ▼
2. extract_header()
   Collect metadata + table column headers (stops before first item row)
   → passed as additional_context to every LangExtract chunk
        │
        ▼
3. extract_with_langextract_optimized()   ← primary model
   LangExtract splits text into chunks, calls LLM in parallel (max_workers=15)
        │
        ▼
4. validate_and_parse()
   JSON repair → flatten chunks → type-coerce numeric fields
   is_valid = 0? → retry with fallback model
        │
        ▼
5. post_fill_from_header() + finalize_items()
   Fill empty metadata from regex-parsed header
   Cost/unit fallbacks → currency resolution
        │
        ▼
   ExtractionJob saved → HTTP 200 with full result
```

---

## Environment Variables

| Variable | Required | Default | Description |
|---|---|---|---|
| `LLM_MODEL_PRIMARY` | ❌ | `gpt-oss` | Default primary model (alias or raw ID) |
| `LLM_MODEL_FALLBACK` | ❌ | `gemini-2.5-flash` | Fallback on validation failure |
| `CEREBRAS_API_KEY` | ✅* | — | Cerebras cloud key (if using `cerebras` alias) |
| `CEREBRAS_BASE_URL` | ❌ | `https://api.cerebras.ai/v1` | Cerebras endpoint |
| `LLM_MAX_WORKERS_CEREBRAS` | ❌ | `1` | Parallel chunks for Cerebras |
| `LLM_MAX_CHAR_BUFFER_CEREBRAS` | ❌ | `100000` | Chunk size for Cerebras (fits whole invoice) |
| `OLLAMA_API_KEY` | ✅* | — | Ollama cloud key (if using `gpt-oss` alias) |
| `OLLAMA_BASE_URL` | ❌ | `https://ollama.com/v1` | Ollama cloud endpoint |
| `LLM_MAX_WORKERS_OLLAMA` | ❌ | `1` | Parallel chunks for Ollama |
| `LLM_MAX_CHAR_BUFFER_OLLAMA` | ❌ | `30000` | Chunk size for Ollama |
| `LANGEXTRACT_API_KEY` | ✅* | — | Gemini API key (if using `gemini` alias) |
| `LLM_MAX_WORKERS_GEMINI` | ❌ | `10` | Parallel chunks for Gemini |
| `LLM_MAX_CHAR_BUFFER` | ❌ | `4000` | Chunk size for Gemini |
| `CURRENCY_DB_JSON` | ❌ | `[]` | JSON array of `{"code","name"}` pairs |
| `DATABASE_URL` | ❌ | SQLite | PostgreSQL URL for production |
| `DJANGO_SECRET_KEY` | ❌ | dev key | Change in production |
| `DEBUG` | ❌ | `True` | Set `False` in production |

*Required for the provider you are actively using.

---

## Development

### Run tests / shell

```bash
# Django shell — replay any stored job
python manage.py shell
>>> from extractor.extraction import run_invoice_extraction
>>> from extractor.models import ExtractionJob
>>> job = ExtractionJob.objects.get(pk=42)
>>> result = run_invoice_extraction(job.ocr_draft, model_id="gemini")
>>> print(result["metrics"])
```

### Production (gunicorn)

```bash
pip install gunicorn
gunicorn doc_extractor.wsgi:application --workers 2 --timeout 300
```

Use `--timeout 300` — large invoice files can take 30–120 s per request.

---

## Project Structure

```
doc_extractor/
├── doc_extractor/
│   ├── settings.py        # Django settings — reads all config from .env
│   └── urls.py            # Root URL conf → /api/
├── extractor/
│   ├── extraction.py      # Full 5-step pipeline + LangExtract integration
│   ├── models.py          # ExtractionJob model
│   ├── views.py           # DRF views for all endpoints
│   ├── serializers.py     # Request/response serializers
│   ├── currency.py        # Currency DB loading and resolution
│   ├── metrics.py         # RunMetrics dataclass + timer helpers
│   ├── health.py          # DB + LLM API health checks
│   ├── visualizer.py      # LangExtract HTML visualizer builder
│   └── migrations/        # DB migrations
├── manage.py
├── requirements.txt
├── skill.md               # Developer reference guide
└── README.md
```
