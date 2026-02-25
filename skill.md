# skill.md — Doc Extractor Project

Reference guide for future work. Grounded in the actual codebase.

---

## 1. Django REST Patterns

### Synchronous POST that saves history
```python
# views.py pattern — create job, run inline, return full result
job = ExtractionJob.objects.create(status=STATUS_PROCESSING, ...)
try:
    output = run_invoice_extraction(job.ocr_draft)
except Exception as exc:
    job.status = STATUS_FAILED; job.error = str(exc)
    job.save(update_fields=["status", "error", "updated_at"])
    return Response(ExtractionJobSerializer(job).data, status=500)
# save result and return 200
```

### Always use `update_fields` on save
```python
# Faster than full save — only touches the columns that changed
job.save(update_fields=["status", "result", "metrics", "duration_ms", "viz_data", "updated_at"])
```

### Return the serialized job from POST
- POST returns `ExtractionJobSerializer(job).data` — same shape as GET `/api/jobs/<pk>/`
- Client gets `job_id`, `status`, `result`, `metrics`, `duration_ms` in one response
- No polling needed

---

## 2. LangExtract Core Usage

### Minimal working call
```python
import langextract as lx
from langextract import factory as lx_factory

config = lx_factory.ModelConfig(
    model_id="gemini-2.0-flash",
    provider_kwargs={"api_key": API_KEY, "max_workers": 4},
)
annotated_doc = lx.extract(
    text_or_documents=text,
    prompt_description=PROMPT,
    examples=EXAMPLES,
    config=config,
    additional_context=header_text,   # injected into EVERY chunk
    max_char_buffer=15000,            # chunk size in chars
)
```

### Reading results
```python
for extraction in annotated_doc.extractions:
    item = {"description": extraction.extraction_text}
    if extraction.attributes:
        item.update(extraction.attributes)  # all structured fields
```

### Serializing AnnotatedDocument for storage
```python
from langextract import data_lib
viz_dict = data_lib.annotated_document_to_dict(annotated_doc)   # → store in JSONField
# later:
annotated_doc = data_lib.annotated_document_from_dict(viz_dict) # → restore for visualizer
```

### Registering custom OpenAI-compatible providers
```python
# Run once at import time — idempotent
import langextract.providers as lx_providers
from langextract.providers import router as lx_router
from langextract.providers.openai import OpenAILanguageModel

lx_providers.load_builtins_once()
lx_router.register(r"^accounts\/fireworks\/", priority=20)(OpenAILanguageModel)
# Then ModelConfig with provider_kwargs={"base_url": "https://api.fireworks.ai/inference/v1"}
```

---

## 3. Prompt Engineering

### Effective prompt structure for invoice extraction
```
<role>        → define the exact persona + mission
<language_logic>   → bilingual priority rules (RU over EN)
<hs_code_rules>    → verbatim copy rules, prefer longer code
<field_specifications> → per-field rules with defaults
<orchestration_data>   → how to propagate header fields to all items
<output_format>    → JSON only, no markdown, start with [, end with ]
```

### Key principles
- **Zero information loss** — full technical description, no summarizing
- **Verbatim HS codes** — copy every digit, prefer longer version from RU section
- **Units normalization** — standardize in the prompt: `kg, pcs, l, set, m`
- **Header in every chunk** — use `additional_context` so metadata fields populate even in later chunks
- **Output format in prompt** — explicitly ban markdown fences, force pure JSON array

### Few-shot example must show
1. Bilingual input (RU section + EN section)
2. Correct field from the right section (RU description, EN country code)
3. Longer HS code preference (10-digit over 8-digit)
4. All field types: string, float, ISO code, date

---

## 4. Data Engineering

### Text cleaning pipeline (`clean_text`)
1. Strip `\r`, collapse `\n{3,}` → `\n\n`
2. Per-line: collapse multiple spaces/tabs → single space, strip
3. Prepend currency DB block so it's visible at chunk 0:
```
=== CURRENCY DATABASE (REFERENCE) ===
[{"code":"EUR","name":"Euro"}, ...]

=== INVOICE CONTENT ===
<cleaned text>
```

### Header extraction (`extract_header`)
- Take first 40 lines → filter non-empty → keep first 25
- Wrap with label: `=== DOCUMENT HEADER (applies to ALL items) ===`
- Pass as `additional_context` to `lx.extract()` — this is the most important optimization for metadata consistency across chunks

### Header regex fallback (`parse_header_metadata`)
- Patterns for: `document_number`, `document_date`, `country_sender`
- Run after LLM extraction as a safety net — fills empty metadata fields

### JSON repair (`_repair_json` + `validate_and_parse`)
Priority order:
1. Direct `json.loads()`
2. Strip markdown fences, isolate `[...]` with regex
3. Fix trailing commas, Python booleans/None → JSON
4. If still broken: find last `}`, truncate there, close with `]` (handles unterminated strings from cutoff)
5. Flatten list-of-lists (chunked output concatenation)
6. Drop items with no `description` field

### Numeric field coercion (Gemini Free Tier returns strings)
```python
val = str(item.get(field, "0")).replace(",", ".")
item[field] = float(re.sub(r"[^\d.]", "", val) or 0)
```

### Deduplication (`deduplicate_items`)
- Key: `(round(price,2), round(quantity,3), round(cost,2))`
- Skip rows where all three are zero/missing (likely header lines)
- Keep the row with most filled fields (prefers RU description with hs_code)
- **Do not use price+quantity only** — false positives (two different items same price×qty)

### Post-fill from header (`post_fill_from_header`)
- After LLM extraction: fill empty `document_date`, `document_number`, `country_sender`, `currency_code/name` from regex-parsed header
- "empty" = `None`, `""`, `"null"`, `"none"`, `"0"`

### Cost fallback
```python
if is_empty_cost:
    item["cost"] = round(price / qty, 4) if qty else price
```

### Unit default
```python
if unit is None or str(unit).strip().lower() in ("", "null", "none"):
    item["unit"] = "pcs"
```

---

## 5. Orchestration (Pipeline)

### 5-step pipeline (`run_invoice_extraction`)
```
clean_text()
    ↓
extract_header() + parse_header_metadata()
    ↓
extract_with_langextract_optimized()   ← primary model (LLM_MODEL_PRIMARY)
    ↓
validate_and_parse()
    ↓ [if is_valid == 0]
extract_with_langextract_optimized()   ← fallback model (LLM_MODEL_FALLBACK)
validate_and_parse()
    ↓
post_fill_from_header()
cost/unit fallback
finalize_items()  ← currency + country resolution
deduplicate_items()
```

### Metrics wrapping
```python
m = RunMetrics()
t_wall_start = time.perf_counter()
with timer() as t:
    ...do step...
m.t_primary_llm_s = t[0]
# ...
m.t_total_s = round(time.perf_counter() - t_wall_start, 3)
return {"result": ..., "metrics": m.to_dict(), "annotated_doc": ...}
```

### Model config via Django settings
```python
primary_model  = getattr(settings, "LLM_MODEL_PRIMARY",  "gemini-2.5-pro")
fallback_model = getattr(settings, "LLM_MODEL_FALLBACK", "gemini-2.0-flash")
```
Override in `.env` — no code change needed.

---

## 6. Speed: Fastest POST on Large Files

### The biggest knob: `max_char_buffer`
```python
# extraction.py → extract_with_langextract_optimized()
annotated_doc = lx.extract(..., max_char_buffer=2000)
```
| Value | Chunks (20k chars) | Effect |
|-------|--------------------|--------|
| 4000  | ~5 chunks          | More API calls, slower, more accurate per chunk |
| 8000  | ~3 chunks          | Balanced |
| **15000** | **~2 chunks** | **Current — fast for large files** |
| 20000+ | 1 chunk           | Fastest but risks exceeding context window |

**Rule:** Raise `max_char_buffer` until you hit model context limit or quality drops.

### `max_workers` parallelism
```python
provider_kwargs={"api_key": ..., "max_workers": 15}
```
LangExtract sends chunks to the LLM in parallel. `max_workers=4` means 4 concurrent API calls. Raise to `8` if the LLM provider allows it and you have many chunks.

### Choose the right primary model
| Model | Speed | Quality |
|-------|-------|---------|
| `gemini-2.0-flash` | fast | good for clean OCR |
| `gemini-2.5-pro`   | slower | best for noisy/complex invoices |
| `accounts/fireworks/models/glm-5` | very fast | experimental |

**For large clean files:** use `gemini-2.0-flash` as primary, `gemini-2.5-pro` as fallback only.

### Skip fallback for speed
- Fallback doubles processing time if primary fails
- If accuracy is acceptable, raise validation threshold or accept partial results
- Monitor `metrics.fallback_used` via `GET /api/metrics/` to see how often fallback triggers

### Avoid re-extraction — post-processing is cheap
Steps 4–5 (post_fill, cost fallback, finalize, dedup) take milliseconds.
The LLM call is 95%+ of total time. Optimize there first.

### Reduce OCR noise before sending
- Remove page headers/footers (repeated every page = wasted tokens)
- Strip page numbers, watermarks
- Clean OCR artifacts before posting to `/api/extract/`
- Fewer tokens → smaller chunks → faster extraction

### Timeout note
- Django dev server has no default request timeout
- For large files, expect 30–120s per request
- Use `gunicorn --timeout 300` in production instead of `runserver`

---

## 7. Model / Environment Wiring

### `.env` keys
```
LANGEXTRACT_API_KEY=...    # Gemini / OpenAI key
FIREWORKS_API_KEY=...      # Fireworks AI (optional)
LLM_MODEL_PRIMARY=gemini-2.0-flash
LLM_MODEL_FALLBACK=gemini-2.5-pro
CURRENCY_DB_JSON=[{"code":"EUR","name":"Euro"}, ...]
DATABASE_URL=postgresql://...   # omit → SQLite
```

### Adding a new model
1. Add prefix to `_OPENAI_COMPATIBLE` in `extraction.py` if it's OpenAI-compatible
2. Set `LLM_MODEL_PRIMARY` in `.env`
3. No other code change needed — `_build_lx_config()` handles routing

---

## 9. Model Version Management

### Per-request model override (POST body)
```json
{
  "document_code": "04021",
  "ocr_draft": "...",
  "model": "fast"
}
```
The `model` field is **optional** (omit → uses `LLM_MODEL_PRIMARY` from settings).

### Accepted values
| `model` value | Resolves to |
|---------------|-------------|
| _(omitted / "")_ | `LLM_MODEL_PRIMARY` from `.env` |
| `"fast"` | `gemini-2.0-flash` |
| `"quality"` | `gemini-2.5-pro` |
| `"glm"` | `accounts/fireworks/models/glm-5` |
| any raw model ID | used verbatim |

### Resolution chain (`extraction.py`)
```python
MODEL_PROFILES: dict[str, str] = {
    "fast":    "gemini-2.0-flash",
    "quality": "gemini-2.5-pro",
    "glm":     "accounts/fireworks/models/glm-5",
}

def resolve_model_id(model_spec: str | None) -> str:
    if not model_spec:
        return getattr(settings, "LLM_MODEL_PRIMARY", "gemini-2.0-flash")
    return MODEL_PROFILES.get(model_spec, model_spec)
```

### Adding a new profile
Edit `MODEL_PROFILES` in `extraction.py` — one line, no migrations needed:
```python
MODEL_PROFILES["turbo"] = "accounts/fireworks/models/llama-3.1-70b-turbo"
```

### Tracking which model was used
- `ExtractionJob.model_id` stores the resolved primary model ID for every job.
- Exposed in `GET /api/jobs/<pk>/` response as `"model_id"`.
- Query jobs by model: `ExtractionJob.objects.filter(model_id="gemini-2.0-flash")`

### Fallback is always from settings
The `model` override applies **only to the primary call**.
The fallback is always `LLM_MODEL_FALLBACK` from `.env` — so you can pin a reliable fallback while experimenting with different primaries.

### Replay a job with a different model from shell
```python
from extractor.extraction import run_invoice_extraction
from extractor.models import ExtractionJob
job = ExtractionJob.objects.get(pk=12)
result = run_invoice_extraction(job.ocr_draft, model_id="quality")
print(result["model_id"], result["metrics"])
```

---

## 8. Debugging Extractions

### Check field fill rates
```
GET /api/metrics/
→ avg_field_fill_rates: {"hs_code": 0.87, "cost": 0.95, ...}
```
Low fill rate on a field → tighten the prompt rule for that field.

### Visualize what LangExtract highlighted
```
GET /api/jobs/<pk>/visualize/
```
Interactive HTML — shows exactly which spans were matched per extraction.

### Check which step is slow
```
GET /api/jobs/<pk>/
→ metrics.t_primary_llm_s   ← LLM time
→ metrics.t_validate_s      ← JSON parsing
→ metrics.t_total_s         ← wall clock
```

### Replay a stuck/failed job from shell
```python
from extractor.extraction import run_invoice_extraction
from extractor.models import ExtractionJob
job = ExtractionJob.objects.get(pk=12)
result = run_invoice_extraction(job.ocr_draft)
print(result)
```
