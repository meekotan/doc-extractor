"""
Invoice extraction service for document code 04021.

Translates the original Dify workflow branch into Python using LangExtract:

  Node 1765260281035  →  clean_text()
  Node 1765260372029  →  extract_with_langextract()   (primary: Gemini)
  Node 1765260722411  →  validate_and_parse()
  Node 1765342175106  →  currency.finalize_items()
  Node 1765261045587  →  extract_with_langextract()   (fallback: GPT-4o)

Optimisations over original Dify implementation:
  - extract_header()      → pulls top 25 lines of OCR as shared context
  - additional_context    → header injected into EVERY LangExtract chunk
  - max_char_buffer=4000  → fewer chunks = fewer API calls = faster
  - post_fill_from_header → fills empty metadata fields from parsed header
  - deduplicate_items()   → removes duplicate rows from bilingual invoices
                            (RU description + EN packing list → same item twice)
"""

import json
import logging
import re
import time
from typing import List, Optional

logger = logging.getLogger(__name__)

from pydantic import BaseModel

# json5 / demjson3 не нужны — используем ручную чистку через regex


# ---------------------------------------------------------------------------
# Pydantic schema for Gemini native structured output
# ---------------------------------------------------------------------------

class InvoiceItem(BaseModel):
    hs_code: Optional[str] = None
    description: str
    quantity: Optional[float] = None
    unit: Optional[str] = None
    cost: Optional[float] = None
    price: Optional[float] = None
    currency_code: Optional[str] = None
    currency_name: Optional[str] = None
    document_date: Optional[str] = None
    document_number: Optional[str] = None
    country_origin: Optional[str] = None
    country_origin_code: Optional[int] = None
    country_sender: Optional[str] = None
    suggestions: Optional[List] = None


class InvoiceExtraction(BaseModel):
    items: List[InvoiceItem]


def _repair_json(text: str) -> str:
    """
    Attempt to fix common JSON issues returned by LLMs:
    - trailing commas before } or ]
    - single-quoted strings → double-quoted
    - Python literals: True/False/None → true/false/null
    """
    # Remove trailing commas before closing bracket/brace
    text = re.sub(r",\s*([}\]])", r"\1", text)
    # Python booleans / None
    text = re.sub(r"\bTrue\b", "true", text)
    text = re.sub(r"\bFalse\b", "false", text)
    text = re.sub(r"\bNone\b", "null", text)
    return text

import langextract as lx
from langextract import factory as lx_factory
from django.conf import settings

from .currency import build_currency_db_string, finalize_items, load_currency_db
from .metrics import RunMetrics, compute_field_fill_rates, timer

# ---------------------------------------------------------------------------
# Prompt / schema mirrors the original system prompt (Node 1765260372029)
# ---------------------------------------------------------------------------

EXTRACTION_PROMPT = """
<role>
You are a highly precise Data Extractor for Customs Invoices. Your mission is to mirror the document's data into JSON with zero information loss.
</role>

<language_logic>
- PRIMARY LANGUAGE: Russian.
- BILINGUAL DATA: If a description is provided in both English and Russian, extract ONLY the Russian version.
- MONOLINGUAL DATA: If the text is only in Russian or only in English, extract it exactly as written.
- NO TRUNCATION: Capture the full technical description. Do not summarize.
</language_logic>

<hs_code_rules>
- VERBATIM COPY: Extract the HS Code (ТН ВЭД) exactly as it appears in the document.
- NO TRUNCATION: Do not remove any digits. If the code is 4, 6, 8, or 10 digits long, copy every single digit.
- PRIORITY: If you see a short code (e.g., in the EN section) and a longer code (e.g., in the RU section) for the same item, ALWAYS prefer the longer version.
</hs_code_rules>

<field_specifications>
- description: Full technical name (Priority: Russian text if available).
- hs_code: Verbatim string of digits. Do not truncate, do not add digits.
- unit: Mandatory. Standardize: kg, pcs, l, set, m. (Default: "pcs").
- quantity: Numeric (use "." as decimal separator).
- cost: Unit price. If missing, calculate: price / quantity.
- price: Total row price.
- country_origin: 2-letter ISO code (e.g., DE, NL, RU). If not found, "Неизвестно".
- currency: From DOCUMENT HEADER (additional_context).
</field_specifications>

<orchestration_data>
Using the "additional_context" (DOCUMENT HEADER), apply "document_number", "document_date", and "country_sender" to EVERY item in the resulting JSON array.
</orchestration_data>

<output_format>
Return ONLY a valid JSON array of objects. 
No markdown fences (```json), no preamble. 
Start directly with [ and end with ].
</output_format>
"""

# Few-shot example that teaches LangExtract the schema shape.
EXAMPLES = [
    lx.data.ExampleData(
        # Bilingual invoice example: RU section first, EN section second.
        # Shows GLM-5 exactly which fields to take from which section.
        text=(
            "=== CURRENCY DATABASE (REFERENCE) ===\n"
            '[{"code":"EUR","name":"Euro"}]\n\n'
            "=== INVOICE CONTENT ===\n"
            "Коммерческий инвойс № INV-001  Дата: 10/01/2025\n"
            "Грузоотправитель: Acme GmbH, Germany\n\n"
            "--- SECTION RU ---\n"
            "| № | Наименование товара | Кол-во | Цена за ед. | Стоимость |\n"
            "| 1 | Ноутбук Dell XPS код ТНВЭД 8471309900 | 2 шт. | 850,00 | 1700,00 |\n\n"
            "--- SECTION EN (CUSTOMS INVOICE) ---\n"
            "| Material No. | Description | Co. of Origin | Customs tariff | Qty | Unit price | Value |\n"
            "| 12345 | Dell XPS Laptop | DE | 84713099 | 2,00 | 850,00 | 1700,00 |\n"
        ),
        extractions=[
            lx.data.Extraction(
                extraction_class="invoice_item",
                # extraction_text must be the Russian description
                extraction_text="Ноутбук Dell XPS",
                attributes={
                    # hs_code: 10-digit from RU section, NOT 8-digit from EN section
                    "hs_code": "8471309900",
                    # description: Russian text, NOT English
                    "description": "Ноутбук Dell XPS",
                    "quantity": 2,
                    "unit": "pcs",
                    "cost": 850.00,
                    "price": 1700.00,
                    "currency_code": "EUR",
                    "currency_name": "Euro",
                    "document_date": "10/01/2025",
                    "document_number": "INV-001",
                    # country_origin: ISO-2 code taken from EN "Co. of Origin" column
                    "country_origin": "DE",
                    "country_origin_code": 276,
                    "country_sender": "Germany",
                },
            )
        ],
    )
]

# ---------------------------------------------------------------------------
# Header extraction — shared context injected into every LangExtract chunk
# ---------------------------------------------------------------------------

# Fields that must be propagated from the header to all items
_HEADER_FIELDS = ("document_date", "document_number", "country_sender",
                  "currency_code", "currency_name")

# Regex patterns to parse header metadata directly from OCR text
_HEADER_PATTERNS = {
    "document_number": re.compile(
        r"(?:invoice\s*(?:no|num|#|:)|№|накладн|счет|invoice)\W*([A-Z0-9\-/]{4,30})",
        re.IGNORECASE,
    ),
    "document_date": re.compile(
        r"(?:date|дата|от)\W*(\d{1,2}[\/\.\-]\d{1,2}[\/\.\-]\d{2,4})",
        re.IGNORECASE,
    ),
    "country_sender": re.compile(
        r"(?:sender|отправитель|shipper|from|страна отправ\w*)\W*:?\s*([A-Za-zА-Яа-яёЁ ]{3,40})",
        re.IGNORECASE,
    ),
}


# Признак строки с товарной позицией: строка начинается с порядкового номера
# (цифра + пробел/таб) И содержит числовую цену (1234.56 или 1234,56).
# Такие строки НЕ включаются в шапку.
_ITEM_ROW_START_RE = re.compile(r'^\s*\d+[\s\t]+\S')
_PRICE_RE          = re.compile(r'\b\d+[.,]\d{2}\b')

# Признак строки-заголовка таблицы: содержит ключевые слова названий колонок.
# Такую строку ВКЛЮЧАЕМ в шапку, но после неё останавливаемся — она даёт ЛЛМ
# контекст структуры колонок без утечки самих товарных позиций.
_TABLE_HEADER_RE = re.compile(
    r'\b(?:наименование|кол[-.\s]?во|количество|стоимость|ед[.\s]?изм'
    r'|unit\b|qty\b|quantity\b|amount\b|description\b|тн\s*вэд|hs\s*code)',
    re.IGNORECASE,
)


def extract_header(cleaned_text: str) -> str:
    """
    Extract document metadata header — includes table column headers but stops
    before the first invoice item data row.

    Two-phase stop logic:
    1. Hard stop BEFORE any actual item row (starts with row-number + price).
       These must never appear in additional_context or every LangExtract chunk
       will extract them and produce duplicates.
    2. Soft stop AFTER the table column header row (Наименование / Qty / HS Code …).
       We include that line so the LLM knows the column layout in each chunk,
       then stop — item rows follow immediately and must not be included.
    """
    lines = cleaned_text.split("\n")
    header_lines = []

    for line in lines:
        if not line.strip():
            continue
        # Phase 1: hard stop — never include actual data rows
        if _ITEM_ROW_START_RE.match(line) and _PRICE_RE.search(line):
            break
        # Include the line
        header_lines.append(line)
        # Phase 2: soft stop — include the column-header row, then stop
        if _TABLE_HEADER_RE.search(line):
            break
        if len(header_lines) >= 25:
            break

    if not header_lines:
        return ""
    return (
        "=== DOCUMENT HEADER (applies to ALL items in this invoice) ===\n"
        + "\n".join(header_lines)
    )



def parse_header_metadata(header_text: str) -> dict:
    """
    Extract key metadata from the header text using regex.
    Used as a fallback to fill empty fields after LLM extraction.
    Returns a dict with any fields found (only non-empty values).
    """
    meta = {}
    for field, pattern in _HEADER_PATTERNS.items():
        m = pattern.search(header_text)
        if m:
            value = m.group(1).strip().rstrip(",;.")
            if value:
                meta[field] = value
    return meta


def post_fill_from_header(items: list[dict], header_meta: dict,
                           currency_db: list[dict]) -> list[dict]:
    """
    For every item, fill in any empty header-derived field using the values
    parsed from the document header.  Currency code/name are resolved through
    the currency DB to ensure consistency.
    """
    if not header_meta:
        return items

    # Resolve currency from header if present
    header_currency_code = header_meta.get("currency_code")
    header_currency_name = header_meta.get("currency_name")

    for item in items:
        for field in _HEADER_FIELDS:
            current = item.get(field)
            is_empty = (
                current is None
                or str(current).strip().lower() in ("", "null", "none", "0")
            )
            if is_empty and field in header_meta:
                item[field] = header_meta[field]

        # Special case: if currency_code is still empty, try from DB via header
        if not item.get("currency_code") and header_currency_code:
            item["currency_code"] = header_currency_code
        if not item.get("currency_name") and header_currency_name:
            item["currency_name"] = header_currency_name

    return items


# ---------------------------------------------------------------------------
# Deduplication — removes duplicate rows from bilingual invoices
# ---------------------------------------------------------------------------

def deduplicate_items(items: list[dict]) -> list[dict]:
    """
    Remove duplicate invoice rows that arise when an OCR document contains
    the same goods in multiple sections (e.g. RU table + EN customs table +
    RU packing list — all three repeat the same items).

    Strategy — single pass by (normalized_description, price, quantity, cost).

    Adding the normalized description to the key makes deduplication safe for
    two conflicting cases:
    - Bilingual invoice with repeated sections: each item appears 2-3× with
      IDENTICAL Russian descriptions and identical numerics → correctly merged.
    - Dense invoice (e.g. 217 items): different items that happen to share the
      same price/qty/cost have DIFFERENT descriptions → NOT merged → all preserved.

    The row that is kept is always the one with the most filled fields (richer row).
    """
    if not items:
        return items

    def _numeric(val, decimals: int = 2) -> float:
        try:
            return round(float(val), decimals)
        except (TypeError, ValueError):
            return -1.0

    def _normalize_description(desc: str) -> str:
        """Lowercase, strip punctuation, collapse whitespace."""
        s = str(desc or "").lower()
        s = re.sub(r'[^\w]', ' ', s)
        s = re.sub(r'\s+', ' ', s).strip()
        return s

    def _filled_count(item: dict) -> int:
        """Count non-empty fields — used to prefer the richer row."""
        return sum(
            1 for v in item.values()
            if v is not None and str(v).strip().lower() not in ("", "null", "none", "0")
        )

    # --- Pass 1: dedup by (normalized_description, price, quantity, cost) ---
    seen_exact: dict[tuple, int] = {}   # key → index in result list
    result: list[dict] = []

    for item in items:
        key = (
            _normalize_description(item.get("description", "")),
            _numeric(item.get("price")),
            _numeric(item.get("quantity"), 3),
            _numeric(item.get("cost")),
        )
        # Skip rows where all numeric values are zero/missing — likely
        # header lines that slipped through validation.
        if key[1:] == (-1.0, -1.0, -1.0) or key[1:] == (0.0, 0.0, 0.0):
            result.append(item)
            continue

        if key in seen_exact:
            # Keep the richer row
            existing_idx = seen_exact[key]
            if _filled_count(item) > _filled_count(result[existing_idx]):
                result[existing_idx] = item
        else:
            seen_exact[key] = len(result)
            result.append(item)

    return result


# ---------------------------------------------------------------------------
# Step 1 — text cleaning  (mirrors Node 1765260281035)
# ---------------------------------------------------------------------------

def clean_text(raw_text: str, currency_db_str: str) -> str:
    """Clean OCR text and prepend the currency database block."""
    if not raw_text:
        return ""

    text = raw_text.replace("\r", "")
    text = re.sub(r"\n{3,}", "\n\n", text)

    lines = []
    for line in text.split("\n"):
        clean_line = re.sub(r"[ \t]+", " ", line).strip()
        if clean_line:
            lines.append(clean_line)

    cleaned_invoice_text = "\n".join(lines)

    return (
        f"=== CURRENCY DATABASE (REFERENCE) ===\n"
        f"{currency_db_str}\n\n"
        f"=== INVOICE CONTENT ===\n"
        f"{cleaned_invoice_text}"
    )


# ---------------------------------------------------------------------------
# Step 3 — validation / JSON repair  (mirrors Node 1765260722411)
# ---------------------------------------------------------------------------

def validate_and_parse(text: str) -> dict:
    """
    Парсит и нормализует JSON, полученный после параллельного извлечения LangExtract.
    Поддерживает склейку чанков и восстановление поврежденного JSON.
    """
    # 1. Предварительная чистка (удаляем Markdown и лишние пробелы)
    clean = re.sub(r"```json|```", "", text).strip()
    
    # Пытаемся изолировать массив, если LLM добавила лишний текст
    match = re.search(r"\[.*\]", clean, re.DOTALL)
    if match:
        clean = match.group()

    parsed_data = []

    # 2. Попытка парсинга с механизмом восстановления
    try:
        parsed_data = json.loads(clean)
    except json.JSONDecodeError:
        # Пытаемся починить (trailing commas, Python-типы)
        try:
            repaired = _repair_json(clean)
            parsed_data = json.loads(repaired)
        except json.JSONDecodeError:
            # Восстановление при обрыве строки (Unterminated string) — берем последний валидный объект
            try:
                # Находим последний закрытый объект в массиве
                last_complete_item = repaired.rfind("}")
                if last_complete_item != -1:
                    fixed = repaired[:last_complete_item + 1] + "]"
                    parsed_data = json.loads(fixed)
            except:
                parsed_data = []

    # 3. Нормализация структуры
    # Если на выходе словарь с ключом 'items' (результат response_schema) — разворачиваем его
    if isinstance(parsed_data, dict):
        parsed_data = parsed_data.get("items", [parsed_data])
    
    # Если это список списков (результат слияния нескольких чанков), выпрямляем его
    if isinstance(parsed_data, list) and any(isinstance(i, list) for i in parsed_data):
        flattened = []
        for sublist in parsed_data:
            if isinstance(sublist, list):
                flattened.extend(sublist)
            else:
                flattened.append(sublist)
        parsed_data = flattened

    # 4. Финальная валидация и типизация полей
    valid_items = []
    for item in parsed_data:
        if not isinstance(item, dict):
            continue
            
        # Обязательное поле — описание. Если его нет, это не товарная позиция.
        description = item.get("description", "")
        if description and str(description).strip().lower() not in ("none", "null", ""):
            
            # Чистим hs_code
            hs = str(item.get("hs_code", "")).strip().lower()
            if hs in ("none", "null", "", "0", "false"):
                item["hs_code"] = None
            
            # Принудительная типизация чисел для Django моделей
            # Gemini 3 на Free Tier иногда может вернуть число как строку
            for field in ["quantity", "cost", "price"]:
                try:
                    val = str(item.get(field, "0")).replace(",", ".")
                    item[field] = float(re.sub(r"[^\d.]", "", val) or 0)
                except (ValueError, TypeError):
                    item[field] = 0.0
            
            valid_items.append(item)

    if not valid_items:
        return {
            "data": {"items": [], "count": 0},
            "error": "No valid items extracted",
            "is_valid": 0,
        }
    return {
        "data": {"items": valid_items, "count": len(valid_items)},
        "error": "",
        "is_valid": 1,
    }


# ---------------------------------------------------------------------------
# Orchestrator — full 04021 pipeline
# ---------------------------------------------------------------------------

def run_invoice_extraction(ocr_draft: str, model_id: str | None = None) -> dict:
    """
    Full pipeline for document_code == '04021'.

    Args:
        ocr_draft:  Raw OCR text from the invoice.
        model_id:   Optional model override.  Accepts a profile alias ("fast",
                    "quality", "glm") or any raw model ID string.  When None,
                    falls back to LLM_MODEL_PRIMARY from Django settings.
                    The fallback model always comes from LLM_MODEL_FALLBACK.

    Returns:
        {"result": {"items": [...], "count": n}, "metrics": {...}, "model_id": "..."}  on success
        {"error": "...", "metrics": {...}, "model_id": "..."}                           on failure
    """
    m = RunMetrics()
    t_wall_start = time.perf_counter()

    currency_db = load_currency_db()
    currency_db_str = build_currency_db_string(currency_db)

    # Step 1 – clean text
    with timer() as t:
        context = clean_text(ocr_draft, currency_db_str)
    m.t_clean_s = t[0]

    # Resolve which model to use for this run
    primary_model = resolve_model_id(model_id)
    fallback_model = getattr(settings, "LLM_MODEL_FALLBACK", "gemini-2.0-flash")
    # Tracks the model that actually produced the accepted result
    effective_model = primary_model

    if not context:
        m.t_total_s = time.perf_counter() - t_wall_start
        return {"error": "Empty OCR text", "metrics": m.to_dict(), "model_id": effective_model}

    # Extract header for additional_context injection into every chunk
    header_context = extract_header(context)
    # Parse header metadata for post-fill fallback
    header_meta = parse_header_metadata(header_context)

    def _extract(mid: str) -> tuple[str, object]:
        """Returns (json_str, annotated_doc) from LangExtract."""
        return extract_with_langextract_optimized(context, mid, header_context)

    # Step 2 – primary extraction
    with timer() as t:
        raw_output, annotated_doc = _extract(primary_model)
    m.t_primary_llm_s = t[0]

    # Step 3 – validate
    with timer() as t:
        validation = validate_and_parse(raw_output)
    m.t_validate_s = t[0]
    m.primary_valid = bool(validation["is_valid"])

    # Step 4 – fallback если primary не справился
    if not validation["is_valid"]:
        m.fallback_used = True
        effective_model = fallback_model   # fallback is now the active model
        with timer() as t:
            raw_output, annotated_doc = _extract(fallback_model)
        m.t_fallback_llm_s = t[0]

        with timer() as t:
            validation = validate_and_parse(raw_output)
        m.t_validate_s += t[0]
        m.fallback_valid = bool(validation["is_valid"])

    if not validation["is_valid"]:
        m.t_total_s = time.perf_counter() - t_wall_start
        return {
            "error": validation.get("error", "Extraction failed after fallback"),
            "metrics": m.to_dict(),
            "model_id": effective_model,
        }

    # Step 5 – currency resolution & finalisation
    with timer() as t:
        items = validation["data"]["items"]
        # Post-fill empty metadata fields from parsed header
        items = post_fill_from_header(items, header_meta, currency_db)
        # Programmatic fallback for cost and unit if LLM left them empty
        for item in items:
            # cost: вычислить из price / quantity если пустой
            cost = item.get("cost")
            is_empty_cost = cost is None or str(cost).strip().lower() in ("", "null", "none", "0")
            if is_empty_cost:
                try:
                    price = float(item.get("price") or 0)
                    qty = float(item.get("quantity") or 1)
                    item["cost"] = round(price / qty, 4) if qty else price
                except (TypeError, ValueError, ZeroDivisionError):
                    pass
            # unit: default "pcs" если пустой
            unit = item.get("unit")
            if unit is None or str(unit).strip().lower() in ("", "null", "none"):
                item["unit"] = "pcs"
        final_items = finalize_items(items, currency_db)
        # Remove cross-section duplicates (bilingual invoices repeat items in
        # RU + EN + packing-list sections). Safe for dense invoices too because
        # the key includes normalized description — different items with the
        # same price/qty/cost are NOT merged.
        final_items = deduplicate_items(final_items)
    m.t_finalize_s = t[0]

    m.items_extracted = len(final_items)
    m.field_fill_rates = compute_field_fill_rates(final_items)
    m.t_total_s = round(time.perf_counter() - t_wall_start, 3)

    return {
        "result": {"items": final_items, "count": len(final_items)},
        "metrics": m.to_dict(),
        "annotated_doc": annotated_doc,   # lx.data.AnnotatedDocument — for visualizer
        "model_id": effective_model,
    }


# ---------------------------------------------------------------------------
# Model profiles — friendly aliases for concrete model IDs
# ---------------------------------------------------------------------------

# Edit this dict to add/rename profiles. Keys are the strings callers can pass
# in the POST body as {"model": "fast"}.  Values are the actual model IDs sent
# to LangExtract.  You can also pass a raw model ID directly (e.g.
# "gemini-2.5-pro") — resolve_model_id() will use it verbatim.
MODEL_PROFILES: dict[str, str] = {
    "fast":    "gemini-2.0-flash",          # good quality, fastest Gemini
    "quality": "gemini-2.5-pro",            # highest quality, slower
    "glm":     "accounts/fireworks/models/glm-5",  # OpenAI-compat via Fireworks
}


def resolve_model_id(model_spec: str | None) -> str:
    """
    Resolve a model spec to a concrete model ID.

    Resolution order:
      1. None / ""       → LLM_MODEL_PRIMARY from Django settings
      2. Profile alias   → MODEL_PROFILES[model_spec]
      3. Raw model ID    → used verbatim (e.g. "gemini-2.5-flash-preview-04-17")
    """
    if not model_spec:
        return getattr(settings, "LLM_MODEL_PRIMARY", "gemini-2.0-flash")
    return MODEL_PROFILES.get(model_spec, model_spec)


# ---------------------------------------------------------------------------
# Provider resolution — Gemini, OpenAI, Fireworks, any OpenAI-compatible API
# ---------------------------------------------------------------------------

# OpenAI-compatible провайдеры: prefix → base_url
_OPENAI_COMPATIBLE = {
    "accounts/fireworks/": "https://api.fireworks.ai/inference/v1",
    "openrouter/":         "https://openrouter.ai/api/v1",
}


def _register_openai_compatible_patterns() -> None:
    """
    Register OpenAI-compatible model prefixes into LangExtract's router so
    model IDs like 'accounts/fireworks/models/glm-5' resolve to
    OpenAILanguageModel automatically (priority 20 beats default 10).
    Idempotent — router deduplicates entries.
    """
    import langextract.providers as lx_providers
    from langextract.providers import router as lx_router
    from langextract.providers.openai import OpenAILanguageModel
    lx_providers.load_builtins_once()
    for prefix in _OPENAI_COMPATIBLE:
        pattern = r"^" + prefix.replace("/", r"\/")
        lx_router.register(pattern, priority=20)(OpenAILanguageModel)


# Register at import time (runs once per process)
_register_openai_compatible_patterns()


def _build_lx_config(model_id: str) -> lx_factory.ModelConfig:
    """
    Build a LangExtract ModelConfig:
      - gemini-*             → Gemini  (LANGEXTRACT_API_KEY)
      - gpt-*                → OpenAI  (LANGEXTRACT_API_KEY)
      - accounts/fireworks/* → Fireworks AI (FIREWORKS_API_KEY) + base_url injected
    """
    for prefix, base_url in _OPENAI_COMPATIBLE.items():
        if model_id.startswith(prefix):
            if "fireworks" in prefix:
                api_key = (
                    getattr(settings, "FIREWORKS_API_KEY", "")
                    or getattr(settings, "LANGEXTRACT_API_KEY", "")
                )
            else:
                api_key = getattr(settings, "LANGEXTRACT_API_KEY", "")

            return lx_factory.ModelConfig(
                model_id=model_id,
                provider_kwargs={
                    "base_url": base_url,
                    "api_key": api_key or None,
                    "max_workers": 15,
                },
            )

    # Gemini / native OpenAI — auto-resolved by router
    api_key = getattr(settings, "LANGEXTRACT_API_KEY", "")
    return lx_factory.ModelConfig(
        model_id=model_id,
        provider_kwargs={
            "api_key": api_key or None,
            "max_workers": 4,
        },
    )


# ---------------------------------------------------------------------------
# Оптимизированный вызов извлечения
# ---------------------------------------------------------------------------

def extract_with_langextract_optimized(
    context_text: str,
    model_id: str,
    header_context: str = "",
) -> tuple[str, object]:
    """
    Run LangExtract extraction and return (json_str, annotated_doc).

    Supports:
      - gemini-*              → Google Gemini (LANGEXTRACT_API_KEY)
      - gpt-*                 → OpenAI (OPENAI_API_KEY / LANGEXTRACT_API_KEY)
      - accounts/fireworks/*  → Fireworks AI (FIREWORKS_API_KEY)

    The second element is an ``lx.data.AnnotatedDocument``.
    """
    config = _build_lx_config(model_id)

    try:
        annotated_doc = lx.extract(
            text_or_documents=context_text,
            prompt_description=EXTRACTION_PROMPT,
            examples=EXAMPLES,
            config=config,
            additional_context=header_context or None,
            max_char_buffer=2000,
        )
    except Exception as exc:
        logger.warning("lx.extract() failed (%s: %s) — returning empty result", type(exc).__name__, exc)
        return "[]", None

    all_items = []
    for extraction in annotated_doc.extractions:

        item = {"description": extraction.extraction_text}
        if extraction.attributes:
            item.update(extraction.attributes)
        all_items.append(item)

    return json.dumps(all_items, ensure_ascii=False), annotated_doc
