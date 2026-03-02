"""
Invoice extraction service for document code 04021.

Translates the original Dify workflow branch into Python using LangExtract:

  Node 1765260281035  →  clean_text()
  Node 1765260372029  →  extract_with_langextract()   (primary: GPT-OSS 120B)
  Node 1765260722411  →  validate_and_parse()
  Node 1765342175106  →  currency.finalize_items()
  Node 1765261045587  →  extract_with_langextract()   (fallback: Gemini 2.5 Flash)

Optimisations over original Dify implementation:
  - extract_header()      → pulls top 25 lines of OCR as shared context
  - additional_context    → header injected into EVERY LangExtract chunk
  - post_fill_from_header → fills empty metadata fields from parsed header
  - deduplicate_items()   → removes cross-chunk duplicates keyed on `position`
                            (same POS in two overlapping chunks → merged once)
"""

import json
import logging
import re
import time

logger = logging.getLogger(__name__)

import langextract as lx
from langextract import factory as lx_factory
from django.conf import settings

from .currency import build_currency_db_string, finalize_items, load_currency_db
from .metrics import RunMetrics, compute_field_fill_rates, timer


# ---------------------------------------------------------------------------
# Cerebras native SDK — singleton client + JSON schema for response_format
# ---------------------------------------------------------------------------

# JSON Schema that the Cerebras API enforces on the model's output.
# Using an {"items": [...]} wrapper because json_schema mode requires an
# object at the root (not a bare array).
# validate_and_parse() already handles the {"items": [...]} dict unwrap.
_CEREBRAS_ITEM_SCHEMA = {
    "type": "object",
    "properties": {
        "items": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "position":            {"type": ["integer", "null"]},
                    "description":         {"type": "string"},
                    "hs_code":             {"type": ["string", "null"]},
                    "quantity":            {"type": ["number", "null"]},
                    "unit":                {"type": "string"},
                    "cost":                {"type": ["number", "null"]},
                    "price":               {"type": ["number", "null"]},
                    "currency_code":       {"type": ["string", "null"]},
                    "currency_name":       {"type": ["string", "null"]},
                    "document_date":       {"type": ["string", "null"]},
                    "document_number":     {"type": ["string", "null"]},
                    "country_origin":      {"type": "string"},
                    "country_origin_code": {"type": ["integer", "null"]},
                    "country_sender":      {"type": ["string", "null"]},
                },
                "required": ["description"],
            },
        }
    },
    "required": ["items"],
}

_CEREBRAS_RESPONSE_FORMAT = {
    "type": "json_schema",
    "json_schema": {
        "name":   "invoice_items",
        "schema": _CEREBRAS_ITEM_SCHEMA,
    },
}

# Module-level singleton — created once, reused across all requests.
# Lazy init so Django settings are available at first call time.
_cerebras_client = None


def _get_cerebras_client():
    """Return the process-wide Cerebras client, creating it on first call."""
    global _cerebras_client
    if _cerebras_client is None:
        from cerebras.cloud.sdk import Cerebras
        _cerebras_client = Cerebras(
            base_url=getattr(settings, "CEREBRAS_BASE_URL", "https://api.cerebras.ai"),
            api_key=getattr(settings, "CEREBRAS_API_KEY", ""),
        )
        logger.debug("[cerebras] singleton client created (TCP warming active)")
    return _cerebras_client


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

# ---------------------------------------------------------------------------
# Prompt / schema mirrors the original system prompt (Node 1765260372029)
# ---------------------------------------------------------------------------

EXTRACTION_PROMPT = """
# ROLE
You are a specialized agent for high-precision extraction of customs invoice data (TN VED/EAEU standards). Your goal is to produce a JSON output that mirrors the document content with 100% fidelity.
# CORE EXTRACTION LOGIC
1. **Language Priority**:
   - Primary: Russian (RU).
   - If a line item has both RU and EN descriptions, extract ONLY the Russian text.
   - If only one language is present (RU or EN), extract it exactly as is.
   - DO NOT translate. DO NOT truncate. Capture full technical strings.
2. **HS Codes (ТН ВЭД) Handling**:
   - **Verbatim Extraction**: Copy digits exactly. No spaces, no dots.
   - **Length Priority**: If you find two versions of a code for one item (e.g., 6-digit and 10-digit), ALWAYS pick the longest one.
   - **No Modification**: Never add or remove digits. If the code is 4, 6, 8, or 10 digits, keep it as is.
3. **Data Integrity**:
   - Extract every single row that has a price.
   - Do not merge similar rows. Do not skip any items.
# FIELD SPECIFICATIONS
- `position`: Integer row number from the № / POS / п/п column. Must be unique per invoice line. If missing, use null.
- `description`: Full Russian name (priority) or English name. No summarization.
- `hs_code`: String of digits. Verbatim copy.
- `unit`: Standardize to: kg, pcs, l, set, m. (Default to "pcs" if missing).
- `quantity`: Numeric value (use "." decimal separator).
- `cost`: Price per unit. If not explicit, calculate: `price` / `quantity`.
- `price`: Total row price.
- `country_origin`: 2-letter ISO code (DE, NL, CN, RU). If missing, use "Неизвестно".
- `country_origin_code`: Numeric ISO country code (e.g., 276 for DE, 156 for CN). If missing, use null.
- `currency_code`: 3-letter ISO code from DOCUMENT HEADER in `additional_context` (e.g., "USD", "EUR").
- `currency_name`: Full currency name from DOCUMENT HEADER in `additional_context` (e.g., "US Dollar", "Euro").
# GLOBAL DOCUMENT CONTEXT
From the `additional_context` (DOCUMENT HEADER), you MUST apply these values to EVERY item in the JSON array:
- `document_number`
- `document_date`
- `country_sender`
- `currency_code`
- `currency_name`
# OUTPUT FORMATTING
- Return ONLY a raw JSON array of objects.
- Do not include markdown code blocks (```json).
- No preamble, no post-text, no explanations.
- Start with `[` and end with `]`.
"""

EXTRACTION_PROMPT_GPT_OSS = """
ROLE:
You are a deterministic customs invoice line-item extractor.
TASK:
Extract ALL line items from the provided invoice text.
EXTRACTION RULES:
1. POSITION
   - Read the row number from the № / POS / п/п / No. column and output it as "position" (integer).
   - Each invoice line has a unique position. Two rows with identical data but different position numbers are genuinely different line items — DO NOT skip either.
   - If no position column is present, output null.
2. LANGUAGE PRIORITY
   - Prefer Russian descriptions.
   - If the document contains both Russian and English sections for the same item, extract ONLY the Russian description.
   - If the document is monolingual, preserve the original language exactly as written.
3. HS CODE
   - Copy exactly as written.
   - Do NOT truncate.
   - Do NOT normalize.
   - Do NOT add digits.
   - If multiple HS codes exist for one item, select the longest one.
4. HEADER DATA (GLOBAL CONTEXT)
   The "=== DOCUMENT HEADER ===" block at the top of the document contains header-level data.
   The following fields MUST be copied to EVERY extracted item:
   - document_number
   - document_date
   - country_sender
   - currency_code
   - currency_name
5. COST / PRICE FIELD MAPPING
   - "cost"  = unit price (price per single item) — maps to columns named "unit_price", "Unit Price", "Preis/Einheit", etc.
   - "price" = total line price (quantity × unit price) — maps to columns named "total_price", "Total", "Gesamtpreis", "Стоимость", etc.
   - If "cost" is missing but "price" and "quantity" are present: cost = price / quantity.
   - If calculation is impossible, use null. Do NOT guess.
6. UNIT NORMALIZATION
   - If unit is missing, default to "pcs".
   - Allowed values ONLY:
     kg, pcs, l, set, m
   - Normalize variations (e.g., "шт." → "pcs", "кг" → "kg").
7. COUNTRY OF ORIGIN
   - country_origin must be a 2-letter ISO code (e.g., DE, CN, RU).
   - If unknown, use "Неизвестно".
   - country_origin_code must be the numeric ISO country code.
   - If unknown, use null.
   - Do NOT invent country data.
OUTPUT:
Return a JSON object: {"items": [...]}.
The "items" array contains one object per invoice line item.
No markdown, no explanation, no preamble.
If no items found, return {"items": []}.
"""

# Few-shot examples that teach LangExtract the schema shape.
#
# Example 1 — BILINGUAL invoice (RU + EN sections).
#   Rule: prefer Russian description; prefer the LONGER hs_code (RU section
#   often has 10-digit, EN section 8-digit — but length varies per document).
#
# Example 2 — MONOLINGUAL ENGLISH invoice.
#   Rule: keep English description as-is; hs_code may be any valid length
#   (4, 6, 8, or 10 digits) — copy verbatim, never truncate or extend.
EXAMPLES = [
    lx.data.ExampleData(
        # BILINGUAL invoice: RU section first, EN customs section second.
        # Demonstrates: pick Russian description, pick longer hs_code.
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
                # Bilingual: extraction_text = Russian description
                extraction_text="Ноутбук Dell XPS",
                attributes={
                    # Row number from the № column
                    "position": 1,
                    # Prefer the longer 10-digit RU code over the 8-digit EN code
                    "hs_code": "8471309900",
                    # Bilingual: use Russian description, not English
                    "description": "Ноутбук Dell XPS",
                    "quantity": 2,
                    "unit": "pcs",
                    "cost": 850.00,
                    "price": 1700.00,
                    "currency_code": "EUR",
                    "currency_name": "Euro",
                    "document_date": "10/01/2025",
                    "document_number": "INV-001",
                    # country_origin: ISO-2 from EN "Co. of Origin" column
                    "country_origin": "DE",
                    "country_origin_code": 276,
                    "country_sender": "Germany",
                },
            )
        ],
    ),
    lx.data.ExampleData(
        # MONOLINGUAL ENGLISH invoice — no Russian section present.
        # Demonstrates: keep English description as-is; hs_code may be
        # 8 digits (or any valid length) — copy verbatim, do not extend.
        text=(
            "=== CURRENCY DATABASE (REFERENCE) ===\n"
            '[{"code":"USD","name":"US Dollar"}]\n\n'
            "=== INVOICE CONTENT ===\n"
            "Commercial Invoice No. CI-2024-089  Date: 05/03/2024\n"
            "Shipper: TechParts Inc., United States\n\n"
            "No. | Description            | HS Code  | Qty | Unit price | Total\n"
            "1   | Industrial servo motor | 85016100 |  5  | 125.00     | 625.00\n"
        ),
        extractions=[
            lx.data.Extraction(
                extraction_class="invoice_item",
                # Monolingual EN: extraction_text = English description
                extraction_text="Industrial servo motor",
                attributes={
                    # Row number from the No. column
                    "position": 1,
                    # 8-digit hs_code — valid length, must NOT be extended to 10
                    "hs_code": "85016100",
                    # Monolingual EN: keep English description as-is
                    "description": "Industrial servo motor",
                    "quantity": 5,
                    "unit": "pcs",
                    "cost": 125.00,
                    "price": 625.00,
                    "currency_code": "USD",
                    "currency_name": "US Dollar",
                    "document_date": "05/03/2024",
                    "document_number": "CI-2024-089",
                    "country_origin": "Неизвестно",
                    "country_origin_code": None,
                    "country_sender": "United States",
                },
            )
        ],
    ),
]

# ---------------------------------------------------------------------------
# Header extraction — shared context injected into every LangExtract chunk
# ---------------------------------------------------------------------------

# Fields that must be propagated from the header to all items
_HEADER_FIELDS = ("document_date", "document_number", "country_sender",
                  "currency_code", "currency_name", "country_origin")

# Patterns to scan the FULL document text for global metadata that lives in
# the footer (e.g. "СТРАНА ПРОИСХОЖДЕНИЯ: КИТАЙ", "Код валюты: 840").
_FOOTER_PATTERNS: dict[str, re.Pattern] = {
    "country_origin": re.compile(
        # Capture 1–3 "words" (letters + hyphens only).
        # The negative lookahead inside the optional group prevents swallowing
        # sentence-starter words that immediately follow the country name on
        # the same OCR line (e.g. "КИТАЙ Итого ДВЕСТИ...").
        # When those words aren't present the {0,2} cap limits over-capture.
        r"страна\s+происхождения\s*:?\s*"
        r"([А-Яа-яёЁA-Za-z][А-Яа-яёЁA-Za-z-]*"
        r"(?:\s+(?!итого\b|total\b|сумма\b|всего\b|\d)"
        r"[А-Яа-яёЁA-Za-z][А-Яа-яёЁA-Za-z-]*){0,2})",
        re.IGNORECASE,
    ),
    "currency_code": re.compile(
        r"код\s+валюты\s*:?\s*(\d{3})",
        re.IGNORECASE,
    ),
}

# ISO 4217 numeric → alpha-3 for the currencies most common in CIS trade docs
_ISO4217_NUMERIC_TO_ALPHA3: dict[str, str] = {
    "840": "USD", "978": "EUR", "156": "CNY", "643": "RUB",
    "417": "KGS", "398": "KZT", "860": "UZS", "826": "GBP",
    "392": "JPY", "756": "CHF", "036": "AUD", "124": "CAD",
}

# Regex patterns to parse header metadata directly from OCR text
_HEADER_PATTERNS = {
    "document_number": re.compile(
        r"(?:invoice[ \t]*(?:no\.?|num\.?|#|:)|№|накладн|счет)[ \t.:,]*([A-Z0-9\-/]{4,30})",
        re.IGNORECASE,
    ),
    "document_date": re.compile(
        r"(?:date|дата|от)\W*(\d{1,2}[\/\.\-]\d{1,2}[\/\.\-]\d{2,4})",
        re.IGNORECASE,
    ),
    "country_sender": re.compile(
        # "sender" removed — too generic (matches "Temperature sender", "Pressure sender", etc.)
        r"\b(?:отправитель|shipper|страна\s+отправ\w*)\b\W*:?\s*([A-Za-zА-Яа-яёЁ ]{3,40})",
        re.IGNORECASE,
    ),
}


# Признак строки с товарной позицией: строка начинается с порядкового номера
# (цифра + пробел/таб) И содержит числовую цену (1234.56 или 1234,56).
# Также матчит pipe-delimited строки вида "| 1 | ..." или "| 107 | ...".
# Такие строки НЕ включаются в шапку.
_ITEM_ROW_START_RE = re.compile(
    r'(?:'
    r'^\s*\d+[\s\t]+\S'       # bare format:  "1  Description" or "107\tSomething"
    r'|^\s*\|\s*\d+\s*\|'     # piped format: "| 1 | ..." or "| 107 | ..."
    r')',
    re.MULTILINE,
)
_PRICE_RE          = re.compile(r'\b\d+[.,]\d{2}\b')

# Признак строки-заголовка таблицы: содержит ключевые слова названий колонок.
# Такую строку ВКЛЮЧАЕМ в шапку, но после неё останавливаемся — она даёт ЛЛМ
# контекст структуры колонок без утечки самих товарных позиций.
_TABLE_HEADER_RE = re.compile(
    r'\b(?:наименование|кол[-.\s]?во|количество|стоимость|ед[.\s]?изм'
    r'|unit\b|qty\b|quantity\b|amount\b|description\b|тн\s*вэд|hs\s*code'
    r'|product\b|item\b|total\b|value\b|price\b|count\b|rate\b)',
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



def parse_full_doc_metadata(context: str) -> dict:
    """
    Scan the FULL cleaned document text for global metadata that lives in the
    footer rather than the header (e.g. СТРАНА ПРОИСХОЖДЕНИЯ, Код валюты).

    Returns a dict with found values only.  Numeric currency codes are mapped
    to ISO 4217 alpha-3 via _ISO4217_NUMERIC_TO_ALPHA3.
    """
    meta: dict = {}
    for field, pattern in _FOOTER_PATTERNS.items():
        m = pattern.search(context)
        if not m:
            continue
        value = m.group(1).strip().rstrip(",;.")
        if not value:
            continue
        if field == "currency_code":
            value = _ISO4217_NUMERIC_TO_ALPHA3.get(value, value)
        meta[field] = value
    return meta


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
                or str(current).strip().lower() in (
                    "", "null", "none", "0", "неизвестно", "unknown",
                )
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
# Single-country spread — fill unknown origins when the whole invoice shares
# one country of origin
# ---------------------------------------------------------------------------

_UNKNOWN_ORIGIN = frozenset({"неизвестно", "unknown", "не указано", "null", "none", ""})


def spread_single_country_origin(items: list[dict]) -> list[dict]:
    """
    If every item that has a known country_origin carries the SAME value,
    copy it to items whose country_origin is missing/unknown.

    Safety rule: if two or more DISTINCT known countries exist in the list,
    do nothing — the invoice has per-item origin data and spreading would
    corrupt it.
    """
    known = {
        str(item.get("country_origin", "")).strip()
        for item in items
        if str(item.get("country_origin", "")).strip().lower()
        not in _UNKNOWN_ORIGIN
    }
    if len(known) != 1:
        # 0 known → nothing to spread; 2+ → mixed origins, don't touch
        return items

    single = next(iter(known))
    for item in items:
        val = str(item.get("country_origin", "")).strip()
        if val.lower() in _UNKNOWN_ORIGIN:
            item["country_origin"] = single
    return items


# ---------------------------------------------------------------------------
# Deduplication — removes duplicate rows from bilingual invoices
# ---------------------------------------------------------------------------

def deduplicate_items(items: list[dict]) -> list[dict]:
    """
    Remove cross-chunk duplicate rows.

    --- Why (position, cost, quantity) and not position alone ---
    Some invoices contain multiple sub-tables, each numbered from 1.
    E.g. sub-invoice 1 POS 1 = "БПЛА 37 410 EUR" and
         sub-invoice 2 POS 1 = "Комплект шасси 3 500 EUR".
    Keying only on position would silently drop one of them.
    Adding cost + quantity makes the key unique per genuinely distinct item.

    --- Why NOT add description to the key ---
    Bilingual invoices: Russian chunk and English chunk carry the same item
    under different descriptions ("Кронштейн" vs "Console").  If description
    were part of the key they would never merge.  Description is used only as
    a tie-breaker (Russian preferred over English), not as a discriminator.

    --- HS-code conflict guard ---
    If two items share (position, cost, quantity) but carry different non-null
    HS codes, they are treated as genuinely different items — both kept.
    This guards against edge-cases where two cheap commodity items coincidentally
    share the same pos/cost/qty combination.

    --- Field-level merge ---
    When a true duplicate is found, fields are merged individually rather than
    replacing the whole object:
      - description : prefer Russian over English
      - hs_code, country_* , currency_*, document_* : prefer non-null
      - cost / price / quantity : prefer non-zero
    This captures the best information from both chunks.
    """
    if not items:
        return items

    _EMPTY = {"", "null", "none"}

    def _is_empty(v) -> bool:
        return v is None or str(v).strip().lower() in _EMPTY

    def _is_cyrillic(text: str) -> bool:
        return bool(re.search(r'[А-Яа-яёЁ]', str(text or "")))

    def _norm_num(v, decimals: int = 2) -> float:
        try:
            return round(float(v or 0), decimals)
        except (TypeError, ValueError):
            return 0.0

    def _norm_hs(v) -> str | None:
        if _is_empty(v):
            return None
        return str(v).strip()

    def _make_key(item: dict) -> tuple | None:
        raw_pos = item.get("position")
        try:
            pos = int(raw_pos) if raw_pos is not None else None
        except (TypeError, ValueError):
            pos = None
        if pos is None:
            return None
        return (pos, _norm_num(item.get("cost")), _norm_num(item.get("quantity"), 3))

    def _hs_conflict(a: dict, b: dict) -> bool:
        """True when both items have non-null, differing HS codes."""
        ha, hb = _norm_hs(a.get("hs_code")), _norm_hs(b.get("hs_code"))
        return ha is not None and hb is not None and ha != hb

    def _merge(base: dict, new: dict) -> dict:
        """Merge *new* into a copy of *base*, field by field."""
        out = dict(base)
        # description: prefer Russian
        if _is_cyrillic(new.get("description", "")) and \
                not _is_cyrillic(out.get("description", "")):
            out["description"] = new["description"]
        # nullable metadata: take first non-empty value
        for f in ("hs_code", "country_origin", "country_origin_code",
                  "currency_code", "currency_name",
                  "document_date", "document_number", "country_sender"):
            if _is_empty(out.get(f)) and not _is_empty(new.get(f)):
                out[f] = new[f]
        # numeric fields: prefer non-zero
        for f in ("cost", "price", "quantity"):
            if _norm_num(out.get(f)) == 0.0 and _norm_num(new.get(f)) != 0.0:
                out[f] = new[f]
        return out

    seen: dict[tuple, int] = {}   # dedup_key → index in result
    result: list[dict] = []

    for item in items:
        key = _make_key(item)

        if key is None:
            # No parseable position — pass through unchanged
            result.append(item)
            continue

        if key in seen:
            existing_idx = seen[key]
            existing = result[existing_idx]

            if _hs_conflict(item, existing):
                # Different items that accidentally share pos+cost+qty
                # Keep both — do NOT update `seen` so the first item's slot
                # remains reachable for future merges of its own duplicates.
                result.append(item)
            else:
                # True cross-chunk duplicate — merge fields
                result[existing_idx] = _merge(existing, item)
        else:
            seen[key] = len(result)
            result.append(item)

    return result




# ---------------------------------------------------------------------------
# Step 1 — text cleaning  (mirrors Node 1765260281035)
# ---------------------------------------------------------------------------

def _normalize_pipe_table(text: str) -> str:
    """
    Normalize OCR invoice text before chunking and LLM extraction.

    Fixes two issues that cause truncation on multi-sub-table invoices:

    1. Single-line OCR — some systems emit the entire Markdown table as one
       long line.  _split_text_into_chunks splits only on \\n, so the whole
       invoice lands in one oversized chunk.  The LLM is then forced to
       generate a very long JSON response and may stop early (output-token
       limit) or misinterpret the structure.

       Fix: insert \\n before every "| N | LETTER" pattern — numbered item
       rows whose description starts with a letter.  This distinguishes item-
       row boundaries from price/quantity cells (which never have a letter
       immediately after the closing | of a digit cell).

    2. Sub-table separator rows — multi-page invoices contain horizontal
       |---|---| dividers at page-break boundaries between sub-tables.
       LLMs trained on Markdown interpret these as "end of table" and stop
       extracting after the separator.  On the sample invoice this caused
       extraction to stop at position 26 (end of the first sub-table).

       Fix: (a) move any trailing separator sequence at the end of a content
       line onto its own line, then (b) delete lines that consist solely of
       |, - and spaces (no letters or digits).

    The transform is idempotent: already-multiline OCR with no separator rows
    passes through unchanged.
    """
    # Step 1 — split single-line tables into one row per line.
    #
    # Lookbehind: right after a | (end of previous row / separator).
    # Consumed:   optional spaces/tabs between rows.
    # Lookahead:  "| N | LETTER" — row-number cell followed by a description
    #             cell whose first character is a letter.
    #
    # Why the letter requirement?  Price cells like "| 900 |" or "| 150 |"
    # must NOT trigger a split — they share the same pattern (| digit |) but
    # are followed by another digit or end-of-row, never by a letter.
    text = re.sub(
        r'(?<=\|)[ \t]*(?=\|[ \t]*\d{1,4}[ \t]*\|[ \t]*[А-Яа-яёЁA-Za-z])',
        '\n',
        text,
    )

    # Step 2 — extract trailing separator sequences to their own line.
    # After step 1 the first-page header line may still end with the column-
    # header separator: "...Сумма USD | |----|----|".
    # Replace "| |---...----|" at end-of-line with just "|" so step 3 can
    # remove the now-isolated separator line.
    text = re.sub(r'[ \t]*\|[- \t|]+\|[ \t]*$', '|', text, flags=re.MULTILINE)

    # Step 3 — remove standalone separator lines.
    # A separator line contains ONLY pipes, dashes, and whitespace — no
    # letters and no digits.  These mark page breaks between sub-tables.
    text = re.sub(r'^[|\- \t]+$', '', text, flags=re.MULTILINE)

    # Step 4 — collapse blank lines left after separator removal.
    text = re.sub(r'\n{2,}', '\n', text)

    return text


def clean_text(raw_text: str, currency_db_str: str) -> str:
    """Clean OCR text and prepend the currency database block."""
    if not raw_text:
        return ""

    text = raw_text.replace("\r", "")
    # Normalise pipe-table OCR: split single-line tables into rows and remove
    # sub-table separator lines (|---|---| page-break markers) that cause the
    # LLM to stop extracting after the first sub-table.
    text = _normalize_pipe_table(text)
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
        parsed_data = parsed_data.get("items", parsed_data.get("extractions", [parsed_data]))
    
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
        model_id:   Optional model override.  Accepts a profile alias ("gpt-oss",
                    "cerebras", "gemini") or any raw model ID string.  When None,
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
    _fb = getattr(settings, "LLM_MODEL_FALLBACK", "gemini-2.5-flash")
    fallback_model = MODEL_PROFILES.get(_fb, _fb)
    # Tracks the model that actually produced the accepted result
    effective_model = primary_model

    if not context:
        m.t_total_s = time.perf_counter() - t_wall_start
        return {"error": "Empty OCR text", "metrics": m.to_dict(), "model_id": effective_model}

    # Extract header for additional_context injection into every chunk
    header_context = extract_header(context)
    # Parse header metadata for post-fill fallback
    header_meta = parse_header_metadata(header_context)
    # Scan the full document for global metadata in the footer (country of
    # origin, numeric currency code) and merge — header takes precedence.
    for k, v in parse_full_doc_metadata(context).items():
        if k not in header_meta:
            header_meta[k] = v

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
        # Spread a single global country_origin to items that are still unknown.
        # Must run BEFORE finalize_items so country_origin_code gets filled too.
        items = spread_single_country_origin(items)
        final_items = finalize_items(items, currency_db)
        # Remove cross-chunk duplicates: same invoice position extracted from
        # two overlapping chunks.  Keyed on the `position` field the model
        # reads from the №/POS column — different positions with identical
        # data (e.g. POS 7 and POS 8 brackets) are NOT merged.
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
        "raw_llm_output": raw_output,     # JSON str from LangExtract — for prompt debugging
    }


# ---------------------------------------------------------------------------
# Model profiles — friendly aliases for concrete model IDs
# ---------------------------------------------------------------------------

# Edit this dict to add/rename profiles. Keys are the strings callers can pass
# in the POST body as {"model": "cerebras"}.  Values are the actual model IDs sent
# to the respective provider.  You can also pass a raw model ID directly (e.g.
# "gemini-2.5-flash") — resolve_model_id() will use it verbatim.
MODEL_PROFILES: dict[str, str] = {
    "cerebras":  "gpt-oss-120b",     # GPT-OSS 120B via Cerebras (3k tok/s)
    "gemini":    "gemini-2.5-flash", # Google Gemini 2.5 Flash (fallback)
}


def resolve_model_id(model_spec: str | None) -> str:
    """
    Resolve a model spec to a concrete model ID.

    Resolution order:
      1. None / ""       → LLM_MODEL_PRIMARY from Django settings (alias-resolved)
      2. Profile alias   → MODEL_PROFILES[model_spec]
      3. Raw model ID    → used verbatim (e.g. "gpt-oss-120b", "gemini-2.5-flash")
    """
    if not model_spec:
        primary = getattr(settings, "LLM_MODEL_PRIMARY", "gpt-oss-120b")
        return MODEL_PROFILES.get(primary, primary)
    return MODEL_PROFILES.get(model_spec, model_spec)


# ---------------------------------------------------------------------------
# Provider resolution — Cerebras and Gemini
# ---------------------------------------------------------------------------

def _build_lx_config(model_id: str) -> lx_factory.ModelConfig:
    """
    Build a LangExtract ModelConfig:
      - gpt-oss-120b  → Cerebras cloud (CEREBRAS_BASE_URL / CEREBRAS_API_KEY)
      - gemini-*      → Google Gemini  (LANGEXTRACT_API_KEY)
    """
    if model_id.startswith("gpt-oss"):
        # Cerebras format: gpt-oss-120b
        return lx_factory.ModelConfig(
            model_id=model_id,
            provider_kwargs={
                "base_url":    getattr(settings, "CEREBRAS_BASE_URL", "https://api.cerebras.ai/v1"),
                "api_key":     getattr(settings, "CEREBRAS_API_KEY", "") or None,
                "max_workers": getattr(settings, "LLM_MAX_WORKERS_CEREBRAS", 20),
            },
        )

    # Gemini — auto-resolved by router
    return lx_factory.ModelConfig(
        model_id=model_id,
        provider_kwargs={
            "api_key":     getattr(settings, "LANGEXTRACT_API_KEY", "") or None,
            "max_workers": getattr(settings, "LLM_MAX_WORKERS_GEMINI", 5),
        },
    )


# ---------------------------------------------------------------------------
# Direct Cerebras extraction (bypasses LangExtract entirely)
# ---------------------------------------------------------------------------

def _split_text_into_chunks(text: str, max_chars: int) -> list[str]:
    """
    Split *text* into chunks of at most *max_chars* characters, always
    cutting at newline boundaries so no invoice row is split mid-way.
    """
    lines = text.split("\n")
    chunks: list[str] = []
    current: list[str] = []
    current_len = 0

    for line in lines:
        line_len = len(line) + 1  # +1 for the rejoining newline
        if current_len + line_len > max_chars and current:
            chunks.append("\n".join(current))
            current = [line]
            current_len = line_len
        else:
            current.append(line)
            current_len += line_len

    if current:
        chunks.append("\n".join(current))

    return chunks


def extract_cerebras_direct(
    context_text: str,
    model_id: str,
    header_context: str = "",
) -> tuple[str, None]:
    """
    Native Cerebras SDK with parallel chunking — no LangExtract, no resolver.

    Why native SDK (not openai + base_url):
      - Auto-retry with exponential backoff on 429 / 5xx.  The old openai
        approach silently caught 429s as empty "[]", causing chunks 2-N to
        disappear under rate limits on large invoices.
      - response_format JSON schema: model is FORCED to emit valid {"items":[…]},
        eliminating ResolverParsingError, unquoted keys, and field-name drift
        (unit_price vs cost, total_price vs price).
      - TCP warming on client construction reduces time-to-first-token.
      - Singleton client (_get_cerebras_client) reuses the HTTP pool.

    Why chunking:
      Sending 200+ items in one shot → model stops early.  5 000-char chunks
      (~40-50 items each) keep each call focused; parallel workers finish the
      whole invoice in roughly single-call wall time.

    Tuning via .env:
      LLM_MAX_CHAR_BUFFER_CEREBRAS  (default 5 000)
      LLM_MAX_WORKERS_CEREBRAS      (default 20)

    Returns (json_str, None) — None because there is no AnnotatedDocument.
    """
    import concurrent.futures

    buffer      = getattr(settings, "LLM_MAX_CHAR_BUFFER_CEREBRAS", 5000)
    max_workers = getattr(settings, "LLM_MAX_WORKERS_CEREBRAS", 20)

    chunks    = _split_text_into_chunks(context_text, buffer)
    n_workers = min(len(chunks), max_workers)

    logger.debug(
        "[cerebras_direct] model=%s  total_chars=%d  chunks=%d  workers=%d",
        model_id, len(context_text), len(chunks), n_workers,
    )

    client = _get_cerebras_client()

    def _call_chunk(idx_chunk: tuple[int, str]) -> tuple[int, str]:
        idx, chunk = idx_chunk
        # Label each chunk so the model knows it's extracting invoice items
        labelled = (
            f"=== INVOICE ITEMS — SECTION {idx + 1} OF {len(chunks)} ===\n{chunk}"
        )
        user_content = f"{header_context}\n\n{labelled}" if header_context else labelled
        try:
            resp = client.chat.completions.create(
                model=model_id,
                messages=[
                    {"role": "system", "content": EXTRACTION_PROMPT_GPT_OSS},
                    {"role": "user",   "content": user_content},
                ],
                temperature=0,
                response_format=_CEREBRAS_RESPONSE_FORMAT,
                # Raise the output-token ceiling so the model never stops
                # mid-array on large invoices (default ~4 096 is too low for
                # 50+ items × ~150 tokens/item).
                max_tokens=32768,
            )
            raw = resp.choices[0].message.content or '{"items":[]}'
            # Log Cerebras-native per-chunk latency breakdown
            if hasattr(resp, "time_info") and resp.time_info:
                ti = resp.time_info
                logger.debug(
                    "[cerebras_direct] chunk %d  queue=%.3fs  prompt=%.3fs  completion=%.3fs  total=%.3fs",
                    idx,
                    getattr(ti, "queue_time", 0) or 0,
                    getattr(ti, "prompt_process_time", 0) or 0,
                    getattr(ti, "completion_time", 0) or 0,
                    getattr(ti, "total_time", 0) or 0,
                )
            return idx, raw
        except Exception as exc:
            logger.warning(
                "[cerebras_direct] chunk %d failed (%s: %s)",
                idx, type(exc).__name__, exc,
                exc_info=True,
            )
            return idx, '{"items":[]}'

    # pool.map preserves submission order and blocks until all chunks finish
    all_items: list[dict] = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=n_workers) as pool:
        for idx, raw in pool.map(_call_chunk, enumerate(chunks)):
            parsed = None
            for candidate in (raw, _repair_json(raw)):
                try:
                    parsed = json.loads(candidate)
                    break
                except json.JSONDecodeError:
                    continue
            if parsed is None:
                logger.warning("[cerebras_direct] chunk %d: JSON parse failed, skipping", idx)
                continue
            # Unwrap {"items":[]} (expected with schema) or bare array (fallback)
            if isinstance(parsed, dict):
                parsed = parsed.get("items", parsed.get("extractions", [parsed]))
            if isinstance(parsed, list):
                all_items.extend(parsed)

    combined = json.dumps(all_items, ensure_ascii=False)
    logger.debug(
        "[cerebras_direct] model=%s  chunks=%d  raw_items=%d  preview=%.400s",
        model_id, len(chunks), len(all_items), combined[:400],
    )
    return combined, None


# ---------------------------------------------------------------------------
# Оптимизированный вызов извлечения
# ---------------------------------------------------------------------------

def extract_with_langextract_optimized(
    context_text: str,
    model_id: str,
    header_context: str = "",
) -> tuple[str, object]:
    """
    Run extraction and return (json_str, annotated_doc).

    Routing:
      - gpt-oss-120b → Cerebras direct API (bypasses LangExtract — see extract_cerebras_direct)
      - gemini-*     → Google Gemini via LangExtract (4k char buffer, N workers)

    The second element is an ``lx.data.AnnotatedDocument`` (None for Cerebras direct).
    """
    if model_id.startswith("gpt-oss"):
        # Cerebras: gpt-oss-120b — bypass LangExtract, call API directly
        return extract_cerebras_direct(context_text, model_id, header_context)

    config = _build_lx_config(model_id)

    # Gemini
    prompt = EXTRACTION_PROMPT
    buffer = getattr(settings, "LLM_MAX_CHAR_BUFFER", 5000)

    try:
        annotated_doc = lx.extract(
            text_or_documents=context_text,
            prompt_description=prompt,
            examples=EXAMPLES,
            config=config,
            additional_context=header_context or None,
            max_char_buffer=buffer,
        )
    except Exception as exc:
        logger.warning(
            "lx.extract() failed (%s: %s) — returning empty result",
            type(exc).__name__, exc,
            exc_info=True,   # full traceback in server logs
        )
        return "[]", None

    all_items = []
    for extraction in annotated_doc.extractions:
        item = {"description": extraction.extraction_text}
        if extraction.attributes:
            item.update(extraction.attributes)
        all_items.append(item)

    logger.debug(
        "[raw_output] model=%s  extracted=%d items  preview=%s",
        model_id,
        len(all_items),
        json.dumps(all_items[:3], ensure_ascii=False)[:500],
    )

    return json.dumps(all_items, ensure_ascii=False), annotated_doc
