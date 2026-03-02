"""
Currency utilities — mirrors the resolve_currency logic from the original
'Checking and sending request' code node (Node 1765342175106).
"""

import json
import re
from django.conf import settings


def load_currency_db() -> list[dict]:
    """Load currency database from Django settings (CURRENCY_DB_JSON env var)."""
    raw = getattr(settings, "CURRENCY_DB_JSON", "[]")
    try:
        return json.loads(raw)
    except (json.JSONDecodeError, TypeError):
        return []


def resolve_currency(raw: str | None, db: list[dict]) -> tuple[str | None, str | None]:
    """
    Return (currency_code, currency_name) by looking up *raw* in the DB.
    Matches first by code, then by name substring — same logic as original workflow.
    """
    if not raw or not db:
        return None, None

    key = str(raw).strip().upper()

    for row in db:
        if str(row.get("code", "")).upper() == key:
            return row["code"], row.get("name")

    for row in db:
        if key in str(row.get("name", "")).upper():
            return row["code"], row.get("name")

    return raw, None


def build_currency_db_string(db: list[dict]) -> str:
    """
    Serialise the currency DB to the plain-text block expected by the LLM prompt
    (mirrors the =CURRENCY DATABASE= section prepended in Node 1765260281035).
    """
    return json.dumps(db, ensure_ascii=False, indent=2)


_ISO_TO_RUSSIAN: dict[str, str] = {
    "AF": "Афганистан", "AL": "Албания", "DZ": "Алжир", "AR": "Аргентина",
    "AM": "Армения", "AU": "Австралия", "AT": "Австрия", "AZ": "Азербайджан",
    "BY": "Беларусь", "BE": "Бельгия", "BR": "Бразилия", "BG": "Болгария",
    "CA": "Канада", "CL": "Чили", "CN": "Китай", "CO": "Колумбия",
    "HR": "Хорватия", "CZ": "Чехия", "DK": "Дания", "EG": "Египет",
    "EE": "Эстония", "FI": "Финляндия", "FR": "Франция", "GE": "Грузия",
    "DE": "Германия", "GR": "Греция", "HK": "Гонконг", "HU": "Венгрия",
    "IN": "Индия", "ID": "Индонезия", "IR": "Иран", "IQ": "Ирак",
    "IE": "Ирландия", "IL": "Израиль", "IT": "Италия", "JP": "Япония",
    "KZ": "Казахстан", "KG": "Кыргызстан", "LV": "Латвия", "LT": "Литва",
    "LU": "Люксембург", "MY": "Малайзия", "MX": "Мексика", "MD": "Молдова",
    "NL": "Нидерланды", "NZ": "Новая Зеландия", "NO": "Норвегия",
    "PK": "Пакистан", "PE": "Перу", "PH": "Филиппины", "PL": "Польша",
    "PT": "Португалия", "RO": "Румыния", "RU": "Россия", "SA": "Саудовская Аравия",
    "RS": "Сербия", "SG": "Сингапур", "SK": "Словакия", "SI": "Словения",
    "ZA": "Южная Африка", "KR": "Южная Корея", "ES": "Испания", "SE": "Швеция",
    "CH": "Швейцария", "TW": "Тайвань", "TJ": "Таджикистан", "TH": "Таиланд",
    "TN": "Тунис", "TR": "Турция", "TM": "Туркменистан", "UA": "Украина",
    "GB": "Великобритания", "US": "США", "UZ": "Узбекистан", "VN": "Вьетнам",
}

# ISO 3166-1 numeric codes (mirrors the ISO-2 keys above)
_ISO_TO_NUMERIC: dict[str, int] = {
    "AF": 4,   "AL": 8,   "DZ": 12,  "AR": 32,  "AM": 51,  "AU": 36,
    "AT": 40,  "AZ": 31,  "BY": 112, "BE": 56,  "BR": 76,  "BG": 100,
    "CA": 124, "CL": 152, "CN": 156, "CO": 170, "HR": 191, "CZ": 203,
    "DK": 208, "EG": 818, "EE": 233, "FI": 246, "FR": 250, "GE": 268,
    "DE": 276, "GR": 300, "HK": 344, "HU": 348, "IN": 356, "ID": 360,
    "IR": 364, "IQ": 368, "IE": 372, "IL": 376, "IT": 380, "JP": 392,
    "KZ": 398, "KG": 417, "LV": 428, "LT": 440, "LU": 442, "MY": 458,
    "MX": 484, "MD": 498, "NL": 528, "NZ": 554, "NO": 578, "PK": 586,
    "PE": 604, "PH": 608, "PL": 616, "PT": 620, "RO": 642, "RU": 643,
    "SA": 682, "RS": 688, "SG": 702, "SK": 703, "SI": 705, "ZA": 710,
    "KR": 410, "ES": 724, "SE": 752, "CH": 756, "TW": 158, "TJ": 762,
    "TH": 764, "TN": 788, "TR": 792, "TM": 795, "UA": 804, "GB": 826,
    "US": 840, "UZ": 860, "VN": 704,
}

# Reverse lookup: Russian name → numeric code (built once at import time)
_RUSSIAN_TO_NUMERIC: dict[str, int] = {
    rus.lower(): _ISO_TO_NUMERIC[iso]
    for iso, rus in _ISO_TO_RUSSIAN.items()
    if iso in _ISO_TO_NUMERIC
}


def resolve_country(raw: str | None) -> str | None:
    """Map ISO-2 country code to Russian name. Returns original value if not mapped."""
    if not raw:
        return raw
    code = str(raw).strip().upper()
    return _ISO_TO_RUSSIAN.get(code, raw)


def resolve_country_code(raw: str | None) -> int | None:
    """
    Return ISO 3166-1 numeric code for a country given either:
      - ISO-2 code  ("SE"      → 752)
      - Russian name ("Швеция" → 752)
    Returns None if not found.
    """
    if not raw:
        return None
    s = str(raw).strip()
    # Try ISO-2 first (short string)
    if len(s) <= 3:
        numeric = _ISO_TO_NUMERIC.get(s.upper())
        if numeric is not None:
            return numeric
    # Try Russian name (case-insensitive)
    return _RUSSIAN_TO_NUMERIC.get(s.lower())


def finalize_items(items: list[dict], currency_db: list[dict]) -> list[dict]:
    """
    Post-process extracted items:
      - normalise whitespace in description
      - resolve currency code/name from DB
      - map ISO country code → Russian name in country_origin
      - ensure hs_code and suggestions fields are present
    Mirrors Node 1765342175106 logic.
    """
    final = []
    for item in items:
        # Clean description
        desc = item.get("description", "")
        desc = re.sub(r"\s+", " ", desc).strip()
        item["description"] = desc

        # Resolve currency
        raw_val = item.get("currency_code") or item.get("currency_name")
        ccode, cname = resolve_currency(raw_val, currency_db)
        if ccode:
            item["currency_code"] = ccode
            item["currency_name"] = cname or ccode

        # Map ISO country code → Russian name (e.g. "DE" → "Германия")
        # and fill country_origin_code when missing/null
        raw_origin = item.get("country_origin")
        if raw_origin and len(str(raw_origin).strip()) <= 3:
            mapped = resolve_country(raw_origin)
            if mapped and mapped != raw_origin:
                item["country_origin"] = mapped

        # Fill numeric country code if null/missing — works for both ISO-2 and
        # Russian names (covers the case where the model set "SE" but forgot the code,
        # or already converted to "Швеция" without a numeric code)
        existing_code = item.get("country_origin_code")
        if existing_code is None or str(existing_code).strip().lower() in ("", "null", "none"):
            resolved_code = resolve_country_code(raw_origin)
            if resolved_code is None:
                # raw_origin may already be a Russian name — try the (possibly mapped) value
                resolved_code = resolve_country_code(item.get("country_origin"))
            if resolved_code is not None:
                item["country_origin_code"] = resolved_code

        # Ensure hs_code and suggestions
        if "hs_code" not in item:
            item["hs_code"] = None
        item["suggestions"] = item.get("suggestions", [])

        final.append(item)

    return final
