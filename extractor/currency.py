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


def resolve_country(raw: str | None) -> str | None:
    """Map ISO-2 country code to Russian name. Returns original value if not mapped."""
    if not raw:
        return raw
    code = str(raw).strip().upper()
    return _ISO_TO_RUSSIAN.get(code, raw)


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
        raw_origin = item.get("country_origin")
        if raw_origin and len(str(raw_origin).strip()) <= 3:
            mapped = resolve_country(raw_origin)
            if mapped and mapped != raw_origin:
                item["country_origin"] = mapped

        # Ensure hs_code and suggestions
        if "hs_code" not in item:
            item["hs_code"] = None
        item["suggestions"] = item.get("suggestions", [])

        final.append(item)

    return final
