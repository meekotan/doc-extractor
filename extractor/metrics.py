"""
Run-level metrics: timing per pipeline step + accuracy indicators.

Collected inside run_invoice_extraction() and stored on ExtractionJob.metrics.
"""

import time
from contextlib import contextmanager
from dataclasses import dataclass, field, asdict


@dataclass
class RunMetrics:
    # ── Timing (seconds, 3 dp) ──────────────────────────────────────────────
    t_clean_s: float = 0.0          # Step 1 – text cleaning
    t_primary_llm_s: float = 0.0    # Step 2 – primary LLM call
    t_validate_s: float = 0.0       # Step 3 – JSON validation
    t_fallback_llm_s: float = 0.0   # Step 4 – fallback LLM call (0 if unused)
    t_finalize_s: float = 0.0       # Step 5 – currency resolution
    t_total_s: float = 0.0          # Wall-clock total

    # ── Accuracy / quality indicators ───────────────────────────────────────
    items_extracted: int = 0        # Number of items in final output
    fallback_used: bool = False     # True if GPT-4o fallback was triggered
    primary_valid: bool = False     # True if primary LLM passed validation
    fallback_valid: bool = False    # True if fallback passed (n/a if unused)

    # Per-field fill rates (0.0–1.0 per item, averaged across all items)
    field_fill_rates: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        d = asdict(self)
        # round all timing floats for readability
        for k in list(d.keys()):
            if k.startswith("t_") and isinstance(d[k], float):
                d[k] = round(d[k], 3)
        return d


# ── Tracked fields for fill-rate calculation ────────────────────────────────
TRACKED_FIELDS = [
    "hs_code",
    "description",
    "quantity",
    "unit",
    "cost",
    "price",
    "currency_code",
    "currency_name",
    "document_date",
    "document_number",
    "country_origin",
    "country_origin_code",
    "country_sender",
]


def compute_field_fill_rates(items: list[dict]) -> dict[str, float]:
    """
    For each tracked field, compute the fraction of items where the field is
    present and non-empty / non-null.
    Returns e.g. {"hs_code": 1.0, "cost": 0.33, ...}
    """
    if not items:
        return {f: 0.0 for f in TRACKED_FIELDS}

    rates = {}
    n = len(items)
    for f in TRACKED_FIELDS:
        filled = sum(
            1 for item in items
            if item.get(f) not in (None, "", "null", "none", 0)
            # 0 is only considered empty for numeric fields that should be > 0
        )
        rates[f] = round(filled / n, 3)
    return rates


@contextmanager
def timer():
    """Context manager that yields a list; list[0] will be elapsed seconds."""
    result = [0.0]
    start = time.perf_counter()
    try:
        yield result
    finally:
        result[0] = time.perf_counter() - start
