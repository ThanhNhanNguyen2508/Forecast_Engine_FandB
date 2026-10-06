from __future__ import annotations

import math
import re
import unicodedata
from datetime import date, datetime
from typing import Any, Literal

import pandas as pd


TRUE_VALUES = {"1", "true", "yes", "y", "co", "có", "x"}
FALSE_VALUES = {"0", "false", "no", "n", "khong", "không"}
UNIT_ALIASES = {
    "kg": "kg", "kilogram": "kg", "kilograms": "kg",
    "g": "g", "gram": "g", "grams": "g",
    "l": "liter", "liter": "liter", "litre": "liter", "liters": "liter",
    "litres": "liter", "lít": "liter", "lit": "liter",
    "ml": "ml", "milliliter": "ml", "millilitre": "ml",
    "cái": "unit", "cai": "unit", "piece": "unit", "pieces": "unit",
    "unit": "unit", "ly": "unit", "cup": "unit",
    "thùng": "pack", "thung": "pack", "bao": "pack", "gói": "pack",
    "goi": "pack", "chai": "pack", "pack": "pack",
}


class AmbiguousValue(ValueError):
    pass


def clean_text(value: Any) -> str | None:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return None
    text = unicodedata.normalize("NFKC", str(value)).strip()
    return text or None


def normalized_entity(value: Any) -> str:
    text = (clean_text(value) or "").casefold().replace("đ", "d")
    text = "".join(c for c in unicodedata.normalize("NFKD", text) if not unicodedata.combining(c))
    return re.sub(r"[^a-z0-9]+", " ", text).strip()


def parse_boolean(value: Any) -> bool | None:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return None
    if isinstance(value, bool):
        return value
    normalized = (clean_text(value) or "").casefold()
    if normalized in TRUE_VALUES:
        return True
    if normalized in FALSE_VALUES:
        return False
    raise AmbiguousValue(f"BOOLEAN_AMBIGUOUS:{value!r}")


def parse_number(value: Any) -> float | None:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return None
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return float(value)
    text = (clean_text(value) or "").replace(" ", "")
    if not text:
        return None
    comma, dot = text.count(","), text.count(".")
    if comma and dot:
        # The right-most separator is decimal; the other is thousands.
        decimal = "," if text.rfind(",") > text.rfind(".") else "."
        thousands = "." if decimal == "," else ","
        text = text.replace(thousands, "").replace(decimal, ".")
    elif comma:
        tail = len(text) - text.rfind(",") - 1
        if comma > 1 or tail == 3:
            raise AmbiguousValue(f"NUMBER_SEPARATOR_AMBIGUOUS:{value!r}")
        text = text.replace(",", ".")
    elif dot and (dot > 1 or len(text) - text.rfind(".") - 1 == 3):
        raise AmbiguousValue(f"NUMBER_SEPARATOR_AMBIGUOUS:{value!r}")
    try:
        return float(text)
    except ValueError as exc:
        raise AmbiguousValue(f"NUMBER_INVALID:{value!r}") from exc


def parse_date(value: Any, locale: Literal["DMY", "MDY", "YMD"] | None) -> date | None:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return None
    if isinstance(value, (pd.Timestamp, datetime, date)):
        return pd.Timestamp(value).date()
    text = clean_text(value)
    if text is None:
        return None
    if re.fullmatch(r"\d{4}[-/]\d{1,2}[-/]\d{1,2}", text):
        return pd.to_datetime(text, yearfirst=True, errors="raise").date()
    match = re.fullmatch(r"(\d{1,2})[/-](\d{1,2})[/-](\d{4})", text)
    if match:
        first, second = int(match.group(1)), int(match.group(2))
        if first <= 12 and second <= 12 and locale is None:
            raise AmbiguousValue(f"DATE_LOCALE_AMBIGUOUS:{value!r}")
        if locale is None:
            locale = "DMY" if first > 12 else "MDY"
        dayfirst = locale == "DMY"
        return pd.to_datetime(text, dayfirst=dayfirst, errors="raise").date()
    parsed = pd.to_datetime(text, errors="coerce")
    if pd.isna(parsed):
        raise AmbiguousValue(f"DATE_INVALID:{value!r}")
    return parsed.date()


def normalize_unit(value: Any) -> str | None:
    text = clean_text(value)
    if text is None:
        return None
    return UNIT_ALIASES.get(text.casefold(), text.casefold())
