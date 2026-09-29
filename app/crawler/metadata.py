"""Date, academic-year and language detection helpers used by the crawler and chunker."""
from __future__ import annotations

import re
from datetime import datetime, timezone

from dateutil import parser as dateparser

from app.crawler.urls import UPLOAD_YEAR

# 2025-26, 2025-2026, 2025/26, "A.Y. 2025-26"
_ACADEMIC_YEAR = re.compile(r"\b(20\d{2})\s*[-–/]\s*(20)?(\d{2})\b")

# Unicode script ranges → language code (script-level guess; refined in Phase 5).
_SCRIPTS = [
    ("gu", 0x0A80, 0x0AFF),
    ("hi", 0x0900, 0x097F),  # Devanagari (Hindi/Marathi — disambiguated later)
    ("bn", 0x0980, 0x09FF),
    ("pa", 0x0A00, 0x0A7F),
    ("or", 0x0B00, 0x0B7F),
    ("ta", 0x0B80, 0x0BFF),
    ("te", 0x0C00, 0x0C7F),
    ("kn", 0x0C80, 0x0CFF),
    ("ml", 0x0D00, 0x0D7F),
    ("ur", 0x0600, 0x06FF),
]


def detect_script_language(text: str, default: str = "en") -> str:
    """Language by dominant script. Latin text → 'en'."""
    counts: dict[str, int] = {}
    latin = 0
    for ch in text[:5000]:
        cp = ord(ch)
        if ch.isascii():
            if ch.isalpha():
                latin += 1
            continue
        for lang, lo, hi in _SCRIPTS:
            if lo <= cp <= hi:
                counts[lang] = counts.get(lang, 0) + 1
                break
    if not counts:
        return default if latin == 0 else "en"
    lang, n = max(counts.items(), key=lambda kv: kv[1])
    return lang if n >= latin * 0.3 else "en"


def find_academic_year(*texts: str) -> str | None:
    """First plausible academic year like '2025-26' in the given texts (title first)."""
    for text in texts:
        if not text:
            continue
        for m in _ACADEMIC_YEAR.finditer(text[:4000]):
            start, end2 = int(m.group(1)), int(m.group(3))
            if (start + 1) % 100 == end2 and 2000 <= start <= 2100:
                return f"{start}-{end2:02d}"
    return None


def parse_date(value: str | None) -> datetime | None:
    if not value:
        return None
    try:
        dt = dateparser.parse(value)
    except (ValueError, OverflowError, TypeError):
        return None
    if dt is None:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt


def date_from_upload_path(url: str) -> datetime | None:
    m = UPLOAD_YEAR.search(url)
    if not m:
        return None
    return datetime(int(m.group(1)), int(m.group(2)), 1, tzinfo=timezone.utc)


_TEXT_DATE = re.compile(
    r"\b(?:date[d]?\s*[:\-]?\s*)(\d{1,2}[./-]\d{1,2}[./-](?:20)?\d{2}|\d{1,2}(?:st|nd|rd|th)?\s+[A-Za-z]{3,9},?\s+20\d{2})",
    re.IGNORECASE,
)


def date_from_text(text: str) -> datetime | None:
    """A 'Date: 12/06/2026'-style date near the top of a notice."""
    m = _TEXT_DATE.search(text[:1500])
    if not m:
        return None
    try:
        dt = dateparser.parse(m.group(1), dayfirst=True)
    except (ValueError, OverflowError):
        return None
    if dt and 2000 <= dt.year <= 2100:
        return dt.replace(tzinfo=timezone.utc)
    return None


def best_date(*, meta: str | None = None, url: str = "", text: str = "") -> datetime | None:
    """Most reliable available date: page metadata > upload path > date written in the text."""
    return parse_date(meta) or date_from_upload_path(url) or date_from_text(text)
