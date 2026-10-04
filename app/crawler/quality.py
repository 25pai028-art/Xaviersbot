"""Which crawled documents should not be searched at all.

- Junk: things no student asks the chatbot about, which only crowd out real answers: the library's book
  catalogue (a third of all chunks on its own), feedback-form results (some list students' names and
  emails), and students' own internship / project / training reports and certificates.
- Superseded: an older edition of a document series that is re-issued every year (Handbook 2025-26 when
  the 2026-27 Handbook exists). Annual reports are not a series in this sense: they record history.

Both are decided again on every crawl from the stored text, so a rule change or a removed newer edition
takes effect on the next run (`scripts.crawl --index-only` applies them without crawling).
"""
from __future__ import annotations

import re
from collections import defaultdict
from difflib import SequenceMatcher
from urllib.parse import unquote, urlparse

MAX_DOCUMENT_CHARS = 1_500_000  # longer than any real notice/handbook: catalogues and data dumps

_CERTIFY = re.compile(r"this is to certify that", re.I)
_ABOUT_A_STUDENT = re.compile(
    r"\b(student|studying|semester|internship|intern\b|trainee|training|project|dissertation|"
    r"has (successfully )?(completed|undergone|worked)|worked (at|with|as))", re.I)
_INTERNSHIP = re.compile(r"\bintern(ship)?\b|training|worked|working with us", re.I)  # in a "to whom it may concern" letter
_TO_WHOM = re.compile(r"to\s+whom(so)?ever\s+it\s+may\s+concern", re.I)
_STUDENT_WORK = re.compile(
    r"(industrial\s+train+ing|internship|project|dissertation|field\s*work|summer\s+training|research)\s+"
    r"(report|project)|(research\s+project|thesis|dissertation)\s+submitted|in\s+partial\s+ful+fil+ment", re.I)
# Signs of a student's title page (a syllabus that *describes* a project report has none of them):
_ROLL = re.compile(r"roll\s*no|\b\d{2}\s*[-–]\s*[A-Z]{2,4}\s*[-–]\s*\d{2,4}\b", re.I)  # 23-PSH-067
_BY_STUDENT = re.compile(r"submit+ed\s+by|^\s*by\s*$", re.I | re.M)  # "Submitted by" / a line that is just "BY"
_SUBMITTED_TO = re.compile(r"submit+ed\s+to\b", re.I)
_CLASS = re.compile(r"\b[FST]\.?\s?Y\.?\s?(B|M)\.?\s?(A|Sc|Com)\b|\((hons|honours)\.?\)", re.I)  # TYBA, (Hons.)
_FEEDBACK_HEAD = re.compile(
    r"curriculum feedback|feedback on (the )?(\w+ )?curriculum|feedback from (the )?students|"
    r"(employer|industry|alumni|parent|student)(s|['’]s|s['’])?\s*(/\s*industry\s*)?feedback|"
    r"respondent['’]s email", re.I)


def junk_reason(text: str) -> str | None:
    """Why this document should not be searched, or None."""
    text = text or ""
    head = text[:1500]
    if len(text) > MAX_DOCUMENT_CHARS:
        return "very long list (e.g. the library book catalogue)"
    if (len(re.findall(r"strongly agree", text, re.I)) >= 20 or _FEEDBACK_HEAD.search(head)
            or ("docs.google.com/forms" in text and re.search(r"\bresponses\b", text))):
        return "feedback form responses"
    title_page = text[:800]
    if (_CERTIFY.search(head) and _ABOUT_A_STUDENT.search(head)) or (_TO_WHOM.search(head) and _INTERNSHIP.search(head)) or (
            _STUDENT_WORK.search(title_page) and (_ROLL.search(title_page) or _BY_STUDENT.search(title_page))) or (
            _SUBMITTED_TO.search(title_page) and (_ROLL.search(title_page) or _CLASS.search(title_page))):
        return "a student's own report, thesis or certificate"
    return None


# ---------------------------------------------------------------- superseded editions

# Re-issued every academic year; the newest edition replaces the older ones.
SERIES_WORDS = ("handbook", "prospectus", "academic calendar", "calendar", "fees structure", "fee structure",
                "brochure", "information bulletin", "policy")
_YEARS = re.compile(r"(19|20)\d{2}(\s*[-–_]\s*(19|20)?\d{2})?|\b\d{2}\s*[-–_]\s*\d{2}\b")
_FILLER = re.compile(r"\b(final|new|updated|revised|latest|compressed|copy|only|printed|pages|sxca|sxc|xaviers?|"
                     r"st|college|of|the|and|for|ay|a\s?y|pdf|v\d|\d+)\b", re.I)


def _file_name(url: str) -> str:
    return unquote(urlparse(url).path.rsplit("/", 1)[-1])


def series_key(url: str) -> str | None:
    """'handbook' for .../SXC-Handbook-2025-26.pdf and .../SXCA-Handbook-2026-27.pdf; None if not a series."""
    name = re.sub(r"\.(pdf|docx?)$", "", _file_name(url).lower())
    name = _YEARS.sub(" ", name)
    name = re.sub(r"[^a-z]+", " ", name)
    name = " ".join(_FILLER.sub(" ", name).split())
    if not any(w in name for w in SERIES_WORDS):
        return None
    # "Scholarships" in one year's file name, "Scholarship" in the next: same series.
    return " ".join(w[:-1] if len(w) > 4 and w.endswith("s") and not w.endswith("us") else w for w in name.split())


def edition_year(url: str, academic_year: str | None) -> int | None:
    """Starting year of the edition: from the file name ("2025-26", "25-26", "2025"), else its academic year."""
    name = _file_name(url)
    m = re.search(r"(20\d{2})\s*[-–_]\s*(20)?\d{2}(?!\d)", name)
    if m:
        return int(m.group(1))
    m = re.search(r"(?<!\d)(\d{2})\s*[-–_]\s*(\d{2})(?!\d)", name)
    if m and int(m.group(2)) == int(m.group(1)) + 1:  # "21-22", not "1-2" or a date
        return 2000 + int(m.group(1))
    m = re.search(r"(?<!\d)(20\d{2})(?!\d)", name)
    if m:
        return int(m.group(1))
    if academic_year and academic_year[:4].isdigit():
        return int(academic_year[:4])
    return None


def superseded_ids(docs: list[tuple[int, str, str | None]]) -> dict[int, str]:
    """{source id: reason} for older editions. `docs` = [(id, url, academic_year)] of documents (not pages)."""
    series: dict[str, list[tuple[int, int, str]]] = defaultdict(list)
    for sid, url, ay in docs:
        key, year = series_key(url), edition_year(url, ay)
        if key and year:
            # File names have typos ("Differerently"): near-identical names are the same series.
            key = next((k for k in series if SequenceMatcher(None, k, key).ratio() >= 0.93), key)
            series[key].append((year, sid, url))
    out: dict[int, str] = {}
    for key, editions in series.items():
        newest = max(y for y, _, _ in editions)
        for year, sid, _url in editions:
            if year < newest:
                out[sid] = f"older {key} ({year}); a {newest} edition exists"
    return out
