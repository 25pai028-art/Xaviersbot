"""Post-generation fact check.

Every fee/amount, long number, date, phone number, email and URL in the answer must
appear in the retrieved context (or in the user's own question). If one does
not, the answer is treated as a possible hallucination and replaced.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import date

EMAIL = re.compile(r"[\w.+-]+@[\w-]+(?:\.[\w-]+)+")
URL = re.compile(r"https?://[^\s<>\"')\]]+", re.I)
PHONE = re.compile(r"(?<!\w)\+?\d[\d \-–]{7,}\d(?!\w)")
AMOUNT = re.compile(r"(?:₹|rs\.?|inr)\s*([\d,]+(?:\.\d+)?)", re.I)
NUMBER = re.compile(r"(?<![\w.])\d{1,3}(?:,\d{2,3})+(?:\.\d+)?|(?<![\w.,])\d{3,}(?:\.\d+)?(?![\w])")

FEE_PERIODS = {
    "year": re.compile(r"\b(per (year|annum)|p\.\s?a\.|annual(ly)?|yearly|a year|/\s?year|each year|every year)\b", re.I),
    "month": re.compile(r"\b(per month|monthly|a month|/\s?month|each month)\b", re.I),
}
AMOUNT_OR_NUMBER = re.compile(r"(?:₹|rs\.?|inr)\s*\d|\d{1,3},\d{3}", re.I)
ACADEMIC_YEAR = re.compile(r"\b(20\d{2})\s*[-–/]\s*(\d{2})\b")

# Dates are checked as whole dates (day + month + year), not as loose numbers: otherwise "26–29 June 2026"
# passes whenever "26", "29" and "2026" each appear somewhere in the sources.
_MONTHS = ["jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec"]
_MONTH = r"(jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|june?|july?|aug(?:ust)?|sep(?:t(?:ember)?)?|oct(?:ober)?|nov(?:ember)?|dec(?:ember)?)"
_DAY = r"(?<!\d)(\d{1,2})(?:st|nd|rd|th)?(?!\d)"
_RANGE = r"(?:\s*(?:-|–|—|to|and|&)\s*" + _DAY + r")?"
DATE_DAY_FIRST = re.compile(_DAY + _RANGE + r"\s*(?:of\s+)?" + _MONTH + r"\b\.?,?(?:\s*'?(\d{4}))?", re.I)
DATE_MONTH_FIRST = re.compile(r"\b" + _MONTH + r"\b\.?\s+" + _DAY + _RANGE + r"(?:,?\s*(\d{4}))?", re.I)
DATE_NUMERIC = re.compile(r"(?<![\d/.-])(\d{1,2})[/.-](\d{1,2})[/.-](\d{4}|\d{2})(?![\d/-])(?!\.\d)")
DATE_ISO = re.compile(r"(?<!\d)(20\d{2})-(\d{2})-(\d{2})(?!\d)")

DateKey = tuple[int, int, int | None]  # (day, month, year or None)

NO_INFO = re.compile(
    r"(don'?t|do not|doesn'?t|does not) (have|contain|include|mention|list|provide)[^.]{0,60}"
    r"(information|details|mention|specific)|not (available|mentioned|provided|listed) in the (context|information)",
    re.I)


@dataclass
class FactCheck:
    ok: bool
    unsupported: list[str] = field(default_factory=list)


def _digits(s: str) -> str:
    return re.sub(r"\D", "", s)


def _clean_url(u: str) -> str:
    return u.rstrip(".,;:!?*").rstrip("/").lower()


def check_answer(answer: str, context: str, question: str = "") -> FactCheck:
    evidence = f"{context}\n{question}"
    # Academic years are written "2024-25" but answers often say "2024-2025": add the full end year.
    evidence += " " + " ".join(f"{m.group(1)} {m.group(1)[:2]}{m.group(2)}"
                               for m in ACADEMIC_YEAR.finditer(evidence))
    ev_lower = evidence.lower()
    ev_digits = {_digits(m) for m in PHONE.findall(evidence)} | {_digits(m) for m in NUMBER.findall(evidence)}
    ev_digits |= {_digits(m) for m in AMOUNT.findall(evidence)}
    ev_urls = {_clean_url(u) for u in URL.findall(evidence)}
    this_year = date.today().year
    unsupported: list[str] = []

    for email in EMAIL.findall(answer):
        if email.lower() not in ev_lower:
            unsupported.append(email)
    for url in URL.findall(answer):
        cu = _clean_url(url)
        if not any(cu == e or e.startswith(cu) or cu.startswith(e) for e in ev_urls):
            unsupported.append(url)
    answer_wo_links = URL.sub(" ", EMAIL.sub(" ", answer))
    for phone in PHONE.findall(answer_wo_links):
        d = _digits(phone)
        if len(d) >= 8 and not any(d[-10:] in e for e in ev_digits if len(e) >= 8):
            unsupported.append(phone.strip())
    answer_wo_phones = PHONE.sub(" ", answer_wo_links)  # phones were checked as a whole above
    for token in AMOUNT.findall(answer_wo_phones) + NUMBER.findall(answer_wo_phones):
        d = _digits(token)
        if not d or len(d) >= 8:  # long digit runs were handled as phone numbers
            continue
        if len(d) == 4 and abs(int(d) - this_year) <= 30:  # years: allow if in evidence or near today
            if d in ev_digits or re.search(rf"\b{d}\b", evidence) or int(d) in (this_year, this_year + 1):
                continue
        if d not in ev_digits and not re.search(rf"(?<!\d){re.escape(d)}(?!\d)", _digits_spaced(evidence)):
            unsupported.append(token)
    ev_dates = {k for _, keys in find_dates(evidence) for k in keys}
    for text, keys in find_dates(answer_wo_links):
        if not all(_date_supported(k, ev_dates) for k in keys):
            unsupported.append(text)
    # A fee period the sources never use: small models turn "Sem-1: 31,250" into "31,250 per year".
    for pattern in FEE_PERIODS.values():
        m = pattern.search(answer)
        if m and AMOUNT_OR_NUMBER.search(answer) and not pattern.search(evidence):
            unsupported.append(m.group(0))
    return FactCheck(ok=not unsupported, unsupported=list(dict.fromkeys(unsupported)))


_BULLET = re.compile(r"^\s*([-*•]|\d+[.)])\s")
# A sentence ends after a lower-case word or a figure, not after "B.S." / "M.Sc." / "Dr."
_SENTENCE_END = re.compile(r"(?<=[a-z0-9)\]*][.!?])(?<!\b[A-Z][a-z]\.)\s+")
_INTRO_ONLY = re.compile(r"^\W*(this (includes|is made up of|consists of)|the breakdown|breakdown|details|including)\b.{0,40}:\s*$",
                         re.I)


def drop_unsupported_lines(answer: str, unsupported: list[str]) -> str:
    """The answer without the bullets / sentences that contain an unsupported fact, or "" when what is left is
    not a real answer. Used when a retry still got a figure wrong: the checked parts are kept."""
    # A wrong fee period ("per year") is just words: remove them and keep the amount, which was checked.
    periods = [u for u in unsupported if any(p.fullmatch(u) for p in FEE_PERIODS.values())]
    for u in periods:
        answer = re.sub(rf"[ \t]*\(?\b{re.escape(u)}\b\)?", "", answer)
    unsupported = [u for u in unsupported if u not in periods]

    def bad(text: str) -> bool:
        return any(u in text for u in unsupported)

    kept = []
    for line in answer.splitlines():
        if not bad(line):
            kept.append(line)
        elif not _BULLET.match(line):  # running text: drop only the wrong sentences
            rest = " ".join(x for x in _SENTENCE_END.split(line) if not bad(x)).strip()
            if rest:
                kept.append(rest)
    text = "\n".join(kept).strip()
    if not re.search(r"\d", text) and len(text) < 80:
        return ""  # nothing substantial survived (the wrong figure *was* the answer)
    return "" if _INTRO_ONLY.match(text) else text


def _date(day: str | None, month: int, year: str | None) -> DateKey | None:
    if not day or not 1 <= int(day) <= 31 or not 1 <= month <= 12:
        return None
    y = int(year) if year else None
    if y is not None and y < 100:
        y += 2000
    return int(day), month, y


def find_dates(text: str) -> list[tuple[str, list[DateKey]]]:
    """Every date in the text as (matched text, [(day, month, year|None), …]); a range gives both ends."""
    found: list[tuple[str, list[DateKey]]] = []
    for m in DATE_DAY_FIRST.finditer(text):
        month = _MONTHS.index(m.group(3)[:3].lower()) + 1
        keys = [_date(d, month, m.group(4)) for d in (m.group(1), m.group(2)) if d]
        found.append((m.group(0).strip(), [k for k in keys if k]))
    for m in DATE_MONTH_FIRST.finditer(text):
        month = _MONTHS.index(m.group(1)[:3].lower()) + 1
        keys = [_date(d, month, m.group(4)) for d in (m.group(2), m.group(3)) if d]
        found.append((m.group(0).strip(), [k for k in keys if k]))
    for m in DATE_NUMERIC.finditer(text):  # Indian order: dd/mm/yyyy
        k = _date(m.group(1), int(m.group(2)), m.group(3))
        found.append((m.group(0), [k] if k else []))
    for m in DATE_ISO.finditer(text):
        k = _date(m.group(3), int(m.group(2)), m.group(1))
        found.append((m.group(0), [k] if k else []))
    return [(t, keys) for t, keys in found if keys]


def _date_supported(key: DateKey, evidence: set[DateKey]) -> bool:
    day, month, year = key
    return any(d == day and m == month and (year is None or y is None or y == year) for d, m, y in evidence)


def _digits_spaced(text: str) -> str:
    """Text with thousands separators removed (20,000 → 20000) for number matching."""
    return re.sub(r"(?<=\d),(?=\d)", "", text)


def is_no_info_answer(answer: str) -> bool:
    return bool(NO_INFO.search(answer)) or "don't have that information" in answer.lower()


# ---------------------------------------------------------------- sentences (for the grounding check)
_STOP = set("""about above after again also although among another because before being below between both
could does doing during each either every from further have having here into itself just more most must
neither other otherwise ought same shall should since some such than that their theirs them then there these
they this those though through under until upon very were what when where whether which while whom whose
will with within without would your yours you are was the and for but not can may has had its our out""".split())


def content_words(text: str) -> list[str]:
    return [w for w in re.findall(r"[a-z]+", text.lower()) if len(w) >= 4 and w not in _STOP]


def answer_units(answer: str) -> list[str]:
    """Bullets and sentences of an answer, exactly as they appear in it."""
    units = []
    for line in answer.splitlines():
        if not line.strip():
            continue
        units += [line] if _BULLET.match(line) else [s for s in _SENTENCE_END.split(line) if s.strip()]
    return units
