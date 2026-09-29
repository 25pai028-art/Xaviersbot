"""Post-generation fact check.

Every fee/amount, long number, phone number, email and URL in the answer must
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

ACADEMIC_YEAR = re.compile(r"\b(20\d{2})\s*[-–/]\s*(\d{2})\b")

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
    return FactCheck(ok=not unsupported, unsupported=list(dict.fromkeys(unsupported)))


def _digits_spaced(text: str) -> str:
    """Text with thousands separators removed (20,000 → 20000) for number matching."""
    return re.sub(r"(?<=\d),(?=\d)", "", text)


def is_no_info_answer(answer: str) -> bool:
    return bool(NO_INFO.search(answer)) or "don't have that information" in answer.lower()
