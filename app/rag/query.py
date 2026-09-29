"""Query understanding: follow-up handling, abbreviation expansion and (fallback) LLM rewriting."""
from __future__ import annotations

import re

from app.llm.base import ChatMessage, LLMProvider

# Terms students use vs. words the website uses. Expansion helps both keyword and vector search.
ABBREVIATIONS = {
    "bca": "BCA Bachelor of Computer Applications",
    "mca": "MCA Master of Computer Applications",
    "bcom": "B.Com Bachelor of Commerce",
    "mcom": "M.Com Master of Commerce",
    "bba": "BBA Bachelor of Business Administration",
    "bsc": "B.Sc Bachelor of Science",
    "msc": "M.Sc Master of Science",
    "ba": "B.A. Bachelor of Arts",
    "ma": "M.A. Master of Arts",
    "bps": "Business Process Services",
    "phd": "Ph.D doctoral research",
    "ug": "UG undergraduate",
    "pg": "PG postgraduate",
    "hod": "Head of Department",
    "coe": "Controller of Examinations examination office",
    "nss": "NSS National Service Scheme",
    "ncc": "NCC National Cadet Corps",
    "iqac": "IQAC Internal Quality Assurance Cell",
    "naac": "NAAC accreditation",
    "nirf": "NIRF ranking",
    "nep": "NEP National Education Policy",
    "fees": "fees fee structure",
    "fee": "fee structure tuition fees",
    "timetable": "timetable schedule",
    "hostel": "hostel accommodation residence",
    "principal": "Principal head of the college",
    "placement": "placement career cell campus recruitment",
    "scholarship": "scholarship freeship financial assistance",
    "contact": "contact phone email office",
}
_FOLLOW_UP = re.compile(
    r"^(and|also|what about|how about|and what|then|so|ok|okay|its|it's|their|his|her|same|that|this|those|these)\b"
    r"|\b(it|its|they|them|their|that|this|those|these|he|she|him|her)\b", re.I)


def contextualize(question: str, history: list[ChatMessage]) -> str:
    """Make follow-ups ('and the fees?', 'how do I apply for it?') self-contained using the last user turn."""
    prev = next((m.content for m in reversed(history) if m.role == "user"), "")
    if not prev:
        return question
    if len(question.split()) <= 6 or _FOLLOW_UP.search(question):
        return f"{prev.strip()} {question.strip()}"
    return question


def expand_abbreviations(query: str) -> str:
    q = re.sub(r"\b([A-Za-z])\.(?=[A-Za-z])", r"\1", query)  # B.Com → BCom
    extra = [exp for key, exp in ABBREVIATIONS.items() if re.search(rf"\b{key}\b", q, re.I)]
    return f"{query} ({'; '.join(extra)})" if extra else query


REWRITE_PROMPT = (
    "Rewrite the user's question as a short search query for St. Xavier's College Ahmedabad website pages. "
    "Expand abbreviations, add likely page words (e.g. 'fee structure', 'eligibility', 'contact'). "
    "Output ONLY the query on one line, no explanation."
)


async def llm_rewrite(llm: LLMProvider, question: str) -> str:
    result = await llm.generate(REWRITE_PROMPT, [ChatMessage(role="user", content=question)])
    line = (result.text or "").strip().splitlines()[0] if result.text.strip() else ""
    line = line.strip().strip('"').strip()
    return line if 3 <= len(line) <= 300 else question
