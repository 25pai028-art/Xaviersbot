"""Query understanding: follow-up handling, abbreviation expansion and (fallback) LLM rewriting."""
from __future__ import annotations

import re

from app.llm.base import ChatMessage, LLMProvider

# Terms students use vs. words the website uses. Expansion helps both keyword and vector search.
ABBREVIATIONS = {
    "bca": "BCA Bachelor of Computer Applications B.S. (BCA)",
    "mca": "MCA Master of Computer Applications",
    "bcom": "B.Com Bachelor of Commerce",
    "mcom": "M.Com Master of Commerce",
    "bba": "BBA Bachelor of Business Administration B.S. (Business Administration)",
    "bsc": "B.Sc Bachelor of Science",
    "msc": "M.Sc Master of Science",
    "ba": "B.A. Bachelor of Arts",
    "ma": "M.A. Master of Arts",
    "bps": "Business Process Services",
    # The fee tables and course pages spell these out ("MSc (ARTIFICIAL INTELLIGENCE)")
    "ai": "Artificial Intelligence",
    "aiml": "Artificial Intelligence and Machine Learning",
    "ml": "Machine Learning",
    "ds": "Data Science",
    "cs": "Computer Science",
    "bda": "Big Data Analytics",
    "bt": "Biotechnology",
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

# Questions whose answer changes over time: the newest source must win, old ones are dropped.
TIME_SENSITIVE = re.compile(
    r"\b(fees?|due|dues|deadline|last date|pay(ment)?|notices?|circulars?|news|latest|recent|upcoming|current|"
    r"this (year|semester|sem|month|week|term)|today|tomorrow|events?|fest|festival|exams?|examinations?|"
    r"time ?tables?|schedule|results?|admissions?|admission form|merit list|calendar|holidays?|vacation|"
    r"registration|last day|starts?|begin|reopen|placement drive|interview|"
    # Who holds a senior post also changes: the 2026 Handbook names today's principal, old yearly reports
    # (AQAR, audits) earlier ones and match "principal of St. Xavier's College" better. Kept to these posts:
    # for cells and committees the contact page beats a newer handbook that doesn't name anyone.
    r"principal|vice[- ]?principal|director|rector|registrar|dean|hods?|head of (the )?(\w+ )?department)\b",
    re.I)


def is_time_sensitive(question: str) -> bool:
    return bool(TIME_SENSITIVE.search(question))


def contextualize(question: str, history: list[ChatMessage]) -> str:
    """Make a follow-up self-contained for search, using the student's previous question.

    A question has a *topic* (fee, eligibility, syllabus…) and a *subject* (BCA, MSc AI, Chemistry…):
    - "how do I apply for it?" / "and the fees?" (no subject of its own) -> previous question + this one
    - "and what about MSc AI?" (new subject, no topic) -> this one + the previous topic: "… MSc AI? fee 2026-27"
    - "Who is the principal?", "what about the hostel?" (complete, or about a different area) -> as asked
    Gluing the whole previous question on ("What is the BCA fee? and what about MSc AI?") made search
    look for both programmes and find neither."""
    prev = next((m.content for m in reversed(history) if m.role == "user"), "").strip()
    q = question.strip()
    if not prev:
        return q
    if _PRONOUN.search(q):  # "how do I apply for it?"
        return f"{prev} {q}"
    words = re.findall(r"[a-z0-9]+", q.lower())
    subject = [w for w in words if w not in _FILLER_WORDS and w not in TOPIC_WORDS]
    if not subject:  # "and the fees?", "eligibility?": needs the earlier subject
        return f"{prev} {q}"
    if any(w in _STANDALONE_SUBJECTS for w in subject):  # "what about the hostel?" is its own question
        return q
    if _FOLLOW_UP_OPENER.match(q) or len(words) <= 3:  # "and what about MSc AI?", "MSc AI?"
        prev_lower = prev.lower()
        borrowed = [t for t in TOPIC_WORDS if re.search(rf"\b{t}\b", prev_lower) and t not in words]
        borrowed += [y for y in re.findall(r"\b20\d{2}\s*[-–]\s*\d{2,4}\b", prev) if y not in q]
        return f"{q} {' '.join(borrowed)}" if borrowed else q
    return q  # a complete new question


# What is being asked about something; carried over to "and what about <another programme>?"
TOPIC_WORDS = ("fee", "fees", "structure", "eligibility", "eligible", "criteria", "admission", "admissions", "apply",
               "application", "syllabus", "duration", "seats", "intake", "timings", "deadline", "last date",
               "documents", "refund", "scholarship", "scholarships", "contact", "email", "phone", "head", "hod",
               "faculty", "subjects", "exam", "exams", "result", "results", "timetable", "cutoff", "merit")
_FILLER_WORDS = set("""and also what about how is are was were the a an for of in on at to then so ok okay tell me
please same do does did i can you could would will there any with its it this that those these which who when where
why much many give show list know need want""".split())
# Areas of their own: "what about the hostel?" after a fee question asks about the hostel, not its fee.
_STANDALONE_SUBJECTS = set("""hostel hostels library canteen cafeteria transport bus parking sports gym ground clubs
club nss ncc events fest placement placements principal director rector campus wifi medical clinic uniform""".split())
_FOLLOW_UP_OPENER = re.compile(r"^\s*(and|also|what about|how about|and what about|what of|same for|then|for)\b", re.I)
_PRONOUN = re.compile(r"\b(it|its|it's|they|them|their|that one|this one|those|these|he|she|him|her|his)\b", re.I)


def glossary(question: str) -> list[str]:
    """Short forms in the question with their meaning, for the model: "BBA = Bachelor of Business
    Administration B.S. (Business Administration)". The college writes the long names, so without this the
    model doesn't connect "BBA" with "B.S. (BUSINESS ADMINISTRATION)" in the fee table."""
    q = re.sub(r"\b([A-Za-z])\.(?=[A-Za-z])", r"\1", question)  # B.Com -> BCom
    out = []
    for short, long in ABBREVIATIONS.items():
        if len(short) <= 5 and short not in ("fee", "fees") and re.search(rf"\b{short}\b", q, re.I):
            meaning = re.sub(rf"^{short}\s+", "", long, flags=re.I)
            out.append(f"{short.upper()} = {meaning}")
    return out


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
