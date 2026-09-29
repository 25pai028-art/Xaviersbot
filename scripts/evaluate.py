"""Evaluate retrieval (fast, no LLM) and optionally full answers.

    python -m scripts.evaluate                    # retrieval only: is the right page in the top results?
    python -m scripts.evaluate --answers          # also generate answers through the full LangGraph flow
    python -m scripts.evaluate --file my.jsonl    # your own questions

    python -m scripts.evaluate --file tests/eval/student_questions.jsonl --answers

Question file (JSON lines): {"q": "...", "expect": ["url-substring", ...], "note": "optional"}
- `expect: [..]`  PASS if a retrieved source URL contains one of the substrings
- `expect: []`    the website cannot answer it: PASS only if the bot refuses
- `expect: null`  correct source not known yet: shown as REVIEW, not scored
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

from rich.console import Console  # noqa: E402
from rich.table import Table  # noqa: E402

from app.rag.embeddings import embed_query  # noqa: E402
from app.rag.query import expand_abbreviations  # noqa: E402
from app.rag.retriever import hybrid_search  # noqa: E402

console = Console()

DEFAULT_QUESTIONS = [
    {"q": "What are the eligibility criteria for B.Sc Physics?", "expect": ["ug-science"]},
    {"q": "Which B.Com programmes are offered?", "expect": ["ug-commerce", "ug-admissions"]},
    {"q": "What documents are required for admission?", "expect": ["faqs", "general-instructions"]},
    {"q": "Is there a hostel for students?", "expect": ["faqs", "hostel"]},
    {"q": "What is the email of the examination office?", "expect": ["contact-us", "site://"]},
    {"q": "Who is the in-charge of the Career Cell?", "expect": ["contact-us", "career-cell", "site://"]},
    {"q": "Does the college have NCC?", "expect": ["ncc"]},
    {"q": "What clubs can students join?", "expect": ["clubs"]},
    {"q": "Tell me about NSS activities", "expect": ["nss"]},
    {"q": "What sports facilities are there?", "expect": ["sports"]},
    {"q": "How can I get my transcript or certificate verified?", "expect": ["academic-affairs", "administrative-assistance"]},
    {"q": "What are the refund rules for fees?", "expect": ["tuition-fees"]},
    {"q": "Which PG courses are available in arts?", "expect": ["pg-arts", "pg-admissions"]},
    {"q": "Where is the syllabus?", "expect": ["syllabus"]},
    {"q": "What is the research policy?", "expect": ["research-policy"]},
    {"q": "Who teaches Botany?", "expect": ["faculty", "author"]},
    # Not answerable from the website: the bot must refuse.
    {"q": "What is the cafeteria menu for Monday?", "expect": []},
    {"q": "Who won the IPL in 2025?", "expect": []},
]


def load(path: str | None) -> list[dict]:
    if not path:
        return DEFAULT_QUESTIONS
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]


def eval_retrieval(questions: list[dict]) -> None:
    t = Table(title="Retrieval evaluation (hybrid search, top results)", show_lines=True)
    for col in ("result", "grade", "question", "top sources", "note"):
        t.add_column(col)
    ok = scored = 0
    for item in questions:
        q, expect = item["q"], item.get("expect")
        sq = expand_abbreviations(q)
        r = hybrid_search(sq, embed_query(sq), grade_query=q)
        urls = [h.metadata.get("url", "") for h in r.hits] if r.relevant else []
        if expect is None:
            label = "[yellow]REVIEW[/yellow]"
        else:
            passed = (r.relevant and any(e in u for u in urls for e in expect)) if expect else not r.relevant
            ok += passed
            scored += 1
            label = "[green]PASS[/green]" if passed else "[red]FAIL[/red]"
        t.add_row(label, f"{r.best_score:.2f}", q,
                  "\n".join(u.replace("https://sxca.edu.in", "")[:60] for u in urls[:3]) or "(refuses)",
                  item.get("note", ""))
    console.print(t)
    console.print(f"[bold]{ok}/{scored} scored questions passed[/bold] ({100 * ok // max(1, scored)}%), "
                  f"{len(questions) - scored} to review")


async def eval_answers(questions: list[dict]) -> None:
    from app.rag.service import answer_stream

    for item in questions:
        t0 = time.time()
        text, sources, answered = "", [], True
        async for ev in answer_stream(item["q"]):
            if ev.type == "token":
                text += ev.text
            elif ev.type == "replace":
                text = ev.text
            elif ev.type == "sources":
                sources = [s["url"] for s in ev.sources]
            elif ev.type == "done":
                answered = ev.answered
        expect = item.get("expect")
        if expect is None:
            verdict = "review"
        else:
            verdict = "ok" if answered == bool(expect) else "CHECK"
        console.print(f"\n[bold]{item['q']}[/bold]  ({time.time() - t0:.0f}s, answered={answered}, {verdict})")
        console.print(text.strip())
        if sources and answered:
            console.print(f"[dim]sources: {', '.join(sources)}[/dim]")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--file", help="JSON-lines question file")
    ap.add_argument("--answers", action="store_true", help="also generate full answers (slow on CPU)")
    args = ap.parse_args()
    questions = load(args.file)
    eval_retrieval(questions)
    if args.answers:
        asyncio.run(eval_answers(questions))


if __name__ == "__main__":
    main()
