import time
from datetime import datetime, timezone

import pytest

from app.rag import graph as graph_mod
from app.rag.query import is_time_sensitive
from app.rag.retriever import content_age_days, drop_outdated, recency
from app.rag.retriever import Retrieval
from app.rag.vectorstore import Hit

DAY = 86400


def meta(days_old=None, ay="", url="https://sxca.edu.in/x.pdf"):
    m = {"url": url, "title": "Notice", "academic_year": ay}
    if days_old is not None:
        ts = time.time() - days_old * DAY
        m["date_ts"] = int(ts)
        m["date"] = datetime.fromtimestamp(ts, tz=timezone.utc).date().isoformat()
    return m


def hit(cid, days_old=None, ay="", grade=0.7, cov=0.8):
    m = meta(days_old, ay)
    return Hit(chunk_id=cid, text="Fees must be paid by the due date.", score=grade, metadata=m,
               signals={"grade": grade, "keyword_coverage": cov,
                        "age_days": None if content_age_days(m) is None else int(content_age_days(m))})


@pytest.mark.parametrize("q", ["When are the fees due?", "latest news", "exam timetable", "When does admission start?",
                               "Is there a holiday on Monday?", "upcoming events this month"])
def test_time_sensitive_questions_detected(q):
    assert is_time_sensitive(q)


@pytest.mark.parametrize("q", ["Who is the principal?", "Does the college have NCC?", "What is the email of Nisarg Vyas?"])
def test_ordinary_questions_not_time_sensitive(q):
    assert not is_time_sensitive(q)


def test_academic_year_makes_an_old_report_old_even_if_uploaded_recently():
    assert content_age_days(meta(days_old=30, ay="2022-23")) > 3 * 365
    assert content_age_days(meta(days_old=30)) < 31
    assert content_age_days(meta()) is None


def test_recency_strongly_favours_new_for_time_sensitive_only():
    new, old = meta(days_old=60), meta(days_old=450)
    assert recency(new, True) == 1.0 and recency(old, True) < 0
    assert recency(meta(), True) == 0.0  # undated pages (faculty, rules) are neutral
    assert 0 < recency(old, False) < recency(new, False)  # ordinary questions: only a gentle tie-breaker


def test_old_fee_notice_dropped_when_a_newer_one_exists():
    kept = drop_outdated([hit("a", 600), hit("b", 40), hit("c")])
    assert [h.chunk_id for h in kept] == ["b", "c"]


def test_old_notice_kept_when_newer_pages_are_about_something_else():
    # The 2025 fee notice (grade .75) is not replaced by a 2026 hostel page (grade .62).
    kept = drop_outdated([hit("fee2025", 470 + 130, grade=0.75), hit("hostel2026", 40, grade=0.62, cov=0.2)])
    assert [h.chunk_id for h in kept] == ["fee2025", "hostel2026"]


def test_last_years_fee_notice_replaced_by_this_years_calendar():
    # Real case: the June 2025 fee notice scores higher (it is only about fees) than the 2026-27 academic
    # calendar, but the calendar also gives the fee-payment date and matches as many of the question's words.
    notice = hit("notice-jun-2025", 473, grade=0.82, cov=0.8)
    calendar = hit("calendar-2026-27", 120, ay="2026-27", grade=0.68, cov=0.8)
    syllabus = hit("syllabus-2026", 15, grade=0.61, cov=0.2)
    kept = drop_outdated([notice, calendar, syllabus])
    assert [h.chunk_id for h in kept] == ["calendar-2026-27", "syllabus-2026"]


def test_old_source_kept_when_newer_ones_cover_less_of_the_question():
    old_exam_page = hit("exams-2024-25", 660, ay="2024-25", grade=0.76, cov=1.0)
    newer = hit("welcome-2026-27", 12, ay="2026-27", grade=0.68, cov=0.5)
    assert [h.chunk_id for h in drop_outdated([old_exam_page, newer])] == ["exams-2024-25", "welcome-2026-27"]


def test_newer_source_must_itself_be_relevant():
    kept = drop_outdated([hit("notice-2025", 473, grade=0.82), hit("weak-2026", 30, grade=0.46)], threshold=0.5)
    assert [h.chunk_id for h in kept] == ["notice-2025", "weak-2026"]


def test_home_page_date_is_ignored():
    assert content_age_days(meta(days_old=1, url="https://sxca.edu.in/")) is None


def test_old_notice_kept_when_it_is_all_there_is():
    kept = drop_outdated([hit("a", 600), hit("c")])
    assert [h.chunk_id for h in kept] == ["a", "c"]


async def _answer(monkeypatch, hits, question):
    from tests.test_rag_graph import FakeLLM

    monkeypatch.setattr(graph_mod, "get_llm", lambda: FakeLLM("Fees must be paid by the due date."))
    monkeypatch.setattr(graph_mod, "embed_query", lambda q: [0.0])
    monkeypatch.setattr(graph_mod.verified_index, "best", lambda e: None)
    monkeypatch.setattr(graph_mod, "hybrid_search", lambda *a, **k: Retrieval(hits=hits, best_score=0.7, relevant=True))
    text = ""
    async for mode, chunk in graph_mod.build_graph().astream({"question": question, "history": []},
                                                              stream_mode=["custom", "values"]):
        if mode == "custom" and chunk["type"] == "token":
            text += chunk["text"]
    return text


async def test_answer_from_old_notice_says_how_old(monkeypatch):
    text = await _answer(monkeypatch, [hit("a", 470, ay="2025-26")], "When are the fees due?")
    assert "Note: the most recent information" in text and "2025" in text and "may be out of date" in text


async def test_answer_from_current_notice_has_no_warning(monkeypatch):
    text = await _answer(monkeypatch, [hit("a", 20)], "When are the fees due?")
    assert "Note:" not in text


async def test_ordinary_question_gets_no_warning(monkeypatch):
    text = await _answer(monkeypatch, [hit("a", 900)], "Who teaches statistics?")
    assert "Note:" not in text
