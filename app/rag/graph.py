"""The answering workflow as a LangGraph state machine.

    guard ─┬─ greeting / thanks / okay / bye ─► small_talk ─► END
           ├─ cheating / hacking ─► refuse ─► END
           ├─ out of scope ───────► out_of_scope ─► END
           ├─ abuse only ─────────► calm ─► END
           └─ ok (abuse stripped) ─► prepare ─► verified ─┬─ official answer matches ─► END
                                                           └─ retrieve ─┬─ relevant ──► generate ─► verify ─► END
                                                           ├─ weak, first try ─► rewrite ─► retrieve
                                                           └─ weak after retry ─► no_answer ─► END

- guard:     rule-based input guardrails (no LLM): answer small talk instantly, refuse misconduct with the official
             channel, redirect off-topic questions, de-escalate pure abuse, strip profanity
- prepare:   make follow-ups self-contained, expand abbreviations (BCA, HOD, CoE…)
- verified:  the college's official Q&A first: a close match is answered word for word (no LLM),
             a partial match is handed to the LLM as the most important source
- retrieve:  hybrid vector + keyword search, relevance grading, freshness, neighbour chunk
- rewrite:   the LLM turns the question into a better search query (only when retrieval is weak)
- no_answer: fixed "I don't have that information" reply (or the out-of-scope reply if the
             question isn't about the college), no LLM call (cannot hallucinate)
- generate:  answer strictly from the retrieved context, streamed token by token
- verify:    every number / fee / phone / email / URL must appear in the context,
             otherwise the streamed answer is replaced by the safe fallback

Nodes emit UI events (sources, tokens, replace) through LangGraph's stream writer,
so any channel (web widget, WhatsApp later) can consume the same stream.
"""
from __future__ import annotations

import asyncio
import logging
import re
from functools import lru_cache
from typing import TypedDict

from langgraph.config import get_stream_writer
from langgraph.graph import END, START, StateGraph

from app.config import get_settings
from app.guardrails.input import check_input, is_about_college
from app.llm.base import ChatMessage, LLMError
from app.llm.factory import get_llm
from app.rag.embeddings import embed_query
from app.rag.factcheck import answer_units, check_answer, content_words, drop_unsupported_lines, is_no_info_answer
from app.rag.prompts import (CALM_MESSAGE, GROUNDING_PROMPT, MISCONDUCT_MESSAGE, OUT_OF_SCOPE_MESSAGE, RETRY_PROMPT,
                             SMALL_TALK_MESSAGES,
                             STALE_NOTE,
                             build_system_prompt, build_user_turn, format_context, no_info_message)
from app.rag.query import contextualize, expand_abbreviations, is_time_sensitive, llm_rewrite
from app.rag.retriever import FRESH_DAYS, Retrieval, content_age_days, hybrid_search
from app.rag.vectorstore import Hit
from app.rag.verified import record_use, verified_index

log = logging.getLogger(__name__)

# One generation at a time on a CPU-only machine; others wait in line.
llm_slot = asyncio.Semaphore(1)

BUSY_MESSAGE = "The assistant is busy or temporarily unavailable. Please try again in a moment."


class RAGState(TypedDict, total=False):
    question: str  # after the guard: profanity removed
    guard: str  # ok | misconduct | off_topic | abuse_only | small_talk
    small_talk: str  # greeting | thanks | bye | how_are_you | identity | ack
    history: list[ChatMessage]
    grade_query: str  # self-contained question in the user's words
    search_query: str  # grade_query + expansions / LLM rewrite
    attempts: int
    time_sensitive: bool  # fees, deadlines, notices, events…: newest source wins
    retrieval: Retrieval
    answer: str
    answered: bool
    reason: str  # why a question was not answered (for the admin's unanswered list)
    verified_hit: Hit  # official answer to give the LLM as its first source
    context_hits: list  # exactly the hits shown to the LLM (for the fact check)
    verified_direct: bool  # answered word for word from an official answer
    usage: dict  # {"calls", "input_tokens", "output_tokens"} for the cost dashboard


def _sources(hits: list[Hit]) -> list[dict]:
    seen, out = set(), []
    for h in hits:
        url = h.metadata.get("url", "")
        if url and not url.startswith("site://") and url not in seen:
            seen.add(url)
            out.append({"title": h.metadata.get("title") or url, "url": url, "date": h.metadata.get("date", "")})
    return out


def _fixed_reply(text: str, reason: str) -> RAGState:
    get_stream_writer()({"type": "token", "text": text})
    return {"answer": text, "answered": False, "reason": reason}


# ---------------------------------------------------------------- nodes
async def guard(state: RAGState) -> RAGState:
    g = check_input(state["question"])
    if g.abusive:
        log.info("guard: abusive language removed")
    return {"question": g.question, "guard": g.kind, "small_talk": g.small_talk}


def after_guard(state: RAGState) -> str:
    return {"misconduct": "refuse", "off_topic": "out_of_scope", "abuse_only": "calm",
            "small_talk": "small_talk"}.get(state["guard"], "prepare")


async def small_talk(state: RAGState) -> RAGState:
    text = SMALL_TALK_MESSAGES[state["small_talk"]].format(bot_name=get_settings().app_name)
    get_stream_writer()({"type": "token", "text": text})
    return {"answer": text, "answered": True, "reason": "guard: small talk"}


async def refuse(state: RAGState) -> RAGState:
    msg = MISCONDUCT_MESSAGE.format(exam_office=get_settings().guard_exam_office_contact)
    return _fixed_reply(msg, "guard: policy violation")


async def out_of_scope(state: RAGState) -> RAGState:
    return _fixed_reply(OUT_OF_SCOPE_MESSAGE.format(bot_name=get_settings().app_name), "guard: out of scope")


async def calm(state: RAGState) -> RAGState:
    return _fixed_reply(CALM_MESSAGE, "guard: abusive, no question")


async def prepare(state: RAGState) -> RAGState:
    q = contextualize(state["question"], state.get("history", []))
    return {"grade_query": q, "search_query": expand_abbreviations(q), "attempts": 0,
            "time_sensitive": is_time_sensitive(q)}


async def verified(state: RAGState) -> RAGState:
    s = get_settings()
    emb = await asyncio.to_thread(embed_query, state["grade_query"])
    match = await asyncio.to_thread(verified_index.best, emb)
    if match is None or match.score < s.verified_context_threshold:
        return {}
    await asyncio.to_thread(record_use, match.id)
    if match.score >= s.verified_direct_threshold:
        write = get_stream_writer()
        if match.source_url:
            write({"type": "sources", "sources": [{"title": "Official college answer", "url": match.source_url,
                                                  "date": ""}]})
        write({"type": "token", "text": match.answer})
        log.info("verified answer #%d used directly (score %.3f)", match.id, match.score)
        return {"answer": match.answer, "answered": True, "verified_direct": True, "reason": ""}
    hit = Hit(chunk_id=f"verified:{match.id}", score=match.score,
              text=f"Official answer from the college.\nQuestion: {match.question}\nAnswer: {match.answer}",
              metadata={"url": match.source_url or f"site://verified/{match.id}", "title": "Official college answer"},
              signals={"grade": match.score, "verified": True})
    log.info("verified answer #%d given to the LLM as context (score %.3f)", match.id, match.score)
    return {"verified_hit": hit}


def after_verified(state: RAGState) -> str:
    return END if state.get("verified_direct") else "retrieve"


def _add_usage(state: RAGState) -> dict:
    u = dict(state.get("usage") or {"calls": 0, "input_tokens": 0, "output_tokens": 0})
    last = get_llm().last_usage
    u["calls"] += 1
    u["input_tokens"] += last.input_tokens
    u["output_tokens"] += last.output_tokens
    return u


async def retrieve(state: RAGState) -> RAGState:
    emb = await asyncio.to_thread(embed_query, state["search_query"])
    r = await asyncio.to_thread(hybrid_search, state["search_query"], emb, None, state["grade_query"],
                                state.get("time_sensitive", False))
    log.info("retrieve attempt=%d best=%.3f relevant=%s hits=%s", state["attempts"] + 1, r.best_score, r.relevant,
             [(h.metadata.get("title", "")[:30], h.signals.get("grade")) for h in r.hits])
    return {"retrieval": r, "attempts": state["attempts"] + 1}


def after_retrieve(state: RAGState) -> str:
    if (state["retrieval"].relevant and state["retrieval"].hits) or state.get("verified_hit"):
        return "generate"
    if state["attempts"] < 2 and get_settings().rag_query_rewrite:
        return "rewrite"
    return "no_answer"


async def rewrite(state: RAGState) -> RAGState:
    get_stream_writer()({"type": "status", "text": "Searching more of the college website…"})
    try:
        async with llm_slot:
            new_q = await llm_rewrite(get_llm(), state["grade_query"])
            usage = _add_usage(state)
    except LLMError as e:
        log.warning("Query rewrite failed: %s", e)
        new_q, usage = state["grade_query"], state.get("usage")
    log.info("rewrite: %r -> %r", state["grade_query"], new_q)
    return {"search_query": expand_abbreviations(new_q), "usage": usage}


async def no_answer(state: RAGState) -> RAGState:
    if not is_about_college(state["grade_query"]):
        return _fixed_reply(OUT_OF_SCOPE_MESSAGE.format(bot_name=get_settings().app_name), "guard: out of scope")
    return _fixed_reply(no_info_message(), "no relevant source found")


async def generate(state: RAGState) -> RAGState:
    write = get_stream_writer()
    hits = list(state["retrieval"].hits)
    if state.get("verified_hit"):
        hits = [state["verified_hit"]] + hits[: max(0, len(hits) - 1)]
    write({"type": "sources", "sources": _sources(hits)})
    s = get_settings()
    history = state.get("history", [])[-2 * s.chat_history_turns:]
    messages = history + [ChatMessage(role="user", content=build_user_turn(state["question"], hits))]
    parts: list[str] = []
    try:
        async with llm_slot:
            async for delta in get_llm().stream(build_system_prompt(s.app_name), messages):
                parts.append(delta)
                write({"type": "token", "text": delta})
            usage = _add_usage(state)
    except LLMError as e:
        log.warning("LLM failure: %s", e)
        write({"type": "error", "text": BUSY_MESSAGE})
        return {"answer": "", "answered": False, "reason": f"llm error: {e}"}
    return {"answer": "".join(parts).strip(), "answered": True, "context_hits": hits, "usage": usage}


async def verify(state: RAGState) -> RAGState:
    """Only what the sources say: (1) every figure, date and contact must be in them (one corrected retry,
    then wrong lines are dropped); (2) every sentence must be stated in them (a second AI pass). Whatever is
    left is the answer; if nothing real is left, the student is sent to the college office."""
    answer = state.get("answer", "")
    if not answer:
        return {}
    write = get_stream_writer()
    if _just_a_refusal(answer):
        return _no_info(write, state.get("usage"), "model found no answer in the sources")
    s = get_settings()
    if not s.fact_check_enabled:
        return {}
    # Check against exactly what the model saw: chunk text plus each source's title, URL, date and year.
    hits = state.get("context_hits") or state["retrieval"].hits
    context = format_context(hits)
    usage = state.get("usage")

    fc = check_answer(answer, context, state["question"])
    if not fc.ok:
        log.warning("Fact check failed, unsupported: %s", fc.unsupported)
        # One more try, told exactly what was wrong. Small models often add up fee heads ("₹500 + ₹1,000 →
        # ₹1,500") while the rest of the answer is right; throwing it all away would lose a good answer.
        write({"type": "replace", "text": ""})
        write({"type": "status", "text": "Double-checking the figures…"})
        retry, usage = await _retry_figures(state, hits, answer, fc.unsupported, usage)
        if retry and not _just_a_refusal(retry):
            answer, fc = retry, check_answer(retry, context, state["question"])
            if not fc.ok:
                log.warning("Fact check failed again, unsupported: %s", fc.unsupported)
        if not fc.ok:  # still wrong: keep the lines whose facts check out, if a real answer remains
            trimmed = drop_unsupported_lines(answer, fc.unsupported)
            if not trimmed or not check_answer(trimmed, context, state["question"]).ok:
                return _no_info(write, usage, "fact check failed: " + ", ".join(fc.unsupported[:5]))
            answer = trimmed
        write({"type": "replace", "text": answer})

    if s.grounding_check:
        grounded, usage = await _grounded(state, context, answer, usage)
        if grounded != answer:
            if not grounded:
                return _no_info(write, usage, "not stated in the sources")
            answer = grounded
            write({"type": "replace", "text": answer})

    out = {"answer": answer, "usage": usage, **_stale_note(state, hits, answer)}
    if is_no_info_answer(answer):  # answered in part: the admin still sees the gap
        out.update(answered=False, reason="model found only part of the answer in the sources")
    return out


def _just_a_refusal(answer: str) -> bool:
    """'I don't have that information.' with nothing else of substance (a partial answer has more)."""
    return is_no_info_answer(answer) and len(answer) < 220 and not re.search(r"\d", answer)


def _no_info(write, usage, reason: str) -> RAGState:
    msg = no_info_message()
    write({"type": "replace", "text": msg})
    return {"answer": msg, "answered": False, "usage": usage, "reason": reason}


async def _retry_figures(state: RAGState, hits: list[Hit], answer: str, wrong: list[str], usage):
    s = get_settings()
    history = state.get("history", [])[-2 * s.chat_history_turns:]
    messages = history + [
        ChatMessage(role="user", content=build_user_turn(state["question"], hits)),
        ChatMessage(role="assistant", content=answer),
        ChatMessage(role="user", content=RETRY_PROMPT.format(wrong=", ".join(wrong[:5]))),
    ]
    parts: list[str] = []
    write = get_stream_writer()
    try:
        async with llm_slot:
            async for delta in get_llm().stream(build_system_prompt(s.app_name), messages):
                parts.append(delta)
                write({"type": "token", "text": delta})
            usage = _add_usage({**state, "usage": usage})
    except LLMError as e:
        log.warning("LLM failure on fact-check retry: %s", e)
        return "", usage
    return "".join(parts).strip(), usage


# Our own pointers ("Please contact the college office…") are advice, not claims about the college.
_OWN_ADVICE = re.compile(r"contact|college office|please (confirm|check|visit|refer)|for (more|further|the latest)",
                         re.I)


async def _grounded(state: RAGState, context: str, answer: str, usage):
    """The answer without the sentences a second AI pass finds are not stated in the sources ("" if nothing
    real is left). On an LLM failure the answer is kept: its figures were already checked."""
    units = [u for u in answer_units(answer) if len(content_words(u)) >= 3 and not _OWN_ADVICE.search(u)]
    if not units:
        return answer, usage
    numbered = "\n".join(f"{i}. {u.strip()}" for i, u in enumerate(units, 1))
    prompt = (f"<website_text>\n{context}\n</website_text>\n\nQuestion: {state['question']}\n\n"
              f"Answer sentences:\n{numbered}")
    try:
        async with llm_slot:
            res = await get_llm().generate(GROUNDING_PROMPT, [ChatMessage(role="user", content=prompt)])
            usage = _add_usage({**state, "usage": usage})
    except LLMError as e:
        log.warning("Grounding check skipped (LLM failure): %s", e)
        return answer, usage
    verdicts = {int(n): v.upper() for n, v in re.findall(r"(?m)^\W*(\d+)\W+(SUPPORTED|NOT)\b", res.text, re.I)}
    bad = [u for i, u in enumerate(units, 1) if verdicts.get(i) == "NOT"]
    if not bad:
        return answer, usage
    log.warning("Not stated in the sources, removed: %s", bad)
    return drop_unsupported_lines(answer, bad), usage


def _stale_note(state: RAGState, hits: list[Hit], answer: str) -> RAGState:
    """Time-sensitive answer built only on old sources: say how old, so nobody takes a 2025 fee notice
    as this year's. Added by code (not the model), so it can't be forgotten."""
    if not state.get("time_sensitive"):
        return {}
    dated = [(content_age_days(h.metadata), h) for h in hits if not h.chunk_id.startswith("verified:")]
    dated = [(a, h) for a, h in dated if a is not None]
    if not dated or min(a for a, _ in dated) <= FRESH_DAYS:
        return {}
    newest = min(dated, key=lambda t: t[0])[1].metadata
    when = _month_year(newest)
    note = STALE_NOTE.format(when=when)
    get_stream_writer()({"type": "token", "text": note})
    return {"answer": answer + note, "stale": True}


def _month_year(meta: dict) -> str:
    from datetime import datetime

    ay = str(meta.get("academic_year") or "")
    if meta.get("date"):
        when = datetime.fromisoformat(meta["date"]).strftime("%B %Y")
        return f"{when} (academic year {ay})" if ay else when
    return f"the academic year {ay}" if ay else "an earlier year"


def after_generate(state: RAGState) -> str:
    return "verify" if state.get("answer") else END


# ---------------------------------------------------------------- graph
@lru_cache
def build_graph():
    g = StateGraph(RAGState)
    g.add_node("guard", guard)
    g.add_node("refuse", refuse)
    g.add_node("out_of_scope", out_of_scope)
    g.add_node("calm", calm)
    g.add_node("small_talk", small_talk)
    g.add_node("prepare", prepare)
    g.add_node("verified", verified)
    g.add_node("retrieve", retrieve)
    g.add_node("rewrite", rewrite)
    g.add_node("no_answer", no_answer)
    g.add_node("generate", generate)
    g.add_node("verify", verify)
    g.add_edge(START, "guard")
    g.add_conditional_edges("guard", after_guard, ["prepare", "refuse", "out_of_scope", "calm", "small_talk"])
    for node in ("refuse", "out_of_scope", "calm", "small_talk"):
        g.add_edge(node, END)
    g.add_edge("prepare", "verified")
    g.add_conditional_edges("verified", after_verified, ["retrieve", END])
    g.add_conditional_edges("retrieve", after_retrieve, ["generate", "rewrite", "no_answer"])
    g.add_edge("rewrite", "retrieve")
    g.add_edge("no_answer", END)
    g.add_conditional_edges("generate", after_generate, ["verify", END])
    g.add_edge("verify", END)
    return g.compile()
