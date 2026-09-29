"""The answering workflow as a LangGraph state machine.

    guard ─┬─ cheating / hacking ─► refuse ─► END
           ├─ out of scope ───────► out_of_scope ─► END
           ├─ abuse only ─────────► calm ─► END
           └─ ok (abuse stripped) ─► prepare ─► retrieve ─┬─ relevant ──► generate ─► verify ─► END
                                                           ├─ weak, first try ─► rewrite ─► retrieve
                                                           └─ weak after retry ─► no_answer ─► END

- guard:     rule-based input guardrails (no LLM): refuse misconduct with the official
             channel, redirect off-topic questions, de-escalate pure abuse, strip profanity
- prepare:   make follow-ups self-contained, expand abbreviations (BCA, HOD, CoE…)
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
from functools import lru_cache
from typing import TypedDict

from langgraph.config import get_stream_writer
from langgraph.graph import END, START, StateGraph

from app.config import get_settings
from app.guardrails.input import check_input, is_about_college
from app.llm.base import ChatMessage, LLMError
from app.llm.factory import get_llm
from app.rag.embeddings import embed_query
from app.rag.factcheck import check_answer, is_no_info_answer
from app.rag.prompts import (CALM_MESSAGE, MISCONDUCT_MESSAGE, NO_INFO_MESSAGE, OUT_OF_SCOPE_MESSAGE,
                             build_system_prompt, build_user_turn, format_context)
from app.rag.query import contextualize, expand_abbreviations, llm_rewrite
from app.rag.retriever import Retrieval, hybrid_search
from app.rag.vectorstore import Hit

log = logging.getLogger(__name__)

# One generation at a time on a CPU-only machine; others wait in line.
llm_slot = asyncio.Semaphore(1)

BUSY_MESSAGE = "The assistant is busy or temporarily unavailable. Please try again in a moment."


class RAGState(TypedDict, total=False):
    question: str  # after the guard: profanity removed
    guard: str  # ok | misconduct | off_topic | abuse_only
    history: list[ChatMessage]
    grade_query: str  # self-contained question in the user's words
    search_query: str  # grade_query + expansions / LLM rewrite
    attempts: int
    retrieval: Retrieval
    answer: str
    answered: bool
    reason: str  # why a question was not answered (for the admin's unanswered list)


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
    return {"question": g.question, "guard": g.kind}


def after_guard(state: RAGState) -> str:
    return {"misconduct": "refuse", "off_topic": "out_of_scope", "abuse_only": "calm"}.get(state["guard"], "prepare")


async def refuse(state: RAGState) -> RAGState:
    msg = MISCONDUCT_MESSAGE.format(exam_office=get_settings().guard_exam_office_contact)
    return _fixed_reply(msg, "guard: policy violation")


async def out_of_scope(state: RAGState) -> RAGState:
    return _fixed_reply(OUT_OF_SCOPE_MESSAGE.format(bot_name=get_settings().app_name), "guard: out of scope")


async def calm(state: RAGState) -> RAGState:
    return _fixed_reply(CALM_MESSAGE, "guard: abusive, no question")


async def prepare(state: RAGState) -> RAGState:
    q = contextualize(state["question"], state.get("history", []))
    return {"grade_query": q, "search_query": expand_abbreviations(q), "attempts": 0}


async def retrieve(state: RAGState) -> RAGState:
    emb = await asyncio.to_thread(embed_query, state["search_query"])
    r = await asyncio.to_thread(hybrid_search, state["search_query"], emb, None, state["grade_query"])
    log.info("retrieve attempt=%d best=%.3f relevant=%s hits=%s", state["attempts"] + 1, r.best_score, r.relevant,
             [(h.metadata.get("title", "")[:30], h.signals.get("grade")) for h in r.hits])
    return {"retrieval": r, "attempts": state["attempts"] + 1}


def after_retrieve(state: RAGState) -> str:
    if state["retrieval"].relevant and state["retrieval"].hits:
        return "generate"
    if state["attempts"] < 2 and get_settings().rag_query_rewrite:
        return "rewrite"
    return "no_answer"


async def rewrite(state: RAGState) -> RAGState:
    get_stream_writer()({"type": "status", "text": "Searching more of the college website…"})
    try:
        async with llm_slot:
            new_q = await llm_rewrite(get_llm(), state["grade_query"])
    except LLMError as e:
        log.warning("Query rewrite failed: %s", e)
        new_q = state["grade_query"]
    log.info("rewrite: %r -> %r", state["grade_query"], new_q)
    return {"search_query": expand_abbreviations(new_q)}


async def no_answer(state: RAGState) -> RAGState:
    if not is_about_college(state["grade_query"]):
        return _fixed_reply(OUT_OF_SCOPE_MESSAGE.format(bot_name=get_settings().app_name), "guard: out of scope")
    return _fixed_reply(NO_INFO_MESSAGE, "no relevant source found")


async def generate(state: RAGState) -> RAGState:
    write = get_stream_writer()
    hits = state["retrieval"].hits
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
    except LLMError as e:
        log.warning("LLM failure: %s", e)
        write({"type": "error", "text": BUSY_MESSAGE})
        return {"answer": "", "answered": False, "reason": f"llm error: {e}"}
    return {"answer": "".join(parts).strip(), "answered": True}


async def verify(state: RAGState) -> RAGState:
    answer = state.get("answer", "")
    if not answer:
        return {}
    if is_no_info_answer(answer):
        return {"answered": False, "reason": "model found no answer in the sources"}
    if not get_settings().fact_check_enabled:
        return {}
    # Check against exactly what the model saw: chunk text plus each source's title, URL, date and year.
    fc = check_answer(answer, format_context(state["retrieval"].hits), state["question"])
    if fc.ok:
        return {}
    log.warning("Fact check failed, unsupported: %s", fc.unsupported)
    get_stream_writer()({"type": "replace", "text": NO_INFO_MESSAGE})
    return {"answer": NO_INFO_MESSAGE, "answered": False,
            "reason": "fact check failed: " + ", ".join(fc.unsupported[:5])}


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
    g.add_node("prepare", prepare)
    g.add_node("retrieve", retrieve)
    g.add_node("rewrite", rewrite)
    g.add_node("no_answer", no_answer)
    g.add_node("generate", generate)
    g.add_node("verify", verify)
    g.add_edge(START, "guard")
    g.add_conditional_edges("guard", after_guard, ["prepare", "refuse", "out_of_scope", "calm"])
    for node in ("refuse", "out_of_scope", "calm"):
        g.add_edge(node, END)
    g.add_edge("prepare", "retrieve")
    g.add_conditional_edges("retrieve", after_retrieve, ["generate", "rewrite", "no_answer"])
    g.add_edge("rewrite", "retrieve")
    g.add_edge("no_answer", END)
    g.add_conditional_edges("generate", after_generate, ["verify", END])
    g.add_edge("verify", END)
    return g.compile()
