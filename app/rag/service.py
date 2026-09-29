"""Channel-independent chat service.

Any channel (web widget, WhatsApp later, …) calls `answer_stream()` with the
question and the recent history it holds. Conversations are not stored; only the
text of questions the bot could not answer is kept (anonymous) for the admin, plus
anonymous daily counters for the dashboard.
The answering logic itself lives in the LangGraph workflow (`app.rag.graph`).
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import date
from typing import AsyncIterator, Literal

from app.config import get_settings
from app.db.models import NegativeFeedback, UnansweredQuestion, UsageDay
from app.db.session import session_scope
from app.llm.base import ChatMessage
from app.rag.graph import BUSY_MESSAGE, build_graph

log = logging.getLogger(__name__)

# USD per million tokens (input, output) for known paid models. Override with LLM_PRICE_* in .env.
KNOWN_PRICES = {
    "claude-opus-5-5": (4.0, 20.0),
    "claude-sonnet-5-5": (2.0, 10.0),
    "claude-haiku-4-5": (1.0, 5.0),
}


@dataclass
class ChatEvent:
    type: Literal["sources", "token", "replace", "status", "done", "error"]
    text: str = ""
    sources: list[dict] = field(default_factory=list)
    answered: bool = True


def estimate_cost(input_tokens: int, output_tokens: int) -> float:
    s = get_settings()
    if s.llm_provider == "ollama":
        return 0.0
    pin, pout = s.llm_price_input_per_mtok, s.llm_price_output_per_mtok
    if not (pin or pout):
        model = {"anthropic": s.anthropic_model, "gemini": s.gemini_model, "openai": s.openai_model}.get(
            s.llm_provider, "")
        pin, pout = KNOWN_PRICES.get(model, (0.0, 0.0))
    return (input_tokens * pin + output_tokens * pout) / 1_000_000


def _count(**increments) -> None:
    """Add to today's anonymous counters."""
    try:
        with session_scope() as db:
            row = db.get(UsageDay, date.today())
            if row is None:
                row = UsageDay(day=date.today())
                db.add(row)
            for k, v in increments.items():
                setattr(row, k, (getattr(row, k) or 0) + v)
    except Exception:  # statistics must never break the chat
        log.exception("Could not update usage counters")


def _record_unanswered(question: str, reason: str) -> None:
    try:
        with session_scope() as db:
            db.add(UnansweredQuestion(question=question[:1000], reason=(reason or "")[:300]))
    except Exception:
        log.exception("Could not record unanswered question")


def record_feedback(rating: str, question: str, sources: list[str]) -> None:
    """Thumbs up/down. For thumbs-down, the question text and source links are kept (no answer, IP, session)."""
    if rating == "up":
        _count(thumbs_up=1)
        return
    _count(thumbs_down=1)
    with session_scope() as db:
        db.add(NegativeFeedback(question=question[:1000], sources="\n".join(sources[:10])[:4000]))


async def answer_stream(question: str, history: list[ChatMessage] | None = None) -> AsyncIterator[ChatEvent]:
    final: dict = {}
    try:
        async for mode, chunk in build_graph().astream(
            {"question": question, "history": history or []}, stream_mode=["custom", "values"]
        ):
            if mode == "values":
                final = chunk
                continue
            if chunk["type"] == "sources":
                yield ChatEvent(type="sources", sources=chunk["sources"])
            else:
                yield ChatEvent(type=chunk["type"], text=chunk.get("text", ""))
    except Exception:
        log.exception("Answer pipeline failed")
        yield ChatEvent(type="error", text=BUSY_MESSAGE)
        return
    answered = bool(final.get("answered"))
    reason = final.get("reason") or ""
    usage = final.get("usage") or {}
    guarded = reason.startswith("guard:")
    _count(
        chats=0 if guarded and reason.endswith("small talk") else 1,
        unanswered=0 if answered or guarded else 1,
        verified_hits=1 if final.get("verified_direct") or final.get("verified_hit") else 0,
        llm_calls=usage.get("calls", 0),
        input_tokens=usage.get("input_tokens", 0),
        output_tokens=usage.get("output_tokens", 0),
        cost_usd=estimate_cost(usage.get("input_tokens", 0), usage.get("output_tokens", 0)),
    )
    # Log genuine content gaps only (not guardrail refusals, off-topic questions or outages).
    if not answered and reason and not reason.startswith(("llm error", "guard:")):
        _record_unanswered(final.get("question", question), reason)
    yield ChatEvent(type="done", answered=answered)
