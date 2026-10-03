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
from app.guardrails.input import small_talk_kind
from app.i18n.languages import answer_language, detect_language
from app.i18n.messages import WELCOME
from app.i18n.translate import TranslationError, translate
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
    type: Literal["sources", "token", "replace", "status", "done", "error", "language"]
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


TRANSLATION_FAILED_NOTE = "\n\n(Sorry, I couldn't translate this answer reliably, so it is shown in English.)"
HISTORY_MESSAGES_TO_TRANSLATE = 2


async def answer_stream(question: str, history: list[ChatMessage] | None = None,
                        language: str = "auto") -> AsyncIterator[ChatEvent]:
    """Answer in the student's language. Non-English questions are translated to English, answered and
    fact-checked in English, and the finished answer is translated back (so it arrives at once, not
    word by word). Numbers, emails and links are verified after translation."""
    lang = answer_language(language, question)
    if lang == "en":
        async for ev in _answer_english(question, history):
            yield ev
        return

    yield ChatEvent(type="language", text=lang)
    # "Hi" / "who are you?" in any language: the hand-written introduction, at once (no translating, no search).
    if small_talk_kind(question) in ("greeting", "identity", "how_are_you") and lang in WELCOME:
        yield ChatEvent(type="token", text=WELCOME[lang])
        yield ChatEvent(type="done", answered=True)
        return
    q_lang = detect_language(question, preferred=lang)
    en_question = question
    if q_lang != "en":
        yield ChatEvent(type="status", text="Understanding your question…")
        try:
            en_question = await translate(question, q_lang, "en")
        except TranslationError:
            pass  # the embedding model is multilingual, so searching with the original still works
    en_history = await _history_in_english(history or [], lang)

    answer, done = "", ChatEvent(type="done", answered=False)
    async for ev in _answer_english(en_question, en_history):
        if ev.type == "token":
            answer += ev.text
        elif ev.type == "replace":
            answer = ev.text
        elif ev.type == "done":
            done = ev
        elif ev.type == "error":
            yield ChatEvent(type="error", text=await _translated_or_english(ev.text, lang))
        else:
            yield ev  # sources, status
    if answer.strip():
        yield ChatEvent(type="status", text="Translating the answer…")
        try:
            answer = await translate(answer.strip(), "en", lang)
        except TranslationError:
            answer = answer.strip() + TRANSLATION_FAILED_NOTE
        yield ChatEvent(type="token", text=answer)
    yield done


async def _translated_or_english(text: str, lang: str) -> str:
    try:
        return await translate(text, "en", lang)
    except TranslationError:
        return text


async def _history_in_english(history: list[ChatMessage], lang: str) -> list[ChatMessage]:
    """Follow-up questions need the previous turns in English. Only the last few user turns are
    translated (each costs time); earlier non-English assistant replies are left out."""
    out: list[ChatMessage] = []
    budget = HISTORY_MESSAGES_TO_TRANSLATE
    for m in reversed(history):
        src = detect_language(m.content, preferred=lang)
        if src == "en":
            out.append(m)
        elif m.role == "user" and budget > 0:
            budget -= 1
            try:
                out.append(ChatMessage(role="user", content=await translate(m.content, src, "en")))
            except TranslationError:
                pass
    return list(reversed(out))


async def _answer_english(question: str, history: list[ChatMessage] | None = None) -> AsyncIterator[ChatEvent]:
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
