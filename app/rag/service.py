"""Channel-independent chat service.

Any channel (web widget, WhatsApp later, …) calls `answer_stream()` with the
question and the recent history it holds. Conversations are not stored; only the
text of questions the bot could not answer is kept (anonymous) for the admin.
The answering logic itself lives in the LangGraph workflow (`app.rag.graph`).
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import AsyncIterator, Literal

from app.db.models import UnansweredQuestion
from app.db.session import session_scope
from app.llm.base import ChatMessage
from app.rag.graph import BUSY_MESSAGE, build_graph

log = logging.getLogger(__name__)


@dataclass
class ChatEvent:
    type: Literal["sources", "token", "replace", "status", "done", "error"]
    text: str = ""
    sources: list[dict] = field(default_factory=list)
    answered: bool = True


def _record_unanswered(question: str, reason: str) -> None:
    try:
        with session_scope() as db:
            db.add(UnansweredQuestion(question=question[:1000], reason=(reason or "")[:300]))
    except Exception:  # logging must never break the chat
        log.exception("Could not record unanswered question")


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
    # Log genuine content gaps only (not guardrail refusals, off-topic questions or outages).
    if not answered and reason and not reason.startswith(("llm error", "guard:")):
        _record_unanswered(final.get("question", question), reason)
    yield ChatEvent(type="done", answered=answered)
