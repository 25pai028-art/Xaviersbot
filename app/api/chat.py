"""Chat API: `POST /api/chat` streams the answer as Server-Sent Events.

The client keeps the conversation and sends recent turns with each request;
the server stores nothing (privacy decision).

SSE events: `status` → {text} (e.g. "searching more…"), `sources` → {sources:[{title,url,date}]},
`token` → {text}, `replace` → {text} (fact check failed: replace the whole answer),
`done` → {answered}, `error` → {message}.
"""
from __future__ import annotations

import json
from typing import Literal

from fastapi import APIRouter
from pydantic import BaseModel, Field, field_validator
from sse_starlette.sse import EventSourceResponse

from app.config import get_settings
from app.llm.base import ChatMessage
from app.rag.service import answer_stream, record_feedback

router = APIRouter(prefix="/api", tags=["chat"])


class HistoryItem(BaseModel):
    role: Literal["user", "assistant"]
    content: str = Field(max_length=4000)


class ChatRequest(BaseModel):
    message: str = Field(min_length=1)
    history: list[HistoryItem] = Field(default_factory=list, max_length=20)
    language: str = "auto"

    @field_validator("message")
    @classmethod
    def _limit(cls, v: str) -> str:
        v = v.strip()
        if not v:
            raise ValueError("Message is empty")
        if len(v) > get_settings().max_message_chars:
            raise ValueError(f"Message is too long (max {get_settings().max_message_chars} characters)")
        return v


class FeedbackRequest(BaseModel):
    rating: Literal["up", "down"]
    question: str = Field(default="", max_length=1000)
    sources: list[str] = Field(default_factory=list, max_length=10)


@router.post("/feedback", status_code=204)
async def feedback(req: FeedbackRequest) -> None:
    """Thumbs up/down on an answer. Only the question text and source links of a thumbs-down are kept."""
    record_feedback(req.rating, req.question.strip(), [u[:500] for u in req.sources])


@router.post("/chat")
async def chat(req: ChatRequest):
    history = [ChatMessage(role=h.role, content=h.content) for h in req.history]

    async def events():
        async for ev in answer_stream(req.message, history):
            if ev.type == "sources":
                data = {"sources": ev.sources}
            elif ev.type in ("token", "replace", "status"):
                data = {"text": ev.text}
            elif ev.type == "done":
                data = {"answered": ev.answered}
            else:
                data = {"message": ev.text}
            yield {"event": ev.type, "data": json.dumps(data, ensure_ascii=False)}

    return EventSourceResponse(events(), ping=15)
