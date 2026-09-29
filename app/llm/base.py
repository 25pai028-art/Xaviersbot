"""Provider-independent LLM interface.

Every provider implements `generate()` (full answer) and `stream()` (async text
deltas). Providers are responsible for hiding their own quirks: system-prompt
format, "thinking" output, token limits and error types.
"""
from __future__ import annotations

import re
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import AsyncIterator, Literal


@dataclass
class ChatMessage:
    role: Literal["user", "assistant"]
    content: str


@dataclass
class LLMUsage:
    input_tokens: int = 0
    output_tokens: int = 0


@dataclass
class LLMResult:
    text: str
    usage: LLMUsage = field(default_factory=LLMUsage)
    model: str = ""


class LLMError(Exception):
    """Raised when the provider is unreachable, overloaded or rejects the request."""

    def __init__(self, message: str, *, retryable: bool = False):
        super().__init__(message)
        self.retryable = retryable


class LLMProvider(ABC):
    name: str = "base"

    def __init__(self, model: str, temperature: float, max_output_tokens: int):
        self.model = model
        self.temperature = temperature
        self.max_output_tokens = max_output_tokens
        self.last_usage = LLMUsage()

    @abstractmethod
    async def generate(self, system: str, messages: list[ChatMessage]) -> LLMResult: ...

    @abstractmethod
    def stream(self, system: str, messages: list[ChatMessage]) -> AsyncIterator[str]:
        """Yield answer text deltas. Usage is available in `self.last_usage` afterwards."""

    async def health(self) -> bool:
        return True

    async def warmup(self) -> None:
        """Load the model ahead of the first question (no-op for hosted APIs)."""

    def describe(self) -> dict:
        return {"provider": self.name, "model": self.model}


_THINK_BLOCK = re.compile(r"<(think|thinking|reasoning)>.*?</\1>", re.DOTALL | re.IGNORECASE)


def strip_thinking(text: str) -> str:
    """Remove <think>…</think>-style reasoning some local models emit inline."""
    text = _THINK_BLOCK.sub("", text)
    # An unterminated opening tag means the whole rest is reasoning.
    text = re.sub(r"<(think|thinking|reasoning)>.*\Z", "", text, flags=re.DOTALL | re.IGNORECASE)
    # A stray closing tag (model started thinking without an opening tag).
    m = re.search(r"</(think|thinking|reasoning)>", text, flags=re.IGNORECASE)
    if m:
        text = text[m.end():]
    return text.strip()


class ThinkingStreamFilter:
    """Incrementally drops <think>…</think> sections from a token stream."""

    _OPEN = ("<think>", "<thinking>", "<reasoning>")
    _CLOSE = ("</think>", "</thinking>", "</reasoning>")

    def __init__(self) -> None:
        self._buf = ""
        self._inside = False

    def feed(self, delta: str) -> str:
        self._buf += delta
        out: list[str] = []
        while True:
            lower = self._buf.lower()
            tags = self._CLOSE if self._inside else self._OPEN
            hits = [(lower.find(t), t) for t in tags if lower.find(t) != -1]
            if hits:
                idx, tag = min(hits)
                if not self._inside:
                    out.append(self._buf[:idx])
                self._buf = self._buf[idx + len(tag):]
                self._inside = not self._inside
                continue
            # Hold back a possible partial tag at the end of the buffer.
            keep = max(
                (k for t in tags for k in range(1, len(t)) if lower.endswith(t[:k])),
                default=0,
            )
            cut = len(self._buf) - keep
            if not self._inside:
                out.append(self._buf[:cut])
            self._buf = self._buf[cut:]
            break
        return "".join(out)

    def flush(self) -> str:
        rest, self._buf = ("" if self._inside else self._buf), ""
        return rest
