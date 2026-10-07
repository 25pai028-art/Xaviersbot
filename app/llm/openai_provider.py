"""OpenAI (or any OpenAI-compatible endpoint) via the official `openai` SDK. Optional."""
from __future__ import annotations

from typing import AsyncIterator

from app.llm.base import ChatMessage, LLMError, LLMProvider, LLMResult, LLMUsage, ThinkingStreamFilter, strip_thinking


class OpenAIProvider(LLMProvider):
    name = "openai"

    def __init__(self, *, api_key: str, base_url: str, timeout: float, reasoning_effort: str = "", **kw):
        super().__init__(**kw)
        # Reasoning models (gpt-oss) spend output tokens thinking; "low" keeps a short answer from coming back empty.
        self._extra = {"reasoning_effort": reasoning_effort} if reasoning_effort else {}
        if not api_key:
            raise LLMError("OPENAI_API_KEY is not set in .env")
        import openai  # lazy import

        self._sdk = openai
        self._client = openai.AsyncOpenAI(api_key=api_key, base_url=base_url or None, timeout=timeout)

    def _messages(self, system: str, messages: list[ChatMessage]) -> list[dict]:
        return [{"role": "system", "content": system}] + [{"role": m.role, "content": m.content} for m in messages]

    async def generate(self, system: str, messages: list[ChatMessage]) -> LLMResult:
        try:
            resp = await self._client.chat.completions.create(
                model=self.model,
                messages=self._messages(system, messages),
                temperature=self.temperature,
                max_completion_tokens=self.max_output_tokens,
                extra_body=self._extra or None,
            )
        except self._sdk.OpenAIError as e:
            raise LLMError(f"OpenAI error: {e}", retryable=True) from e
        if resp.usage:
            self.last_usage = LLMUsage(resp.usage.prompt_tokens, resp.usage.completion_tokens)
        text = strip_thinking(resp.choices[0].message.content or "")
        return LLMResult(text=text, usage=self.last_usage, model=self.model)

    async def stream(self, system: str, messages: list[ChatMessage]) -> AsyncIterator[str]:
        filt = ThinkingStreamFilter()
        self.last_usage = LLMUsage()
        try:
            stream = await self._client.chat.completions.create(
                model=self.model,
                messages=self._messages(system, messages),
                temperature=self.temperature,
                max_completion_tokens=self.max_output_tokens,
                stream=True,
                stream_options={"include_usage": True},
                extra_body=self._extra or None,
            )
            async for chunk in stream:
                if chunk.usage:
                    self.last_usage = LLMUsage(chunk.usage.prompt_tokens, chunk.usage.completion_tokens)
                if chunk.choices and chunk.choices[0].delta.content:
                    text = filt.feed(chunk.choices[0].delta.content)
                    if text:
                        yield text
        except self._sdk.OpenAIError as e:
            raise LLMError(f"OpenAI error: {e}", retryable=True) from e
        tail = filt.flush()
        if tail:
            yield tail
