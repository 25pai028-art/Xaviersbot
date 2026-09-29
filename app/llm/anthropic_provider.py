"""Claude via the official `anthropic` SDK (production provider).

Model-family differences handled here:
- Claude Opus 5.5 / Sonnet 5.5 and other current models reject non-default
  sampling parameters, so `temperature` is only sent to Haiku 4.5 / older models.
- Thinking on the current models is adaptive and is steered with `effort`;
  thinking blocks are never shown to users (only text deltas are streamed).
- Opus 5.5 / Sonnet 5.5 requests opt into server-side refusal fallbacks
  (`fallbacks: "default"`), so a safety-classifier decline is retried on
  Anthropic's recommended fallback model instead of failing the answer.
"""
from __future__ import annotations

from typing import AsyncIterator

from app.llm.base import ChatMessage, LLMError, LLMProvider, LLMResult, LLMUsage

# Models that reject temperature/top_p/top_k and take `output_config.effort`.
_EFFORT_MODELS = ("claude-opus-5", "claude-sonnet-5", "claude-opus-4-7", "claude-opus-4-8", "claude-fable")
_FALLBACK_MODELS = ("claude-opus-5-5", "claude-sonnet-5-5", "claude-opus-5", "claude-fable-5-1")
_FALLBACK_BETA = "server-side-fallback-2026-07-01"

REFUSAL_TEXT = "I'm sorry, I can't help with that request."


class AnthropicProvider(LLMProvider):
    name = "anthropic"

    def __init__(self, *, api_key: str, effort: str, timeout: float, **kw):
        super().__init__(**kw)
        if not api_key:
            raise LLMError("ANTHROPIC_API_KEY is not set in .env")
        import anthropic  # lazy import

        self._sdk = anthropic
        self._client = anthropic.AsyncAnthropic(api_key=api_key, timeout=timeout, max_retries=2)
        self.effort = effort

    def _kwargs(self, system: str, messages: list[ChatMessage]) -> dict:
        kwargs: dict = {
            "model": self.model,
            "max_tokens": self.max_output_tokens,
            "system": system,
            "messages": [{"role": m.role, "content": m.content} for m in messages],
        }
        if self.model.startswith(_EFFORT_MODELS):
            kwargs["output_config"] = {"effort": self.effort}
        else:
            kwargs["temperature"] = self.temperature
        if self.model.startswith(_FALLBACK_MODELS):
            kwargs["betas"] = [_FALLBACK_BETA]
            kwargs["fallbacks"] = "default"
        return kwargs

    def _messages_api(self, kwargs: dict):
        return self._client.beta.messages if "betas" in kwargs else self._client.messages

    def _wrap(self, e: Exception) -> LLMError:
        sdk = self._sdk
        if isinstance(e, sdk.RateLimitError):
            return LLMError("Claude rate limit reached", retryable=True)
        if isinstance(e, sdk.APIStatusError):
            return LLMError(f"Claude API error {e.status_code}: {e.message}", retryable=e.status_code >= 500)
        if isinstance(e, sdk.APIConnectionError):
            return LLMError("Cannot reach the Claude API", retryable=True)
        return LLMError(f"Claude error: {e}")

    async def generate(self, system: str, messages: list[ChatMessage]) -> LLMResult:
        kwargs = self._kwargs(system, messages)
        try:
            resp = await self._messages_api(kwargs).create(**kwargs)
        except self._sdk.APIError as e:
            raise self._wrap(e) from e
        self.last_usage = LLMUsage(resp.usage.input_tokens, resp.usage.output_tokens)
        if resp.stop_reason == "refusal":
            return LLMResult(text=REFUSAL_TEXT, usage=self.last_usage, model=resp.model)
        text = "".join(b.text for b in resp.content if b.type == "text")
        return LLMResult(text=text.strip(), usage=self.last_usage, model=resp.model)

    async def stream(self, system: str, messages: list[ChatMessage]) -> AsyncIterator[str]:
        kwargs = self._kwargs(system, messages)
        self.last_usage = LLMUsage()
        try:
            async with self._messages_api(kwargs).stream(**kwargs) as stream:
                async for text in stream.text_stream:
                    yield text
                final = await stream.get_final_message()
        except self._sdk.APIError as e:
            raise self._wrap(e) from e
        self.last_usage = LLMUsage(final.usage.input_tokens, final.usage.output_tokens)
        if final.stop_reason == "refusal":
            yield "\n\n" + REFUSAL_TEXT
