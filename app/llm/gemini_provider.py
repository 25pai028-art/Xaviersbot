"""Google Gemini via the official `google-genai` SDK."""
from __future__ import annotations

from typing import AsyncIterator

from app.llm.base import ChatMessage, LLMError, LLMProvider, LLMResult, LLMUsage


class GeminiProvider(LLMProvider):
    name = "gemini"

    def __init__(self, *, api_key: str, thinking_budget: int, **kw):
        super().__init__(**kw)
        if not api_key:
            raise LLMError("GEMINI_API_KEY is not set in .env")
        from google import genai  # imported lazily so other providers don't need it

        self._genai = genai
        self._client = genai.Client(api_key=api_key)
        self.thinking_budget = thinking_budget

    def _config(self, system: str):
        types = self._genai.types
        cfg = dict(
            system_instruction=system,
            temperature=self.temperature,
            max_output_tokens=self.max_output_tokens,
        )
        if self.thinking_budget >= 0:
            cfg["thinking_config"] = types.ThinkingConfig(thinking_budget=self.thinking_budget)
        return types.GenerateContentConfig(**cfg)

    @staticmethod
    def _contents(messages: list[ChatMessage]) -> list[dict]:
        # Gemini calls the assistant role "model".
        return [
            {"role": "model" if m.role == "assistant" else "user", "parts": [{"text": m.content}]}
            for m in messages
        ]

    @staticmethod
    def _answer_text(chunk) -> str:
        """Text of a response/chunk, excluding any 'thought' parts."""
        out = []
        for cand in chunk.candidates or []:
            for part in (cand.content.parts if cand.content and cand.content.parts else []):
                if getattr(part, "thought", False):
                    continue
                if part.text:
                    out.append(part.text)
        return "".join(out)

    def _usage(self, resp) -> LLMUsage:
        um = getattr(resp, "usage_metadata", None)
        if not um:
            return LLMUsage()
        return LLMUsage(input_tokens=um.prompt_token_count or 0, output_tokens=um.candidates_token_count or 0)

    async def generate(self, system: str, messages: list[ChatMessage]) -> LLMResult:
        try:
            resp = await self._client.aio.models.generate_content(
                model=self.model, contents=self._contents(messages), config=self._config(system)
            )
        except Exception as e:  # SDK raises google.genai.errors.APIError subclasses
            raise LLMError(f"Gemini error: {e}", retryable=True) from e
        self.last_usage = self._usage(resp)
        return LLMResult(text=self._answer_text(resp).strip(), usage=self.last_usage, model=self.model)

    async def stream(self, system: str, messages: list[ChatMessage]) -> AsyncIterator[str]:
        self.last_usage = LLMUsage()
        try:
            it = await self._client.aio.models.generate_content_stream(
                model=self.model, contents=self._contents(messages), config=self._config(system)
            )
            async for chunk in it:
                text = self._answer_text(chunk)
                if text:
                    yield text
                usage = self._usage(chunk)
                if usage.input_tokens or usage.output_tokens:
                    self.last_usage = usage
        except LLMError:
            raise
        except Exception as e:
            raise LLMError(f"Gemini error: {e}", retryable=True) from e
