"""Builds the active LLM provider from settings (`LLM_PROVIDER` in .env)."""
from __future__ import annotations

from functools import lru_cache

from app.config import Settings, get_settings
from app.llm.base import LLMError, LLMProvider


def build_provider(settings: Settings) -> LLMProvider:
    common = dict(temperature=settings.llm_temperature, max_output_tokens=settings.llm_max_output_tokens)
    name = settings.llm_provider.lower().strip()

    if name == "ollama":
        from app.llm.ollama_provider import OllamaProvider

        return OllamaProvider(
            model=settings.ollama_model,
            base_url=settings.ollama_base_url,
            num_ctx=settings.ollama_num_ctx,
            keep_alive=settings.ollama_keep_alive,
            timeout=settings.llm_timeout_seconds,
            **common,
        )
    if name == "gemini":
        from app.llm.gemini_provider import GeminiProvider

        return GeminiProvider(
            model=settings.gemini_model,
            api_key=settings.gemini_api_key,
            thinking_budget=settings.gemini_thinking_budget,
            **common,
        )
    if name == "anthropic":
        from app.llm.anthropic_provider import AnthropicProvider

        return AnthropicProvider(
            model=settings.anthropic_model,
            api_key=settings.anthropic_api_key,
            effort=settings.anthropic_effort,
            timeout=settings.llm_timeout_seconds,
            **common,
        )
    if name == "openai":
        from app.llm.openai_provider import OpenAIProvider

        return OpenAIProvider(
            model=settings.openai_model,
            api_key=settings.openai_api_key,
            base_url=settings.openai_base_url,
            timeout=settings.llm_timeout_seconds,
            **common,
        )
    raise LLMError(f"Unknown LLM_PROVIDER '{settings.llm_provider}'. Use ollama, gemini, anthropic or openai.")


@lru_cache
def get_llm() -> LLMProvider:
    return build_provider(get_settings())
