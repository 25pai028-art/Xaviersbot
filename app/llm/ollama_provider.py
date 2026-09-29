"""Local models through Ollama's HTTP API (`/api/chat`)."""
from __future__ import annotations

import json
from typing import AsyncIterator

import httpx

from app.llm.base import ChatMessage, LLMError, LLMProvider, LLMResult, LLMUsage, ThinkingStreamFilter, strip_thinking


class OllamaProvider(LLMProvider):
    name = "ollama"

    def __init__(self, *, base_url: str, num_ctx: int, keep_alive: str, timeout: float, **kw):
        super().__init__(**kw)
        self.base_url = base_url.rstrip("/")
        self.num_ctx = num_ctx
        self.keep_alive = keep_alive
        self.timeout = timeout
        # Thinking models (qwen3.x) accept `think: false`; older models reject the field.
        self._send_think_flag = True

    def _payload(self, system: str, messages: list[ChatMessage], stream: bool) -> dict:
        payload = {
            "model": self.model,
            "stream": stream,
            "keep_alive": self.keep_alive,
            "messages": [{"role": "system", "content": system}]
            + [{"role": m.role, "content": m.content} for m in messages],
            "options": {
                "temperature": self.temperature,
                "num_ctx": self.num_ctx,
                "num_predict": self.max_output_tokens,
            },
        }
        if self._send_think_flag:
            payload["think"] = False
        return payload

    async def _post(self, client: httpx.AsyncClient, payload: dict) -> httpx.Response:
        req = client.build_request("POST", f"{self.base_url}/api/chat", json=payload)
        resp = await client.send(req, stream=True)
        if resp.status_code == 400 and "think" in payload:
            body = (await resp.aread()).decode(errors="ignore")
            await resp.aclose()
            if "think" in body.lower():
                self._send_think_flag = False
                payload.pop("think")
                return await self._post(client, payload)
            raise LLMError(f"Ollama rejected request: {body[:300]}")
        if resp.status_code == 404:
            await resp.aclose()
            raise LLMError(f"Ollama model '{self.model}' not found. Run: ollama pull {self.model}")
        if resp.status_code >= 400:
            body = (await resp.aread()).decode(errors="ignore")
            await resp.aclose()
            raise LLMError(f"Ollama error {resp.status_code}: {body[:300]}", retryable=resp.status_code >= 500)
        return resp

    async def generate(self, system: str, messages: list[ChatMessage]) -> LLMResult:
        parts = [chunk async for chunk in self.stream(system, messages)]
        return LLMResult(text="".join(parts).strip(), usage=self.last_usage, model=self.model)

    async def stream(self, system: str, messages: list[ChatMessage]) -> AsyncIterator[str]:
        filt = ThinkingStreamFilter()
        self.last_usage = LLMUsage()
        try:
            async with httpx.AsyncClient(timeout=httpx.Timeout(self.timeout, connect=5.0)) as client:
                resp = await self._post(client, self._payload(system, messages, stream=True))
                try:
                    async for line in resp.aiter_lines():
                        if not line.strip():
                            continue
                        data = json.loads(line)
                        if data.get("error"):
                            raise LLMError(f"Ollama: {data['error']}")
                        # `thinking` field (if any) is ignored on purpose; only `content` is the answer.
                        delta = data.get("message", {}).get("content", "")
                        if delta:
                            text = filt.feed(delta)
                            if text:
                                yield text
                        if data.get("done"):
                            self.last_usage = LLMUsage(
                                input_tokens=data.get("prompt_eval_count", 0),
                                output_tokens=data.get("eval_count", 0),
                            )
                finally:
                    await resp.aclose()
        except httpx.ConnectError as e:
            raise LLMError(f"Cannot reach Ollama at {self.base_url}. Is it running?", retryable=True) from e
        except httpx.TimeoutException as e:
            raise LLMError("Ollama timed out (the model may be busy or too large for this machine).", retryable=True) from e
        tail = filt.flush()
        if tail:
            yield strip_thinking(tail)

    async def warmup(self) -> None:
        # A chat request with no messages just loads the model into RAM (~10 s on CPU).
        async with httpx.AsyncClient(timeout=httpx.Timeout(self.timeout, connect=5.0)) as client:
            await client.post(f"{self.base_url}/api/chat",
                              json={"model": self.model, "messages": [], "keep_alive": self.keep_alive})

    async def health(self) -> bool:
        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                r = await client.get(f"{self.base_url}/api/tags")
                return r.status_code == 200
        except httpx.HTTPError:
            return False
