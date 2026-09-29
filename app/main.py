"""FastAPI application entry point: `uvicorn app.main:app`."""
from __future__ import annotations

import asyncio
import logging

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, RedirectResponse

from app.api.chat import router as chat_router
from app.config import BASE_DIR, get_settings
from app.db.session import get_engine
from app.llm.factory import get_llm
from app.rag import vectorstore

settings = get_settings()
logging.basicConfig(level=settings.log_level, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
# Per-request HTTP logs from the HTTP client / Hugging Face are noise at INFO level.
for noisy in ("httpx", "httpcore", "huggingface_hub", "sentence_transformers"):
    logging.getLogger(noisy).setLevel(logging.WARNING)

app = FastAPI(title=settings.app_name, docs_url="/api/docs" if settings.environment != "production" else None,
              redoc_url=None)
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type"],
)
app.include_router(chat_router)


async def _warmup() -> None:
    """Load the embedding model and the LLM in the background so the first question is not slow."""
    log = logging.getLogger("app.warmup")
    try:
        from app.rag.embeddings import embed_query

        await asyncio.to_thread(embed_query, "warm up")
        await get_llm().warmup()
        log.info("Models loaded and ready")
    except Exception as e:  # the app still works; the first question just pays the load time
        log.warning("Warm-up skipped: %s", e)


@app.on_event("startup")
async def _startup() -> None:
    get_engine()  # create tables
    asyncio.create_task(_warmup())


@app.get("/health")
async def health():
    llm = get_llm()
    llm_ok = await llm.health()
    return {
        "status": "ok" if llm_ok else "degraded",
        "llm": {**llm.describe(), "reachable": llm_ok},
        "embedding_model": settings.embedding_model,
        "indexed_chunks": vectorstore.count(),
    }


@app.get("/", include_in_schema=False)
def index():
    return RedirectResponse("/test")


@app.get("/test", include_in_schema=False)
def test_page():
    return FileResponse(BASE_DIR / "app" / "static" / "test_chat.html")
