"""FastAPI application entry point: `uvicorn app.main:app` (one worker: the scheduler runs in-process)."""
from __future__ import annotations

import asyncio
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles

from app.admin.jobs import mark_interrupted_runs
from app.admin.routes import LoginRequired
from app.admin.routes import router as admin_router
from app.api.chat import router as chat_router
from app.api.widget import router as widget_router
from app.config import BASE_DIR, get_settings
from app.db.session import get_engine
from app.jobs import scheduler
from app.llm.factory import get_llm
from app.rag import vectorstore

settings = get_settings()
logging.basicConfig(level=settings.log_level, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
# Per-request HTTP logs from the HTTP client / Hugging Face are noise at INFO level.
for noisy in ("httpx", "httpcore", "huggingface_hub", "sentence_transformers", "apscheduler"):
    logging.getLogger(noisy).setLevel(logging.WARNING)


async def _warmup() -> None:
    """Load the embedding model and the LLM in the background so the first question is not slow."""
    log = logging.getLogger("app.warmup")
    try:
        from app.rag.embeddings import embed_query

        await asyncio.to_thread(embed_query, "warm up")
        from app.rag.bm25 import bm25_index

        await asyncio.to_thread(bm25_index.warmup)  # keyword index over all chunks
        await get_llm().warmup()
        log.info("Models loaded and ready")
        from app.i18n import translate

        await asyncio.to_thread(translate.warmup)  # IndicTrans2, when its models are downloaded
        log.info("Translation: %s", translate.provider())
    except Exception as e:  # the app still works; the first question just pays the load time
        log.warning("Warm-up skipped: %s", e)


@asynccontextmanager
async def lifespan(_app: FastAPI):
    get_engine()  # create / migrate tables
    mark_interrupted_runs()
    scheduler.start()
    warm = asyncio.create_task(_warmup())
    yield
    warm.cancel()
    scheduler.stop()


app = FastAPI(title=settings.app_name, docs_url="/api/docs" if not settings.is_production else None,
              redoc_url=None, lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type"],
)
app.include_router(chat_router)
app.include_router(widget_router)
app.include_router(admin_router)
app.mount("/static", StaticFiles(directory=BASE_DIR / "app" / "static"), name="static")


@app.middleware("http")
async def admin_security_headers(request: Request, call_next):
    response = await call_next(request)
    if request.url.path.startswith("/admin"):
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["Referrer-Policy"] = "same-origin"
        response.headers["Cache-Control"] = "no-store"
        response.headers["X-Robots-Tag"] = "noindex, nofollow"
        response.headers["Content-Security-Policy"] = (
            "default-src 'self'; img-src 'self' data:; style-src 'self'; script-src 'self'; "
            "form-action 'self'; frame-ancestors 'none'; base-uri 'none'")
    return response


@app.exception_handler(LoginRequired)
async def _login_required(request: Request, exc: LoginRequired):
    return RedirectResponse(f"/admin/login?next={exc.next_url}", status_code=303)


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
    return RedirectResponse("/demo")


@app.get("/widget.js", include_in_schema=False)
def widget_script():
    """The one file the college website includes: <script src="https://SERVER/widget.js" defer></script>."""
    return FileResponse(BASE_DIR / "app" / "static" / "widget" / "widget.js", media_type="application/javascript",
                        headers={"Cache-Control": "public, max-age=300", "X-Content-Type-Options": "nosniff"})


@app.get("/demo", include_in_schema=False)
def demo_page():
    """A page styled like sxca.edu.in with the widget on it, for local testing."""
    return FileResponse(BASE_DIR / "app" / "static" / "demo.html")


@app.get("/test", include_in_schema=False)
def test_page():
    return FileResponse(BASE_DIR / "app" / "static" / "test_chat.html")
