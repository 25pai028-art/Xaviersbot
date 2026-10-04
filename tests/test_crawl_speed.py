"""Parallel downloads, deferred OCR and skipping unchanged files, on a small fake website (no network)."""
import io
import threading
import time

import pytest
from sqlalchemy import select

from app.config import get_settings
from app.crawler import discovery, documents, pipeline
from app.crawler.fetcher import Fetcher, FetchResult
from app.db.models import Source
from app.db.session import session_scope

SITE = "https://speedtest.sxca.edu.in"
FILLER = "The college offers undergraduate and postgraduate programmes in commerce, arts and science. " * 4
SCAN_TEXT = "SCANNED NOTICE: the last date to pay the semester fees is 25 June 2026 for all programmes."


def scanned_pdf() -> bytes:
    import pymupdf as fitz
    from PIL import Image

    doc = fitz.open()
    doc.new_page().insert_text((72, 72), "Fee notice for the academic year 2026-27, typed page with real text.")
    png = io.BytesIO()
    Image.new("L", (800, 1000), 255).save(png, "PNG")
    doc.new_page().insert_image(fitz.Rect(0, 0, 595, 842), stream=png.getvalue())  # a scan: image, no text
    return doc.tobytes()


def html(body: str) -> bytes:
    return f"<html><head><title>T</title></head><body><main><p>{FILLER}</p>{body}</main></body></html>".encode()


@pytest.fixture
def site(monkeypatch):
    pdf = scanned_pdf()
    pages = {
        f"{SITE}/": html(''.join(f'<a href="/p{i}/">Page {i}</a>' for i in range(8))
                         + '<a href="/wp-content/uploads/2026/06/fee-notice.pdf">Fee notice 2026-27</a>'
                         + '<a href="/wp-content/uploads/2023/01/old.pdf">Old notice</a>'),
        **{f"{SITE}/p{i}/": html(f"<p>Page {i} details.</p>") for i in range(8)},
        f"{SITE}/wp-content/uploads/2026/06/fee-notice.pdf": pdf,
    }
    calls = {"fetched": [], "active": 0, "max_active": 0, "ocr": 0}
    lock = threading.Lock()

    def fake_fetch(self, url, *, etag=None, last_modified=None):
        with lock:
            calls["fetched"].append(url)
            calls["active"] += 1
            calls["max_active"] = max(calls["max_active"], calls["active"])
        time.sleep(0.05)  # a slow college server
        with lock:
            calls["active"] -= 1
        body = pages.get(url)
        if body is None:
            return FetchResult(url=url, status=404, error="HTTP 404")
        ctype = "application/pdf" if url.endswith(".pdf") else "text/html"
        return FetchResult(url=url, status=200, content=body, content_type=ctype)

    def fake_ocr(_image):
        calls["ocr"] += 1
        return SCAN_TEXT

    monkeypatch.setattr(Fetcher, "fetch", fake_fetch)
    monkeypatch.setattr(Fetcher, "allowed_by_robots", lambda self, url: True)
    monkeypatch.setattr(discovery, "sitemap_seeds", lambda fetcher, site: [])
    monkeypatch.setattr(discovery, "wp_api_seeds", lambda fetcher, site, year: [])
    monkeypatch.setattr(documents, "ocr_image", fake_ocr)
    monkeypatch.setattr(pipeline, "ocr_available", lambda: (True, "eng"))
    monkeypatch.setattr(pipeline.indexer, "index_sources", lambda srcs: {s.id: 1 for s in srcs})
    monkeypatch.setattr(pipeline.indexer, "remove_source", lambda src: None)
    monkeypatch.setattr(pipeline.Crawler, "_cleanup_removed", lambda self: None)
    return calls


def settings(**extra):
    return get_settings().model_copy(update={
        "crawl_start_urls": [f"{SITE}/"], "crawl_delay_seconds": 0.0, "crawl_concurrency": 4,
        "crawl_use_playwright": False, "crawl_min_upload_year": 2025, "crawl_defer_ocr": True,
        "crawl_allowed_domains": ["speedtest.sxca.edu.in"], **extra})


def notice() -> Source:
    with session_scope() as db:
        return db.scalar(select(Source).where(Source.url == f"{SITE}/wp-content/uploads/2026/06/fee-notice.pdf"))


def test_downloads_run_in_parallel_and_old_uploads_are_skipped(site):
    stats = pipeline.Crawler(settings()).run()
    assert site["max_active"] > 1  # several downloads at once
    assert stats.processed == 10 and stats.errors == 0
    assert not any("2023" in u for u in site["fetched"])  # before CRAWL_MIN_UPLOAD_YEAR


def test_scanned_pages_are_read_after_everything_else(site, monkeypatch):
    phases = []
    crawler = pipeline.Crawler(settings(), on_progress=lambda s: phases.append(s.phase))
    original_ocr_stage = crawler._ocr_stage

    def ocr_stage():
        # By the time OCR starts, the typed text is already stored and indexed.
        src = notice()
        assert src.ocr_pending and src.index_status == "indexed" and "typed page" in src.text
        assert SCAN_TEXT not in src.text
        original_ocr_stage()

    monkeypatch.setattr(crawler, "_ocr_stage", ocr_stage)
    stats = crawler.run()
    src = notice()
    assert stats.ocr_deferred == 1 and stats.ocr_done == 1
    assert not src.ocr_pending and src.ocr_used and SCAN_TEXT in src.text and src.index_status == "indexed"
    assert phases.index("reading scanned documents") > phases.index("indexing")

    # Next crawl: the same file again (no 304 from the server) is neither re-read nor OCR'd again.
    ocr_calls = site["ocr"]
    stats = pipeline.Crawler(settings()).run()
    assert site["ocr"] == ocr_calls and stats.ocr_deferred == 0
    assert SCAN_TEXT in notice().text


def test_ocr_inline_when_not_deferred():
    pdf = scanned_pdf()
    later = documents.extract_document(pdf, "pdf", ocr=False)
    assert later.needs_ocr and "typed page" in later.text


def test_requests_to_one_host_stay_spaced_out_across_threads():
    # Measured by the slots handed out and the total time, not by when each thread reports back: on a busy
    # machine a thread can be descheduled after its turn, which would make two turns look close together.
    f = Fetcher(settings(crawl_delay_seconds=0.1))
    begin = time.monotonic()
    try:
        threads = [threading.Thread(target=f._wait_turn, args=("sxca.edu.in",)) for _ in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        last_slot = f._last_hit["sxca.edu.in"]
    finally:
        f.close()
    assert last_slot - begin >= 0.39  # 5 requests: slots at +0, +0.1, +0.2, +0.3, +0.4
    assert time.monotonic() - begin >= 0.38  # and the last one really waited for its slot


def test_browser_rendering_given_up_on_a_site_where_it_adds_nothing(site, monkeypatch):
    renders = []

    def render(self, url):
        renders.append(url)
        return html("").decode()  # the same text the plain download had: JavaScript adds nothing here

    monkeypatch.setattr(Fetcher, "render_with_browser", render)
    pipeline.Crawler(settings(crawl_use_playwright=True, crawl_playwright_min_chars=100_000)).run(ocr=False)
    assert len(renders) == pipeline.RENDER_PROBES  # 9 short pages, but only the first few were tried
