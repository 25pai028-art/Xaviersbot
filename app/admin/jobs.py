"""Background work started from the admin panel or the scheduler: crawls, single URLs, uploads.

Only one crawl runs at a time (the CPU is shared with the chat). Uploads are indexed
immediately in their own thread. Progress is kept in memory for the admin page to poll.
"""
from __future__ import annotations

import logging
import re
import threading
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

from app.config import get_settings
from app.crawler.documents import extract_document, sniff_type
from app.crawler.metadata import best_date, detect_script_language, find_academic_year
from app.crawler.pipeline import CrawlStats, Crawler, QueueItem, crawl_single_url, text_hash
from app.crawler.urls import domain_allowed, is_public_ip_host, normalize_url
from app.db.models import Source, utcnow
from app.db.session import session_scope
from app.rag import indexer

log = logging.getLogger(__name__)

ALLOWED_UPLOADS = {".pdf": "pdf", ".docx": "docx", ".txt": "txt", ".png": "image", ".jpg": "image", ".jpeg": "image"}


@dataclass
class CrawlJob:
    kind: str  # full | url
    started_by: str
    started_at: datetime = field(default_factory=utcnow)
    target: str = ""
    running: bool = True
    error: str | None = None
    stats: CrawlStats = field(default_factory=CrawlStats)
    crawler: Crawler | None = None


class JobManager:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.crawl: CrawlJob | None = None

    # ------------------------------------------------------------ crawls
    def crawl_running(self) -> bool:
        return bool(self.crawl and self.crawl.running)

    def start_full_crawl(self, by: str) -> bool:
        with self._lock:
            if self.crawl_running():
                return False
            job = CrawlJob(kind="full", started_by=by)
            job.crawler = Crawler(trigger="admin" if by != "scheduler" else "schedule",
                                  on_progress=lambda st: setattr(job, "stats", st))
            self.crawl = job
        threading.Thread(target=self._run_full, args=(job,), daemon=True, name="crawl").start()
        return True

    def _run_full(self, job: CrawlJob) -> None:
        try:
            job.stats = job.crawler.run()
        except Exception as e:  # already logged by the crawler
            job.error = str(e)
        finally:
            job.running = False

    def start_url(self, url: str, by: str) -> str | None:
        """Crawl one URL. Returns an error message, or None when started."""
        s = get_settings()
        n = normalize_url(url)
        if not n:
            return "That is not a valid web address."
        if not domain_allowed(n, s.crawl_allowed_domains, s.crawl_blocked_domains):
            return "Only college domains can be added: " + ", ".join(s.crawl_allowed_domains)
        if not is_public_ip_host(n):  # SSRF guard: never fetch internal/private addresses
            return "That address points to a private or unreachable server."
        with self._lock:
            if self.crawl_running():
                return "A crawl is already running. Try again when it has finished."
            job = CrawlJob(kind="url", started_by=by, target=n)
            self.crawl = job
        threading.Thread(target=self._run_url, args=(job,), daemon=True, name="crawl-url").start()
        return None

    def _run_url(self, job: CrawlJob) -> None:
        try:
            job.stats = crawl_single_url(job.target, trigger="admin")
        except Exception as e:
            log.exception("Single URL crawl failed")
            job.error = str(e)
        finally:
            job.running = False

    def stop_crawl(self) -> bool:
        if self.crawl_running() and self.crawl.crawler:
            self.crawl.crawler.stop_event.set()
            return True
        return False

    def status(self) -> dict:
        job = self.crawl
        if job is None:
            return {"running": False}
        st = job.stats
        return {
            "running": job.running, "kind": job.kind, "target": job.target, "started_by": job.started_by,
            "started_at": job.started_at.isoformat(), "error": job.error, "phase": st.phase,
            "processed": st.processed, "queued": st.queued, "discovered": st.discovered, "new": st.new,
            "updated": st.updated, "unchanged": st.unchanged, "errors": st.errors, "skipped": st.skipped,
            "to_index": st.to_index, "indexed": st.indexed, "chunks": st.chunks_written, "current": st.current,
            "notes": st.notes,
        }

    # ------------------------------------------------------------ uploads
    def start_upload(self, source_id: int) -> None:
        threading.Thread(target=process_upload, args=(source_id,), daemon=True, name=f"upload-{source_id}").start()


jobs = JobManager()


def mark_interrupted_runs() -> int:
    """At startup no crawl can be running in this process: runs still marked 'running' were killed."""
    from sqlalchemy import update

    from app.db.models import CrawlRun

    with session_scope() as db:
        return db.execute(update(CrawlRun).where(CrawlRun.status == "running", CrawlRun.trigger != "cli")
                          .values(status="interrupted")).rowcount or 0


# ---------------------------------------------------------------- upload handling
def safe_filename(name: str) -> str:
    name = Path(name).name  # no directories
    stem, dot, ext = name.rpartition(".")
    stem = re.sub(r"[^A-Za-z0-9._ -]+", "_", stem or ext)[:100].strip(" ._") or "file"
    return f"{stem}.{ext.lower()}" if dot else stem


def validate_upload(filename: str, data: bytes) -> tuple[str | None, str]:
    """(error, kind). Checks extension, size and the real file type (magic bytes)."""
    s = get_settings()
    ext = Path(filename).suffix.lower()
    kind = ALLOWED_UPLOADS.get(ext)
    if kind is None:
        return "Only PDF, DOCX, TXT, PNG and JPG files can be uploaded.", ""
    if len(data) > s.upload_max_mb * 1024 * 1024:
        return f"The file is larger than {s.upload_max_mb:.0f} MB.", ""
    if not data:
        return "The file is empty.", ""
    real = sniff_type(data)
    if kind == "txt":
        if real is not None or b"\x00" in data[:4096]:
            return "This does not look like a plain text file.", ""
    elif real != kind:
        return f"The file content is not a real {ext[1:].upper()} file.", ""
    return None, kind


def save_upload(filename: str, data: bytes, kind: str, title: str, by: str) -> int:
    """Store the file outside the web root and create its Source. Returns the source id."""
    s = get_settings()
    s.uploads_dir.mkdir(parents=True, exist_ok=True)
    name = safe_filename(filename)
    uid = uuid.uuid4().hex[:12]
    (s.uploads_dir / f"{uid}_{name}").write_bytes(data)
    with session_scope() as db:
        src = Source(url=f"upload://{uid}/{name}", source_type="upload", content_type=kind,
                     title=(title.strip() or Path(name).stem.replace("_", " ").replace("-", " "))[:500],
                     section="Uploaded by the college", link_text=by, index_status="processing")
        db.add(src)
        db.flush()
        return src.id


def upload_path(src: Source) -> Path:
    uid, name = src.url.removeprefix("upload://").split("/", 1)
    return get_settings().uploads_dir / f"{uid}_{name}"


def process_upload(source_id: int) -> None:
    """Extract text (OCR for scans/images) and index the upload."""
    try:
        with session_scope() as db:
            src = db.get(Source, source_id)
            path = upload_path(src)
            doc = extract_document(path.read_bytes(), src.content_type)
            text = doc.text
            src.raw_text = src.text = text
            src.content_hash = text_hash(text) if text else ""
            src.language = detect_script_language(text)
            src.published_at = best_date(text=text) or utcnow()
            src.academic_year = find_academic_year(src.title, text)
            src.ocr_used = doc.ocr_used
            src.last_crawled_at = src.last_changed_at = utcnow()
            if len(text.strip()) < 40:
                src.index_status, src.chunk_count = "empty", 0
                src.error = ("No readable text found. If this is a scanned document or image, "
                             "install Tesseract OCR on the server." if not doc.ocr_used else "No readable text found.")
                return
            counts = indexer.index_sources([src])
            src.chunk_count = counts[src.id]
            src.index_status = "indexed" if src.chunk_count else "empty"
    except Exception as e:
        log.exception("Upload processing failed")
        with session_scope() as db:
            src = db.get(Source, source_id)
            if src:
                src.index_status, src.error = "error", str(e)[:500]


def remove_source(source_id: int) -> str:
    """Admin 'delete'. Uploads are deleted with their file; website pages are blocked so the next
    crawl does not add them back. Returns what happened."""
    with session_scope() as db:
        src = db.get(Source, source_id)
        if src is None:
            return "not found"
        indexer.remove_source(src)
        if src.source_type == "upload":
            try:
                upload_path(src).unlink(missing_ok=True)
            except (OSError, ValueError):
                pass
            db.delete(src)
            return "deleted"
        src.status, src.index_status, src.chunk_count = "blocked", "blocked", 0
        return "blocked"


def unblock_source(source_id: int) -> None:
    """Allow a blocked page again; it is indexed on the next crawl (or now via 'add URL')."""
    with session_scope() as db:
        src = db.get(Source, source_id)
        if src and src.status == "blocked":
            src.status, src.index_status, src.content_hash = "active", "pending", ""


__all__ = ["jobs", "validate_upload", "save_upload", "remove_source", "unblock_source", "QueueItem"]
