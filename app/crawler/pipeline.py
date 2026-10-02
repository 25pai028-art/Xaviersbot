"""Deep-crawl pipeline, in three stages:

1. CRAWL  – discover URLs (sitemaps, WordPress REST API, every link on every page,
            iframes / PDF viewers / onclick targets, links inside PDFs), fetch them
            politely, render JavaScript pages with Chromium when needed, and extract
            clean text from HTML, PDF, DOCX and images (OCR). Only changed content is
            marked for indexing (content hash, HTTP 304, WordPress `modified` date).
2. CLEAN  – remove text repeated across many pages (site-wide boilerplate), keeping
            one copy in a "common site information" source.
3. INDEX  – skip empty and duplicate documents, chunk (heading- and section-aware),
            embed with bge-m3 in batches and store in ChromaDB.

Stages 2–3 work from the database, so an interrupted run resumes indexing where it
stopped (`index_status = "pending"`). Errors are logged per URL and never stop the crawl.
"""
from __future__ import annotations

import hashlib
import logging
import os
import re
import threading
from collections import Counter, deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable
from urllib.parse import unquote, urlparse

from sqlalchemy import select

from app.config import Settings, get_settings
from app.crawler import discovery
from app.crawler.boilerplate import (COMMON_SOURCE_TITLE, COMMON_SOURCE_URL, common_information, find_boilerplate,
                                     site_information, strip_boilerplate)
from app.crawler.documents import extract_document, ocr_available
from app.crawler.extract_html import extract_html
from app.crawler.fetcher import Fetcher
from app.crawler.metadata import best_date, detect_script_language, find_academic_year
from app.crawler.urls import (classify, domain_allowed, google_drive_download_url, host_of, is_google_drive,
                              normalize_url, upload_year)
from app.db.models import CrawlError, CrawlRun, Source, utcnow
from app.db.session import session_scope
from app.rag import indexer

log = logging.getLogger(__name__)

CONTENT_TYPE_KIND = {
    "application/pdf": "pdf",
    "application/vnd.openxmlformats-officedocument.wordprocessingml.document": "docx",
    "text/plain": "txt",
    "image/jpeg": "image",
    "image/png": "image",
    "text/html": "html",
    "application/xhtml+xml": "html",
}
MIN_TEXT_CHARS = 40
INDEX_BATCH_SOURCES = 16
GENERIC_LINK_TEXT = re.compile(
    r"^(click here|here|download|view|view pdf|pdf|read more|know more|more|open|link|file|document|notice|"
    r"circular|details|view details|see more|image|jpg|png)\W*$", re.I)


@dataclass
class QueueItem:
    url: str
    depth: int = 0
    remote_modified: str | None = None
    title: str | None = None  # from the WordPress API / sitemap
    parent_url: str | None = None  # page that links here
    link_text: str | None = None  # anchor text on that page


@dataclass
class CrawlStats:
    phase: str = "starting"
    discovered: int = 0
    processed: int = 0
    new: int = 0
    updated: int = 0
    unchanged: int = 0
    skipped: int = 0
    errors: int = 0
    deleted: int = 0
    boilerplate_lines: int = 0
    to_index: int = 0
    indexed: int = 0
    empty: int = 0
    duplicates: int = 0
    chunks_written: int = 0
    queued: int = 0
    current: str = ""
    limit_reached: bool = False
    notes: list[str] = field(default_factory=list)

    def as_dict(self) -> dict:
        return {k: v for k, v in self.__dict__.items() if k != "current"}


def text_hash(text: str) -> str:
    return hashlib.sha256(re.sub(r"\s+", " ", text).strip().encode("utf-8")).hexdigest()


def title_from_url(url: str) -> str:
    name = unquote(urlparse(url).path.rstrip("/").rsplit("/", 1)[-1])
    name = re.sub(r"\.(pdf|docx|txt|jpe?g|png)$", "", name, flags=re.I)
    return re.sub(r"[-_]+", " ", name).strip().title() or url


def _alnum(s: str) -> str:
    return re.sub(r"[^a-z0-9]", "", s.lower())


def choose_document_title(url: str, *, link_text: str | None, api_title: str | None, doc_title: str | None) -> str:
    """Best human title for a PDF/DOCX: the link text on the college page usually beats file names."""
    filename = _alnum(title_from_url(url))

    def tidy(t: str | None) -> str:
        # PDFs exported from Word/Excel carry titles like "Microsoft Word - Exam Timetable.docx".
        t = re.sub(r"^Microsoft (Word|Excel|PowerPoint)\s*-\s*", "", (t or "").strip(), flags=re.I)
        return re.sub(r"\.(docx?|xlsx?|pptx?|pdf)$", "", t, flags=re.I).strip()

    def good(t: str) -> bool:
        return (len(t) >= 6 and not GENERIC_LINK_TEXT.match(t) and _alnum(t) != filename
                and not re.match(r"^(untitled|document\d*)\b", t, re.I))

    for cand in (tidy(link_text), tidy(api_title), tidy(doc_title)):
        if good(cand):
            return cand[:300]
    return title_from_url(url)


class CrashGuard:
    """Remembers the URL being processed in a small file. If the process dies on it (out of memory, killed
    runtime), the file still names that URL at the next start; it is then skipped for good (listed in
    crawl-skip.txt; delete the line to try it again). A normal stop or Ctrl+C clears the file, so only
    real crashes count."""

    def __init__(self, state_dir: Path):
        state_dir.mkdir(parents=True, exist_ok=True)
        self.current = state_dir / "crawl-current.txt"
        self.skip_file = state_dir / "crawl-skip.txt"
        self.crashed: str | None = None
        if self.current.exists():
            url = self.current.read_text(encoding="utf-8").strip()
            if url:
                self.crashed = url
                with self.skip_file.open("a", encoding="utf-8") as f:
                    f.write(url + "\n")
            self._write("")
        self.skip: set[str] = set()
        if self.skip_file.exists():
            self.skip = {l.strip() for l in self.skip_file.read_text(encoding="utf-8").splitlines() if l.strip()}

    def _write(self, text: str) -> None:
        with self.current.open("w", encoding="utf-8") as f:
            f.write(text)
            f.flush()
            os.fsync(f.fileno())  # must be on disk before the risky work starts

    def start(self, url: str) -> None:
        self._write(url)

    def done(self) -> None:
        self._write("")


class Crawler:
    def __init__(self, settings: Settings | None = None, trigger: str = "cli",
                 on_progress: Callable[[CrawlStats], None] | None = None, reindex: bool = False):
        self.s = settings or get_settings()
        self.trigger = trigger
        # Re-extract, re-chunk and re-embed everything (after changing extraction/chunk/embedding settings).
        self.reindex = reindex
        self.on_progress = on_progress or (lambda _stats: None)
        self.stats = CrawlStats()
        self.stop_event = threading.Event()
        self.fetcher = Fetcher(self.s)
        self.run_id: int | None = None
        self._pages: deque[QueueItem] = deque()
        self._docs: deque[QueueItem] = deque()
        self._seen: set[str] = set()
        self._queued: dict[str, QueueItem] = {}  # not yet processed, so link context can still be improved
        self._page_context: dict[str, str] = {}  # page url -> "Section › Title" (section for linked documents)
        self._ocr_ok = ocr_available()[0]
        self.guard = CrashGuard(Path(self.s.crawl_state_dir) if self.s.crawl_state_dir else self.s.data_dir)

    # ================================================================ public entry points
    def run(self, crawl: bool = True, index: bool = True) -> CrawlStats:
        with session_scope() as db:
            run = CrawlRun(trigger=self.trigger)
            db.add(run)
            db.flush()
            self.run_id = run.id
        if self.guard.crashed:
            log.warning("The previous crawl crashed while processing %s; it is skipped from now on", self.guard.crashed)
            self._error(self.guard.crashed, "Skipped from now on: the previous crawl crashed while processing it "
                                            "(probably out of memory). Remove it from crawl-skip.txt to try again.")
        status = "completed"
        try:
            if crawl:
                self._crawl_stage()
            if not self.stop_event.is_set():
                self._clean_stage()
            if index and not self.stop_event.is_set():
                self._index_stage()
            if crawl and not self.stop_event.is_set() and not self.stats.limit_reached:
                self._cleanup_removed()
            if self.stop_event.is_set():
                status = "stopped"
        except KeyboardInterrupt:
            status = "stopped"
            raise
        except Exception as e:
            status = "failed"
            self.stats.notes.append(f"Crawl failed: {e}")
            log.exception("Crawl failed")
            raise
        finally:
            self.fetcher.close()
            self.stats.current = ""
            self.stats.phase = status
            with session_scope() as db:
                run = db.get(CrawlRun, self.run_id)
                run.status = status
                run.finished_at = utcnow()
                run.stats = self.stats.as_dict()
            self.on_progress(self.stats)
        return self.stats

    # ================================================================ stage 1: crawl
    def _enqueue(self, item: QueueItem) -> None:
        url = item.url
        if url in self._seen:
            queued = self._queued.get(url)
            if queued and item.link_text and not queued.link_text:  # better context for a queued document
                queued.parent_url, queued.link_text = item.parent_url, item.link_text
            return
        kind = classify(url)
        if kind is None:
            return
        if kind == "drive":
            if not self.s.crawl_include_google_drive:
                return
        elif not domain_allowed(url, self.s.crawl_allowed_domains, self.s.crawl_blocked_domains):
            return
        year = upload_year(url)
        if year is not None and year < self.s.crawl_min_upload_year:
            return
        if kind == "image" and not self._ocr_ok:
            return  # picked up on a later crawl once Tesseract is installed
        if kind == "html" and item.depth > self.s.crawl_max_depth:
            return
        self._seen.add(url)
        self._queued[url] = item
        (self._pages if kind == "html" else self._docs).append(item)
        self.stats.discovered += 1

    def _seed(self) -> None:
        sites = {f"{urlparse(u).scheme}://{urlparse(u).netloc}" for u in self.s.crawl_start_urls}
        for u in self.s.crawl_start_urls:
            n = normalize_url(u)
            if n:
                self._enqueue(QueueItem(url=n, depth=0))
        for site in sorted(sites):
            if self.stop_event.is_set():
                return
            self.stats.current = f"discovering {site}"
            self.on_progress(self.stats)
            for seed in discovery.sitemap_seeds(self.fetcher, site):
                self._enqueue(QueueItem(url=seed.url, depth=1, remote_modified=seed.remote_modified))
            for seed in discovery.wp_api_seeds(self.fetcher, site, self.s.crawl_min_upload_year):
                self._enqueue(QueueItem(url=seed.url, depth=1, remote_modified=seed.remote_modified,
                                        title=discovery.clean_title(seed.title)))

    def _crawl_stage(self) -> None:
        self.stats.phase = "discovering"
        self._seed()
        self.stats.phase = "crawling"
        # Pages before documents: pages supply the link text / parent page that titles each document.
        while (self._pages or self._docs) and not self.stop_event.is_set():
            if self.stats.processed >= self.s.crawl_max_pages:
                self.stats.limit_reached = True
                self.stats.notes.append(f"Stopped at CRAWL_MAX_PAGES={self.s.crawl_max_pages}")
                break
            item = self._pages.popleft() if self._pages else self._docs.popleft()
            self._queued.pop(item.url, None)
            self.stats.queued = len(self._pages) + len(self._docs)
            self.stats.current = item.url
            self.on_progress(self.stats)
            if item.url in self.guard.skip:  # crashed a previous crawl
                self.stats.skipped += 1
                self.stats.processed += 1
                continue
            self.guard.start(item.url)
            try:
                self._process(item)
            except Exception as e:  # never let one URL kill the crawl
                log.exception("Failed processing %s", item.url)
                self._error(item.url, f"{type(e).__name__}: {e}")
            finally:
                self.guard.done()
            self.stats.processed += 1

    def _process(self, item: QueueItem) -> None:
        url = item.url
        kind = classify(url)
        fetch_url = google_drive_download_url(url) if kind == "drive" else url
        if kind != "drive" and not self.fetcher.allowed_by_robots(url):
            self.stats.skipped += 1
            self._note_once(f"robots.txt disallows crawling {host_of(url)}")
            return

        with session_scope() as db:
            existing = db.scalar(select(Source).where(Source.url == url))
            existing_info = None
            if existing:
                if existing.status == "blocked":  # an admin removed it: never fetch or index it again
                    self.stats.skipped += 1
                    return
                existing_info = (existing.id, existing.remote_modified, existing.etag, existing.last_modified_header)

        conditional = existing_info and kind != "html" and not self.reindex
        # Documents: skip the download entirely when WordPress says nothing changed.
        if conditional and item.remote_modified and existing_info[1] == item.remote_modified:
            self._touch(existing_info[0], item)
            self.stats.unchanged += 1
            return

        res = self.fetcher.fetch(fetch_url, etag=existing_info[2] if conditional else None,
                                 last_modified=existing_info[3] if conditional else None)
        if res.not_modified and existing_info:
            self._touch(existing_info[0], item)
            self.stats.unchanged += 1
            return
        if not res.ok:
            self._error(url, res.error or f"HTTP {res.status}")
            if existing_info and res.status in (404, 410):
                self._delete(existing_info[0])
            return

        # Redirected somewhere we don't crawl (e.g. a login page)?
        final = normalize_url(res.url) or url
        if kind != "drive" and not domain_allowed(final, self.s.crawl_allowed_domains, self.s.crawl_blocked_domains):
            self.stats.skipped += 1
            return
        if final != url:
            self._seen.add(final)

        real_kind = CONTENT_TYPE_KIND.get(res.content_type) or (kind if kind != "drive" else None)
        if kind == "drive" and real_kind != "pdf":
            self.stats.skipped += 1  # only public Drive PDFs
            return

        meta_date = item.remote_modified
        ocr_used = False
        if real_kind == "html":
            page = extract_html(res.text, final)
            if len(page.text) < self.s.crawl_playwright_min_chars:
                rendered = self.fetcher.render_with_browser(final)  # JavaScript-built content
                if rendered:
                    page2 = extract_html(rendered, final)
                    if len(page2.text) > len(page.text):
                        page = page2
                    known = {u for u, _ in page.links}
                    page.links += [l for l in page2.links if l[0] not in known]
            text = page.text
            footer = page.footer
            title = page.title or item.title or title_from_url(url)
            section = page.section
            meta_date = page.meta_published or page.meta_modified or meta_date
            self._page_context[final] = " › ".join(x for x in (section, title) if x)
            if item.depth < self.s.crawl_max_depth:
                for link, anchor in page.links:
                    self._enqueue(QueueItem(url=link, depth=item.depth + 1, parent_url=final, link_text=anchor))
        elif real_kind in ("pdf", "docx", "txt", "image"):
            doc = extract_document(res.content, real_kind)
            text, ocr_used, footer = doc.text, doc.ocr_used, ""
            title = choose_document_title(url, link_text=item.link_text, api_title=item.title, doc_title=doc.title)
            section = self._page_context.get(item.parent_url or "", "")
            for link in doc.links:  # links inside PDFs lead deeper too
                n = normalize_url(link)
                if n:
                    self._enqueue(QueueItem(url=n, depth=item.depth + 1, parent_url=url))
        else:
            self.stats.skipped += 1
            return

        self._store(url=url, kind=real_kind, title=title, section=section, item=item, text=text, footer=footer,
                    meta_date=meta_date, etag=res.etag, last_modified=res.last_modified, ocr_used=ocr_used)

    def _store(self, *, url: str, kind: str, title: str, section: str, item: QueueItem, text: str, footer: str,
               meta_date: str | None, etag: str | None, last_modified: str | None, ocr_used: bool) -> None:
        now = utcnow()
        h = text_hash(text) if text else ""
        with session_scope() as db:
            src = db.scalar(select(Source).where(Source.url == url))
            is_new = src is None
            if is_new:
                src = Source(url=url, source_type="website")
                db.add(src)
            changed = (is_new or self.reindex or src.content_hash != h or src.title != title[:500]
                       or (src.section or "") != section)
            src.content_type = kind
            src.remote_modified = item.remote_modified or src.remote_modified
            src.etag, src.last_modified_header = etag, last_modified
            src.parent_url = item.parent_url or src.parent_url
            src.link_text = (item.link_text or src.link_text or "")[:500] or None
            src.last_crawled_at = now
            src.last_seen_run_id = self.run_id
            src.status, src.error = "active", None
            src.footer_text = footer or None
            if not changed:
                self.stats.unchanged += 1
                return
            src.title = title[:500]
            src.section = section[:500] or None
            src.raw_text = text
            src.text = text  # HTML text is refined by the clean stage
            src.content_hash = h
            src.language = detect_script_language(text)
            src.published_at = best_date(meta=meta_date, url=url, text=text)
            src.academic_year = find_academic_year(title, url, text)
            src.ocr_used = ocr_used
            src.last_changed_at = now
            src.index_status = "pending"
            self.stats.new += is_new
            self.stats.updated += not is_new

    # ================================================================ stage 2: clean
    def _clean_stage(self) -> None:
        self.stats.phase = "cleaning"
        self.stats.current = "finding text repeated across pages"
        self.on_progress(self.stats)
        with session_scope() as db:
            pages = db.scalars(select(Source).where(
                Source.content_type == "html", Source.source_type == "website", Source.status == "active",
                Source.url != COMMON_SOURCE_URL)).all()
            raws = [p.raw_text or p.text for p in pages]  # `or text`: rows from before raw_text existed
            boilerplate = find_boilerplate(raws) if len(raws) >= 20 else set()
            self.stats.boilerplate_lines = len(boilerplate)
            for p, raw in zip(pages, raws):
                cleaned = strip_boilerplate(raw, boilerplate)
                if cleaned != p.text:
                    p.text = cleaned
                    p.index_status = "pending"

            footers = Counter(p.footer_text for p in pages if p.footer_text)
            footer = footers.most_common(1)[0][0] if footers else ""
            common_text = site_information(footer, common_information(raws, boilerplate))
            common = db.scalar(select(Source).where(Source.url == COMMON_SOURCE_URL))
            if common_text and common is None:
                common = Source(url=COMMON_SOURCE_URL, source_type="website", content_type="html")
                db.add(common)
            if common is not None:
                common.title = COMMON_SOURCE_TITLE
                common.last_seen_run_id = self.run_id
                if common.text != common_text or common.published_at is None:
                    common.raw_text = common.text = common_text
                    common.content_hash = text_hash(common_text) if common_text else ""
                    common.index_status = "pending"
                    common.last_changed_at = common.published_at = utcnow()

    # ================================================================ stage 3: index
    def _index_stage(self) -> None:
        self.stats.phase = "indexing"
        with session_scope() as db:
            if self.reindex:
                for src in db.scalars(select(Source).where(Source.status == "active")):
                    src.index_status = "pending"
            pending = db.scalars(select(Source.id).where(Source.index_status == "pending").order_by(Source.id)).all()
        self.stats.to_index = len(pending)
        self.on_progress(self.stats)

        for start in range(0, len(pending), INDEX_BATCH_SOURCES):
            if self.stop_event.is_set():
                return  # remaining sources stay "pending" and are indexed next run
            ids = pending[start:start + INDEX_BATCH_SOURCES]
            with session_scope() as db:
                batch = [db.get(Source, i) for i in ids]
                to_embed: list[Source] = []
                hashes_in_batch: dict[str, int] = {}
                for src in batch:
                    if src is None:
                        continue
                    self.stats.current = src.url
                    if len((src.text or "").strip()) < MIN_TEXT_CHARS:
                        self._mark(src, "empty")
                        self.stats.empty += 1
                        continue
                    th = text_hash(src.text)
                    src.text_hash = th
                    dup = hashes_in_batch.get(th) or db.scalar(
                        select(Source.id).where(Source.text_hash == th, Source.id != src.id,
                                                Source.index_status == "indexed").limit(1))
                    if dup:
                        self._mark(src, "duplicate", duplicate_of=dup)
                        self.stats.duplicates += 1
                        continue
                    hashes_in_batch[th] = src.id
                    to_embed.append(src)
                self.on_progress(self.stats)
                counts = indexer.index_sources(to_embed)
                for src in to_embed:
                    src.chunk_count = counts[src.id]
                    src.index_status = "indexed" if counts[src.id] else "empty"
                    src.duplicate_of = None
                    self.stats.chunks_written += counts[src.id]
                self.stats.indexed += len(batch)
                self.on_progress(self.stats)

    @staticmethod
    def _mark(src: Source, status: str, duplicate_of: int | None = None) -> None:
        indexer.remove_source(src)
        src.index_status, src.chunk_count, src.duplicate_of = status, 0, duplicate_of

    # ================================================================ bookkeeping
    def _touch(self, source_id: int, item: QueueItem) -> None:
        """Unchanged document: record that it still exists, and pick up better link context."""
        with session_scope() as db:
            src = db.get(Source, source_id)
            src.last_seen_run_id = self.run_id
            src.last_crawled_at = utcnow()
            if item.link_text and src.content_type != "html":
                title = choose_document_title(src.url, link_text=item.link_text, api_title=item.title,
                                              doc_title=src.title)
                section = self._page_context.get(item.parent_url or "", "") or src.section or ""
                if title != src.title or section != (src.section or ""):
                    src.title, src.section = title[:500], section[:500] or None
                    src.parent_url, src.link_text = item.parent_url, item.link_text[:500]
                    src.index_status = "pending"

    def _delete(self, source_id: int) -> None:
        with session_scope() as db:
            src = db.get(Source, source_id)
            if src:
                indexer.remove_source(src)
                db.delete(src)
                self.stats.deleted += 1

    def _error(self, url: str, message: str) -> None:
        self.stats.errors += 1
        with session_scope() as db:
            db.add(CrawlError(run_id=self.run_id, url=url[:2000], error=message[:2000]))

    def _note_once(self, note: str) -> None:
        if note not in self.stats.notes:
            self.stats.notes.append(note)

    def _cleanup_removed(self) -> None:
        """Website sources not seen in this full crawl: delete them if they are really gone (404/410)."""
        self.stats.phase = "checking removed pages"
        with session_scope() as db:
            stale = db.scalars(
                select(Source).where(Source.source_type == "website", ~Source.url.startswith("site://"),
                                     Source.status != "blocked",
                                     (Source.last_seen_run_id != self.run_id) | (Source.last_seen_run_id.is_(None)))
            ).all()
            stale_info = [(s.id, s.url) for s in stale]
        fetcher = Fetcher(self.s)
        try:
            for sid, url in stale_info:
                if self.stop_event.is_set():
                    return
                self.stats.current = f"checking: {url}"
                self.on_progress(self.stats)
                target = google_drive_download_url(url) if is_google_drive(url) else url
                res = fetcher.fetch(target)
                if res.status in (404, 410):
                    self._delete(sid)
        finally:
            fetcher.close()


def crawl_single_url(url: str, trigger: str = "admin") -> CrawlStats:
    """Fetch, clean and index one URL (used by the admin 'add URL' feature in Phase 3)."""
    n = normalize_url(url)
    if not n:
        raise ValueError("Invalid URL")
    crawler = Crawler(trigger=trigger)
    with session_scope() as db:
        run = CrawlRun(trigger=trigger)
        db.add(run)
        db.flush()
        crawler.run_id = run.id
    status = "completed"
    try:
        crawler._process(QueueItem(url=n, depth=crawler.s.crawl_max_depth))
        crawler._clean_stage()
        crawler._index_stage()
    except Exception:
        status = "failed"
        raise
    finally:
        crawler.fetcher.close()
        with session_scope() as db:
            run = db.get(CrawlRun, crawler.run_id)
            run.status, run.finished_at, run.stats = status, utcnow(), crawler.stats.as_dict()
    return crawler.stats
