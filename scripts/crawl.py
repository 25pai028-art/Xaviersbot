"""Command-line deep crawler with a live progress display.

    python -m scripts.crawl                 # full incremental crawl: crawl → clean → index
    python -m scripts.crawl --max-pages 50  # quick test run
    python -m scripts.crawl --no-index      # crawl + clean only (fast; index later)
    python -m scripts.crawl --index-only    # index whatever is pending (resume after an interruption)
    python -m scripts.crawl --reindex       # re-extract, re-chunk and re-embed everything
    python -m scripts.crawl --url URL       # crawl/index a single URL
    python -m scripts.crawl --stats         # show what is indexed
"""
from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

from rich.console import Console  # noqa: E402
from rich.live import Live  # noqa: E402
from rich.table import Table  # noqa: E402
from sqlalchemy import func, select  # noqa: E402

from app.config import get_settings  # noqa: E402
from app.crawler.documents import ocr_available  # noqa: E402
from app.crawler.pipeline import CrawlStats, Crawler, crawl_single_url  # noqa: E402
from app.db.models import CrawlError, Source  # noqa: E402
from app.db.session import session_scope  # noqa: E402
from app.rag import vectorstore  # noqa: E402

console = Console()


def _bar(done: int, total: int) -> str:
    pct = min(100, int(100 * done / max(1, total)))
    return "█" * (pct // 4) + "░" * (25 - pct // 4) + f" {pct}%"


def render(stats: CrawlStats, max_pages: int) -> Table:
    t = Table(title=f"SXCA deep crawl: {stats.phase}", show_header=False, expand=False)
    crawl_total = min(max_pages, stats.processed + stats.queued)
    t.add_row("1. Crawl", f"{_bar(stats.processed, crawl_total)}  {stats.processed} fetched, {stats.queued} queued, "
                          f"{stats.discovered} discovered")
    t.add_row("   new / updated / unchanged", f"{stats.new} / {stats.updated} / {stats.unchanged}")
    t.add_row("   skipped / errors / deleted", f"{stats.skipped} / {stats.errors} / {stats.deleted}")
    t.add_row("2. Clean", f"{stats.boilerplate_lines} repeated site-wide lines removed")
    t.add_row("3. Index", f"{_bar(stats.indexed, stats.to_index)}  {stats.indexed}/{stats.to_index} sources, "
                          f"{stats.chunks_written} chunks, {stats.empty} empty, {stats.duplicates} duplicates")
    t.add_row("Current", (stats.current or "")[-100:])
    return t


def show_stats() -> None:
    with session_scope() as db:
        rows = db.execute(
            select(Source.content_type, Source.index_status, func.count(), func.sum(Source.chunk_count))
            .group_by(Source.content_type, Source.index_status)
            .order_by(Source.content_type, Source.index_status)
        ).all()
        errors = db.scalars(select(CrawlError).order_by(CrawlError.id.desc()).limit(15)).all()
        t = Table(title="Knowledge base")
        for col in ("type", "index status", "sources", "chunks"):
            t.add_column(col)
        for ctype, status, n, chunks in rows:
            t.add_row(ctype, status, str(n), str(chunks or 0))
        console.print(t)
        console.print(f"Vector store chunks: {vectorstore.count()}")
        if errors:
            console.print("\n[bold]Latest crawl errors[/bold]")
            for e in errors:
                console.print(f"  {e.error[:60]:60}  {e.url}")


def main() -> None:
    ap = argparse.ArgumentParser(description="Deep-crawl the SXCA website into the knowledge base")
    ap.add_argument("--url", help="crawl a single URL only")
    ap.add_argument("--max-pages", type=int, help="override CRAWL_MAX_PAGES for this run")
    ap.add_argument("--reindex", action="store_true",
                    help="re-extract, re-chunk and re-embed everything (after changing extraction/chunk settings)")
    ap.add_argument("--no-index", action="store_true", help="crawl and clean only; index later with --index-only")
    ap.add_argument("--index-only", action="store_true", help="skip crawling; index pending sources")
    ap.add_argument("--stats", action="store_true", help="show index statistics and exit")
    ap.add_argument("-v", "--verbose", action="store_true")
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO if args.verbose else logging.WARNING,
                        format="%(levelname)s %(name)s: %(message)s")
    for noisy in ("httpx", "httpcore", "huggingface_hub", "sentence_transformers"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
    settings = get_settings()
    if args.stats:
        show_stats()
        return
    if not ocr_available()[0]:
        console.print("[yellow]Tesseract OCR not found: scanned PDFs and image notices will be skipped.[/yellow]")

    if args.url:
        stats = crawl_single_url(args.url, trigger="cli")
        console.print(stats.as_dict())
        return

    if args.max_pages:
        settings.crawl_max_pages = args.max_pages
    if not args.no_index:
        console.print("Loading the embedding model…")
        from app.rag.embeddings import _model

        _model()

    live = Live(render(CrawlStats(), settings.crawl_max_pages), console=console, refresh_per_second=2)
    crawler = Crawler(settings, trigger="cli", reindex=args.reindex,
                      on_progress=lambda s: live.update(render(s, settings.crawl_max_pages)))
    with live:
        try:
            stats = crawler.run(crawl=not args.index_only, index=not args.no_index)
        except KeyboardInterrupt:
            crawler.stop_event.set()
            stats = crawler.stats
            console.print("[yellow]Stopped. Anything not yet indexed stays pending: "
                          "run `python -m scripts.crawl --index-only` to finish.[/yellow]")
    for note in stats.notes:
        console.print(f"[yellow]Note:[/yellow] {note}")
    console.print("[green]Done.[/green] Run `python -m scripts.crawl --stats` for a summary.")


if __name__ == "__main__":
    main()
