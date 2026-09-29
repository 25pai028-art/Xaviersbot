"""SQLAlchemy models for the metadata database (SQLite).

Only knowledge-base / crawl tables exist in Phase 1. Later phases add admins,
verified answers, feedback, bookings, audit log, etc.
Chat conversations are intentionally NOT stored (privacy decision).
"""
from __future__ import annotations

from datetime import datetime, timezone

from sqlalchemy import JSON, DateTime, ForeignKey, Integer, String, Text
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


class Base(DeclarativeBase):
    pass


class Source(Base):
    """One document in the knowledge base: a web page, a PDF/DOCX/image, or an admin upload."""

    __tablename__ = "sources"

    id: Mapped[int] = mapped_column(primary_key=True)
    url: Mapped[str] = mapped_column(String(2048), unique=True, index=True)
    source_type: Mapped[str] = mapped_column(String(20), default="website")  # website | upload | verified
    content_type: Mapped[str] = mapped_column(String(20), default="html")  # html | pdf | docx | image | txt
    title: Mapped[str] = mapped_column(String(512), default="")
    # Where the source sits on the site, e.g. "Admissions › UG Admissions" or the page that links to a PDF.
    section: Mapped[str | None] = mapped_column(String(512), nullable=True)
    parent_url: Mapped[str | None] = mapped_column(String(2048), nullable=True)
    link_text: Mapped[str | None] = mapped_column(String(512), nullable=True)
    raw_text: Mapped[str] = mapped_column(Text, default="")  # as extracted, before site-wide boilerplate removal
    footer_text: Mapped[str | None] = mapped_column(Text, nullable=True)  # site footer facts (indexed once)
    text: Mapped[str] = mapped_column(Text, default="")  # final text that is chunked and indexed
    language: Mapped[str] = mapped_column(String(10), default="en")
    published_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    academic_year: Mapped[str | None] = mapped_column(String(20), nullable=True)
    content_hash: Mapped[str] = mapped_column(String(64), default="")  # of raw_text: change detection
    text_hash: Mapped[str | None] = mapped_column(String(64), nullable=True)  # of final text: duplicate detection
    # pending → indexed | empty | duplicate. "pending" survives interruptions and is indexed on the next run.
    index_status: Mapped[str] = mapped_column(String(20), default="pending", index=True)
    duplicate_of: Mapped[int | None] = mapped_column(Integer, nullable=True)
    etag: Mapped[str | None] = mapped_column(String(256), nullable=True)
    last_modified_header: Mapped[str | None] = mapped_column(String(128), nullable=True)
    remote_modified: Mapped[str | None] = mapped_column(String(64), nullable=True)  # WP/sitemap lastmod
    status: Mapped[str] = mapped_column(String(20), default="active")  # active | error
    error: Mapped[str | None] = mapped_column(Text, nullable=True)
    chunk_count: Mapped[int] = mapped_column(Integer, default=0)
    ocr_used: Mapped[bool] = mapped_column(default=False)
    first_seen_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
    last_crawled_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    last_changed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    last_seen_run_id: Mapped[int | None] = mapped_column(Integer, nullable=True)


class UnansweredQuestion(Base):
    """Question text only (no IP, session or answer) — shown to the admin to find missing content.
    Auto-deleted after the retention period (Phase 7)."""

    __tablename__ = "unanswered_questions"

    id: Mapped[int] = mapped_column(primary_key=True)
    question: Mapped[str] = mapped_column(Text)
    reason: Mapped[str] = mapped_column(String(300), default="")
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow, index=True)


class CrawlRun(Base):
    __tablename__ = "crawl_runs"

    id: Mapped[int] = mapped_column(primary_key=True)
    trigger: Mapped[str] = mapped_column(String(20), default="cli")  # cli | schedule | admin
    status: Mapped[str] = mapped_column(String(20), default="running")  # running | completed | failed | stopped
    started_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
    finished_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    stats: Mapped[dict] = mapped_column(JSON, default=dict)


class CrawlError(Base):
    __tablename__ = "crawl_errors"

    id: Mapped[int] = mapped_column(primary_key=True)
    run_id: Mapped[int] = mapped_column(ForeignKey("crawl_runs.id", ondelete="CASCADE"), index=True)
    url: Mapped[str] = mapped_column(String(2048))
    error: Mapped[str] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
