"""SQLAlchemy models for the metadata database (SQLite).

Knowledge base (sources, crawl runs), admin panel (users, sessions, audit log,
verified answers), anonymous quality signals (unanswered questions, thumbs-down
questions, daily counters). Chat conversations are intentionally NOT stored.
"""
from __future__ import annotations

from datetime import date, datetime, timezone

from sqlalchemy import JSON, Boolean, Date, DateTime, Float, ForeignKey, Integer, String, Text
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


def as_utc(dt: datetime | None) -> datetime | None:
    """SQLite returns naive datetimes; everything is stored in UTC."""
    if dt is None:
        return None
    return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)


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
    status: Mapped[str] = mapped_column(String(20), default="active")  # active | error | blocked (by an admin)
    error: Mapped[str | None] = mapped_column(Text, nullable=True)
    chunk_count: Mapped[int] = mapped_column(Integer, default=0)
    ocr_used: Mapped[bool] = mapped_column(default=False)
    ocr_pending: Mapped[bool] = mapped_column(default=False)  # has scanned pages not OCR'd yet (deferred OCR)
    file_hash: Mapped[str | None] = mapped_column(String(64), nullable=True)  # downloaded bytes: skip re-reading
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


# ================================================================ admin panel

class AdminUser(Base):
    __tablename__ = "admin_users"

    id: Mapped[int] = mapped_column(primary_key=True)
    username: Mapped[str] = mapped_column(String(64), unique=True, index=True)
    password_hash: Mapped[str] = mapped_column(String(256))
    role: Mapped[str] = mapped_column(String(20), default="editor")  # super | editor
    totp_secret: Mapped[str | None] = mapped_column(String(64), nullable=True)  # set when 2FA is enabled
    active: Mapped[bool] = mapped_column(Boolean, default=True)
    failed_logins: Mapped[int] = mapped_column(Integer, default=0)
    locked_until: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    last_login_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)


class AdminSession(Base):
    """Server-side session. The cookie holds a random token; only its SHA-256 is stored."""

    __tablename__ = "admin_sessions"

    id: Mapped[int] = mapped_column(primary_key=True)
    token_hash: Mapped[str] = mapped_column(String(64), unique=True, index=True)
    user_id: Mapped[int] = mapped_column(ForeignKey("admin_users.id", ondelete="CASCADE"), index=True)
    csrf_token: Mapped[str] = mapped_column(String(64))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
    last_seen_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
    pending_2fa: Mapped[bool] = mapped_column(Boolean, default=False)  # password OK, TOTP code still needed


class AuditLog(Base):
    __tablename__ = "audit_log"

    id: Mapped[int] = mapped_column(primary_key=True)
    username: Mapped[str] = mapped_column(String(64))
    action: Mapped[str] = mapped_column(String(64), index=True)  # upload, delete_source, crawl_start, …
    target: Mapped[str] = mapped_column(String(1024), default="")
    details: Mapped[str] = mapped_column(Text, default="")
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow, index=True)


class VerifiedAnswer(Base):
    """Official Q&A written by the college. Checked before the website and used word for word."""

    __tablename__ = "verified_answers"

    id: Mapped[int] = mapped_column(primary_key=True)
    question: Mapped[str] = mapped_column(Text)
    alt_questions: Mapped[str] = mapped_column(Text, default="")  # other phrasings, one per line
    answer: Mapped[str] = mapped_column(Text)
    language: Mapped[str] = mapped_column(String(10), default="any")  # any | en | hi | gu | …
    source_url: Mapped[str | None] = mapped_column(String(2048), nullable=True)  # optional "more info" link
    active: Mapped[bool] = mapped_column(Boolean, default=True)
    times_used: Mapped[int] = mapped_column(Integer, default=0)
    updated_by: Mapped[str] = mapped_column(String(64), default="")
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)


class NegativeFeedback(Base):
    """Thumbs-down: question text and the source links used (no answer, IP or session). Auto-deleted."""

    __tablename__ = "negative_feedback"

    id: Mapped[int] = mapped_column(primary_key=True)
    question: Mapped[str] = mapped_column(Text)
    sources: Mapped[str] = mapped_column(Text, default="")  # newline-separated URLs
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow, index=True)


class UsageDay(Base):
    """Anonymous daily counters for the dashboard (no content)."""

    __tablename__ = "usage_days"

    day: Mapped[date] = mapped_column(Date, primary_key=True)
    chats: Mapped[int] = mapped_column(Integer, default=0)
    unanswered: Mapped[int] = mapped_column(Integer, default=0)
    verified_hits: Mapped[int] = mapped_column(Integer, default=0)
    thumbs_up: Mapped[int] = mapped_column(Integer, default=0)
    thumbs_down: Mapped[int] = mapped_column(Integer, default=0)
    llm_calls: Mapped[int] = mapped_column(Integer, default=0)
    input_tokens: Mapped[int] = mapped_column(Integer, default=0)
    output_tokens: Mapped[int] = mapped_column(Integer, default=0)
    cost_usd: Mapped[float] = mapped_column(Float, default=0.0)
