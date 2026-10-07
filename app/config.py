"""Central configuration. Every setting comes from environment variables / `.env`.

See `.env.example` for documentation of each setting.
"""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Annotated

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, NoDecode, SettingsConfigDict

# Comma-separated list in .env (e.g. `A,B,C`) instead of JSON.
CsvList = Annotated[list[str], NoDecode]

BASE_DIR = Path(__file__).resolve().parent.parent


def _csv(value: str | list[str]) -> list[str]:
    if isinstance(value, list):
        return value
    return [v.strip() for v in value.split(",") if v.strip()]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=BASE_DIR / ".env", env_file_encoding="utf-8", extra="ignore")

    # --- General ---
    app_name: str = "Xavier's Assistant"
    environment: str = "development"
    data_dir: Path = BASE_DIR / "data"
    log_level: str = "INFO"

    # --- LLM ---
    llm_provider: str = "ollama"  # ollama | gemini | anthropic | openai
    llm_temperature: float = 0.1
    llm_max_output_tokens: int = 400
    llm_timeout_seconds: float = 180.0

    ollama_base_url: str = "http://localhost:11434"
    ollama_model: str = "qwen3.5:4b"
    ollama_num_ctx: int = 8192
    ollama_keep_alive: str = "30m"

    gemini_api_key: str = ""
    gemini_model: str = "gemini-2.5-flash"
    gemini_thinking_budget: int = 0

    anthropic_api_key: str = ""
    anthropic_model: str = "claude-opus-5-5"
    anthropic_effort: str = "low"

    openai_api_key: str = ""
    openai_model: str = "gpt-4.1-mini"
    openai_base_url: str = ""

    # --- Embeddings / retrieval ---
    embedding_model: str = "BAAI/bge-m3"
    embedding_device: str = "cpu"
    embedding_batch_size: int = 8
    embedding_max_seq_length: int = 1024
    chunk_target_tokens: int = 350
    chunk_overlap_tokens: int = 50
    retrieval_top_k: int = 3
    max_context_tokens: int = 1100
    similarity_threshold: float = 0.5
    chat_history_turns: int = 3
    rag_query_rewrite: bool = True
    fact_check_enabled: bool = True
    # After the figures check, a second AI pass removes sentences the sources don't state. Adds one LLM call
    # (~15-30 s on a laptop CPU, ~1-2 s with a cloud model).
    grounding_check: bool = True
    # Legitimate channel offered when refusing cheating/hacking requests (from sxca.edu.in/contact-us/).
    guard_exam_office_contact: str = "Examination Office (coe@sxca.edu.in, 079-26308055)"

    # --- Crawler ---
    crawl_start_urls: CsvList = Field(
        default_factory=lambda: ["https://sxca.edu.in/", "https://admissions.sxca.edu.in/SXCA/"])
    # The college authorised crawling its admissions portal (fee structures, refund rules, brochures),
    # although that site's robots.txt disallows all crawlers. Only these hosts skip robots.txt.
    crawl_ignore_robots_domains: CsvList = Field(default_factory=lambda: ["admissions.sxca.edu.in"])
    # The admissions portal answers "Invalid Browser" to unknown user agents; these hosts get a browser
    # user agent that still names the crawler at the end.
    crawl_browser_ua_domains: CsvList = Field(default_factory=lambda: ["admissions.sxca.edu.in"])
    crawl_browser_user_agent: str = (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/140.0 "
        "Safari/537.36 SXCA-Chatbot-Crawler/1.0 (+https://sxca.edu.in/)")
    crawl_allowed_domains: CsvList = Field(
        default_factory=lambda: ["sxca.edu.in", "admissions.sxca.edu.in", "library.sxca.edu.in"]
    )
    crawl_blocked_domains: CsvList = Field(
        default_factory=lambda: ["lms.sxca.edu.in", "portal.sxca.edu.in", "erp.sxca.edu.in"]
    )
    crawl_include_google_drive: bool = True
    crawl_max_pages: int = 5000
    crawl_max_depth: int = 6
    crawl_max_file_mb: float = 25.0
    # Downloads run in parallel (at most CRAWL_CONCURRENCY at once), and requests to the same host start at
    # least CRAWL_DELAY_SECONDS apart, so the college server sees at most 1/delay requests per second.
    crawl_concurrency: int = 4
    crawl_delay_seconds: float = 0.25
    crawl_timeout_seconds: float = 30.0
    # Uploads (wp-content/uploads/YYYY/) older than this are not crawled: old notices are hidden from answers
    # anyway. Lower it if older documents that are still valid (syllabi, rules) are missing.
    crawl_min_upload_year: int = 2025
    crawl_user_agent: str = "SXCA-Chatbot-Crawler/1.0 (+https://sxca.edu.in/; official college assistant)"
    crawl_use_playwright: bool = True
    crawl_playwright_min_chars: int = 200
    # Where the crash guard keeps crawl-current.txt / crawl-skip.txt (default: DATA_DIR). On Colab this is a
    # Google Drive folder, so it survives a runtime that crashed.
    crawl_state_dir: str = ""
    crawl_ocr_max_pages: int = 15  # scanned pages OCR'd per document (the important part is usually first)
    # Read scanned PDFs and image notices (slow OCR) in a separate pass after everything else is indexed,
    # so the chatbot is usable sooner. False: OCR them during the crawl.
    crawl_defer_ocr: bool = True
    ocr_languages: str = "eng+hin+guj"
    tesseract_cmd: str = ""

    # --- Verified (official) answers ---
    verified_direct_threshold: float = 0.86  # cosine: answer word for word, no LLM
    verified_context_threshold: float = 0.72  # cosine: give it to the LLM as the top source

    # --- Admin panel ---
    admin_session_idle_minutes: int = 30
    admin_session_max_hours: int = 8
    admin_max_failed_logins: int = 5
    admin_lockout_minutes: int = 15
    upload_max_mb: float = 20.0
    retention_days: int = 30  # unanswered / thumbs-down question texts are deleted after this

    # --- Background jobs ---
    scheduler_enabled: bool = True
    crawl_schedule_day: str = "sun"  # mon..sun, or "*" for daily
    crawl_schedule_hour: int = 2  # local time on the server

    # --- Paid LLM cost estimate (USD per 1M tokens; 0 = look up known models) ---
    llm_price_input_per_mtok: float = 0.0
    llm_price_output_per_mtok: float = 0.0

    # --- API ---
    cors_origins: CsvList = Field(default_factory=lambda: ["https://sxca.edu.in", "http://localhost:8000"])
    max_message_chars: int = 1000
    # Questions one visitor (IP address) may send; 0 = no limit. Kept in memory only.
    chat_rate_per_minute: int = 10
    chat_rate_per_day: int = 100

    # --- Languages ---
    translation_provider: str = "auto"  # auto (LLM, IndicTrans2 as backup) | llm | indictrans2 | off

    # --- Chat widget (colours are CSS variables, see README) ---
    widget_welcome: str = ""  # empty = the standard greeting
    # Shown in the language selector, in this order.
    widget_languages: CsvList = Field(
        default_factory=lambda: ["en", "hi", "gu", "ml", "ta", "te", "kn", "mr", "bn", "pa", "or", "ur"])
    widget_logo_url: str = ""  # empty = /static/widget/crest.png on this server
    college_office_url: str = "https://sxca.edu.in/contact-us/"
    college_office_email: str = "info@sxca.edu.in"
    college_office_phone: str = "079-29708056/7"

    _split_lists = field_validator(
        "crawl_start_urls", "crawl_allowed_domains", "crawl_blocked_domains", "cors_origins", "widget_languages",
        "crawl_ignore_robots_domains", "crawl_browser_ua_domains",
        mode="before",
    )(_csv)

    @property
    def sqlite_path(self) -> Path:
        return self.data_dir / "sxca.db"

    @property
    def chroma_dir(self) -> Path:
        return self.data_dir / "chroma"

    @property
    def cache_dir(self) -> Path:
        return self.data_dir / "cache"

    @property
    def uploads_dir(self) -> Path:
        # Outside any web-served folder; files are never executed or served back raw.
        return self.data_dir / "uploads"

    @property
    def is_production(self) -> bool:
        return self.environment.lower() == "production"


@lru_cache
def get_settings() -> Settings:
    s = Settings()
    s.data_dir.mkdir(parents=True, exist_ok=True)
    s.cache_dir.mkdir(parents=True, exist_ok=True)
    return s
