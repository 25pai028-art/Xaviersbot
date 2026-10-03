"""Polite HTTP fetching: robots.txt, per-host delay, size limits, conditional GETs,
and an optional Playwright fallback for JavaScript-rendered pages."""
from __future__ import annotations

import logging
import threading
import time
import urllib.robotparser
from dataclasses import dataclass

import httpx

from app.config import Settings
from app.crawler.urls import host_of

log = logging.getLogger(__name__)


@dataclass
class FetchResult:
    url: str  # final URL after redirects
    status: int
    content: bytes = b""
    content_type: str = ""
    etag: str | None = None
    last_modified: str | None = None
    not_modified: bool = False
    error: str | None = None

    @property
    def ok(self) -> bool:
        return self.error is None and 200 <= self.status < 300

    @property
    def text(self) -> str:
        return self.content.decode("utf-8", errors="replace")


class TooLarge(Exception):
    pass


class Fetcher:
    def __init__(self, settings: Settings):
        self.settings = settings
        self.max_bytes = int(settings.crawl_max_file_mb * 1024 * 1024)
        self.client = httpx.Client(
            headers={"User-Agent": settings.crawl_user_agent, "Accept-Language": "en,hi;q=0.8,gu;q=0.7"},
            timeout=httpx.Timeout(settings.crawl_timeout_seconds, connect=10.0),
            follow_redirects=True,
            max_redirects=5,
        )
        self._robots: dict[str, urllib.robotparser.RobotFileParser | None] = {}
        self._last_hit: dict[str, float] = {}
        self._turn_lock = threading.Lock()  # fetch() is called from several download threads
        self._browser = None
        self._playwright = None
        self._browser_broken = False

    # ------------------------------------------------------------ politeness
    def allowed_by_robots(self, url: str) -> bool:
        host = host_of(url)
        if host in self.settings.crawl_ignore_robots_domains:  # crawling authorised by the college
            return True
        if host not in self._robots:
            rp = urllib.robotparser.RobotFileParser()
            scheme = "https" if url.startswith("https") else "http"
            try:
                r = self.client.get(f"{scheme}://{host}/robots.txt", timeout=10.0)
                if r.status_code == 200:
                    rp.parse(r.text.splitlines())
                    self._robots[host] = rp
                else:
                    self._robots[host] = None  # no robots.txt → everything allowed
            except httpx.HTTPError:
                self._robots[host] = None
        rp = self._robots[host]
        return True if rp is None else rp.can_fetch(self.settings.crawl_user_agent, url)

    def _wait_turn(self, host: str) -> None:
        """Requests to one host start at least `crawl_delay_seconds` apart, whichever thread sends them."""
        with self._turn_lock:
            now = time.monotonic()
            start = max(now, self._last_hit.get(host, -1e9) + self.settings.crawl_delay_seconds)
            self._last_hit[host] = start  # reserve the slot, then wait outside the lock
        if start > now:
            time.sleep(start - now)

    # ------------------------------------------------------------ fetching
    def fetch(self, url: str, *, etag: str | None = None, last_modified: str | None = None) -> FetchResult:
        headers = {}
        if etag:
            headers["If-None-Match"] = etag
        if last_modified:
            headers["If-Modified-Since"] = last_modified
        if host_of(url) in self.settings.crawl_browser_ua_domains:
            headers["User-Agent"] = self.settings.crawl_browser_user_agent
        self._wait_turn(host_of(url))
        for attempt in range(2):
            try:
                with self.client.stream("GET", url, headers=headers) as r:
                    if r.status_code == 304:
                        return FetchResult(url=str(r.url), status=304, not_modified=True, etag=etag, last_modified=last_modified)
                    declared = int(r.headers.get("content-length") or 0)
                    if declared > self.max_bytes:
                        raise TooLarge(f"{declared / 1e6:.1f} MB exceeds limit")
                    buf = bytearray()
                    for chunk in r.iter_bytes():
                        buf.extend(chunk)
                        if len(buf) > self.max_bytes:
                            raise TooLarge(f"exceeds {self.settings.crawl_max_file_mb} MB limit")
                    return FetchResult(
                        url=str(r.url),
                        status=r.status_code,
                        content=bytes(buf),
                        content_type=r.headers.get("content-type", "").split(";")[0].strip().lower(),
                        etag=r.headers.get("etag"),
                        last_modified=r.headers.get("last-modified"),
                        error=None if r.status_code < 400 else f"HTTP {r.status_code}",
                    )
            except TooLarge as e:
                return FetchResult(url=url, status=0, error=f"Skipped: {e}")
            except (httpx.TimeoutException, httpx.TransportError) as e:
                if attempt == 0:
                    time.sleep(3)
                    continue
                return FetchResult(url=url, status=0, error=f"{type(e).__name__}: {e}")
        return FetchResult(url=url, status=0, error="unreachable")

    def render_with_browser(self, url: str) -> str | None:
        """HTML after JavaScript runs (Playwright/Chromium). None if Playwright isn't available."""
        if not self.settings.crawl_use_playwright or self._browser_broken:
            return None
        try:
            if self._browser is None:
                from playwright.sync_api import sync_playwright

                self._playwright = sync_playwright().start()
                self._browser = self._playwright.chromium.launch(headless=True)
            ua = (self.settings.crawl_browser_user_agent if host_of(url) in self.settings.crawl_browser_ua_domains
                  else self.settings.crawl_user_agent)
            page = self._browser.new_page(user_agent=ua)
            try:
                self._wait_turn(host_of(url))
                page.goto(url, wait_until="networkidle", timeout=int(self.settings.crawl_timeout_seconds * 1000))
                return page.content()
            finally:
                page.close()
        except Exception as e:  # Playwright not installed, browser missing, navigation error…
            log.info("Playwright render failed for %s: %s", url, e)
            if self._browser is None:  # couldn't even start Chromium → stop trying
                self._browser_broken = True
            return None

    def close(self) -> None:
        self.client.close()
        if self._browser is not None:
            self._browser.close()
            self._playwright.stop()
