"""Seed URL discovery: XML sitemaps (Yoast), the HTML sitemap page and the WordPress REST API."""
from __future__ import annotations

import html
import json
import logging
import re
from dataclasses import dataclass
from urllib.parse import urlparse

from bs4 import BeautifulSoup

from app.crawler.fetcher import Fetcher
from app.crawler.urls import normalize_url

log = logging.getLogger(__name__)


@dataclass
class Seed:
    url: str
    remote_modified: str | None = None  # lastmod / WP `modified`
    title: str | None = None


def _sitemap_urls(fetcher: Fetcher, sitemap_url: str, seen: set[str], depth: int = 0) -> list[Seed]:
    if depth > 3 or sitemap_url in seen:
        return []
    seen.add(sitemap_url)
    res = fetcher.fetch(sitemap_url)
    if not res.ok or b"<" not in res.content[:200]:
        return []
    soup = BeautifulSoup(res.content, "xml")
    seeds: list[Seed] = []
    for sm in soup.find_all("sitemap"):
        loc = sm.find("loc")
        if loc and loc.text.strip():
            seeds += _sitemap_urls(fetcher, loc.text.strip(), seen, depth + 1)
    for u in soup.find_all("url"):
        loc = u.find("loc")
        lastmod = u.find("lastmod")
        url = normalize_url(loc.text) if loc else None
        if url:
            seeds.append(Seed(url=url, remote_modified=lastmod.text.strip() if lastmod else None))
    return seeds


def sitemap_seeds(fetcher: Fetcher, site: str) -> list[Seed]:
    root = f"{urlparse(site).scheme}://{urlparse(site).netloc}"
    seen: set[str] = set()
    seeds: list[Seed] = []
    for path in ("/sitemap_index.xml", "/sitemap.xml", "/wp-sitemap.xml"):
        seeds += _sitemap_urls(fetcher, root + path, seen)
    # Human-readable HTML sitemap.
    res = fetcher.fetch(root + "/sitemap/")
    if res.ok and "html" in res.content_type:
        soup = BeautifulSoup(res.content, "lxml")
        for a in soup.find_all("a", href=True):
            u = normalize_url(a["href"], res.url)
            if u:
                seeds.append(Seed(url=u))
    return seeds


def wp_api_seeds(fetcher: Fetcher, site: str, min_upload_year: int) -> list[Seed]:
    """Pages, posts and uploaded documents listed by the WordPress REST API."""
    root = f"{urlparse(site).scheme}://{urlparse(site).netloc}"
    seeds: list[Seed] = []

    def paged(endpoint: str, params: str) -> list[dict]:
        items: list[dict] = []
        page = 1
        while page <= 100:
            res = fetcher.fetch(f"{root}/wp-json/wp/v2/{endpoint}?per_page=100&page={page}&{params}")
            if not res.ok:
                break
            try:
                batch = json.loads(res.content)
            except ValueError:
                break
            if not isinstance(batch, list) or not batch:
                break
            items += batch
            if len(batch) < 100:
                break
            page += 1
        return items

    for endpoint in ("pages", "posts"):
        for item in paged(endpoint, "_fields=link,modified_gmt,title"):
            url = normalize_url(item.get("link", ""))
            if url:
                title = (item.get("title") or {}).get("rendered")
                seeds.append(Seed(url=url, remote_modified=item.get("modified_gmt"), title=title))

    # Documents in the media library (PDF / DOCX), uploaded since the cut-off year.
    after = f"after={min_upload_year}-01-01T00:00:00"
    for mime in ("application/pdf", "application/vnd.openxmlformats-officedocument.wordprocessingml.document"):
        for item in paged("media", f"mime_type={mime}&{after}&_fields=source_url,modified_gmt,title"):
            url = normalize_url(item.get("source_url", ""))
            if url:
                title = (item.get("title") or {}).get("rendered")
                seeds.append(Seed(url=url, remote_modified=item.get("modified_gmt"), title=title))
    log.info("WordPress API: %d seeds from %s", len(seeds), root)
    return seeds


def clean_title(title: str | None) -> str | None:
    if not title:
        return title
    return html.unescape(re.sub(r"<[^>]+>", "", title)).strip()
