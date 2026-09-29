"""URL normalisation, domain allowlist and link classification."""
from __future__ import annotations

import ipaddress
import re
import socket
from urllib.parse import parse_qsl, urldefrag, urlencode, urljoin, urlparse, urlunparse

DOC_EXTENSIONS = {".pdf": "pdf", ".docx": "docx", ".txt": "txt"}
IMAGE_EXTENSIONS = {".jpg": "image", ".jpeg": "image", ".png": "image"}
SKIP_EXTENSIONS = {
    ".gif", ".svg", ".webp", ".ico", ".bmp", ".tif", ".tiff",
    ".mp4", ".mp3", ".wav", ".avi", ".mov", ".webm", ".m4a",
    ".zip", ".rar", ".7z", ".gz", ".tar", ".exe", ".msi", ".apk",
    ".css", ".js", ".json", ".xml", ".rss", ".woff", ".woff2", ".ttf", ".eot",
    ".doc", ".xls", ".xlsx", ".ppt", ".pptx", ".csv",
}
# Paths that are never useful content (WordPress internals, feeds, login pages, …).
# Note: /author/ is NOT skipped — on sxca.edu.in the author pages are the faculty profiles.
SKIP_PATH_PATTERNS = re.compile(
    r"(/wp-admin|/wp-login|/wp-json|/xmlrpc\.php|/feed/?$|/comments/feed|/trackback|/tag/|"
    r"/cart|/checkout|/my-account|/login|/signin|/sign-in|/logout|/register|/erp|/cdn-cgi/|/wp-content/plugins/|"
    r"/wp-content/themes/|/wp-includes/)",
    re.IGNORECASE,
)
# Query parameters that only track / sort / share and would create duplicates.
DROP_QUERY_PARAMS = re.compile(r"^(utm_.*|fbclid|gclid|replytocom|share|amp|print|_ga|ref|elementor-preview)$", re.I)
UPLOAD_YEAR = re.compile(r"/wp-content/uploads/(\d{4})/(\d{2})/")
DRIVE_FILE_ID = re.compile(r"(?:/file/d/|[?&]id=)([A-Za-z0-9_-]{20,})")


def normalize_url(url: str, base: str | None = None) -> str | None:
    """Absolute, fragment-free, tracking-free URL; None for non-http(s) links."""
    if not url:
        return None
    url = url.strip()
    if url.lower().startswith(("mailto:", "tel:", "javascript:", "data:", "whatsapp:", "#")):
        return None
    if base:
        url = urljoin(base, url)
    url, _ = urldefrag(url)
    p = urlparse(url)
    if p.scheme not in ("http", "https") or not p.netloc:
        return None
    host = p.hostname.lower() if p.hostname else ""
    if p.port and p.port not in (80, 443):
        host = f"{host}:{p.port}"
    query = urlencode([(k, v) for k, v in parse_qsl(p.query, keep_blank_values=True) if not DROP_QUERY_PARAMS.match(k)])
    path = re.sub(r"/{2,}", "/", p.path or "/")
    return urlunparse((p.scheme, host, path, "", query, ""))


def host_of(url: str) -> str:
    return (urlparse(url).hostname or "").lower()


def domain_allowed(url: str, allowed: list[str], blocked: list[str]) -> bool:
    host = host_of(url)
    if any(host == b or host.endswith("." + b) for b in blocked):
        return False
    # "sxca.edu.in" allows only the bare domain and www.; subdomains must be listed explicitly.
    return any(host == a or host == "www." + a for a in allowed)


def extension(url: str) -> str:
    path = urlparse(url).path.lower()
    m = re.search(r"(\.[a-z0-9]{2,5})$", path)
    return m.group(1) if m else ""


def classify(url: str) -> str | None:
    """'html' | 'pdf' | 'docx' | 'txt' | 'image' | 'drive' | None (skip)."""
    if is_google_drive(url):
        return "drive"
    ext = extension(url)
    if ext in DOC_EXTENSIONS:
        return DOC_EXTENSIONS[ext]
    if ext in IMAGE_EXTENSIONS:
        return "image"
    if ext in SKIP_EXTENSIONS:
        return None
    if SKIP_PATH_PATTERNS.search(urlparse(url).path):
        return None
    return "html"


def upload_year(url: str) -> int | None:
    m = UPLOAD_YEAR.search(url)
    return int(m.group(1)) if m else None


def is_google_drive(url: str) -> bool:
    return host_of(url) in ("drive.google.com", "docs.google.com") and bool(DRIVE_FILE_ID.search(url))


def google_drive_download_url(url: str) -> str | None:
    m = DRIVE_FILE_ID.search(url)
    if not m:
        return None
    return f"https://drive.google.com/uc?export=download&id={m.group(1)}"


def is_public_ip_host(url: str) -> bool:
    """SSRF guard: True only if every address the host resolves to is public."""
    host = host_of(url)
    if not host:
        return False
    try:
        infos = socket.getaddrinfo(host, None)
    except socket.gaierror:
        return False
    for info in infos:
        ip = ipaddress.ip_address(info[4][0])
        if ip.is_private or ip.is_loopback or ip.is_link_local or ip.is_reserved or ip.is_multicast:
            return False
    return True
