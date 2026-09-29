"""Clean main-content extraction from HTML pages.

- Removes navigation, headers/footers, sliders, cookie banners, social links,
  hidden text and scripts (hidden-text removal also defends against instructions
  hidden in pages). Accordion / tab panels are kept: they are hidden only until clicked.
- Converts tables into readable "Header: value" lines.
- Turns contact details hidden in attributes (`data-emailid`, `mailto:`, `tel:`)
  into visible "Email: …" / "Phone: …" text.
- Keeps headings as markdown `#` lines so the chunker can keep them with their content.
- Discovers links from anchors, iframes/embeds (PDF viewers), `data-*` attributes,
  `onclick` handlers and plain-text URLs.
"""
from __future__ import annotations

import copy
import html
import re
from dataclasses import dataclass, field
from urllib.parse import parse_qs, unquote, urlparse

from bs4 import BeautifulSoup, Comment, NavigableString, Tag

from app.crawler.urls import normalize_url

REMOVE_TAGS = ["script", "style", "noscript", "template", "svg", "canvas", "form", "button",
               "select", "input", "textarea", "header", "footer", "nav", "aside", "iframe", "object", "embed",
               "video", "audio"]

# Class/id fragments of boilerplate blocks (WordPress / Elementor / this theme).
BOILERPLATE_HINTS = re.compile(
    r"(^|[\s_-])(menu|navbar|nav-|breadcrumbs?|cookie|consent|gdpr|social|share|sharing|slider|swiper|carousel|"
    r"owl-|slick|marquee|ticker|popup|modal|offcanvas|sidebar|widget_recent|related|comments?|comment-respond|"
    r"search-form|skip-link|screen-reader|sr-only|visually-hidden|back-to-top|scroll-top|whatsapp|copyright|"
    r"header-section|footer-section|elementor-location-header|elementor-location-footer|site-header|site-footer)",
    re.IGNORECASE,
)
# Interactive panels that are hidden until clicked — real content, never boilerplate.
TOGGLE_PANEL = re.compile(r"(tab-content|tab-pane|accordion|toggle-content|collapse|e-n-tab)", re.I)
# Elementor renders some blocks twice (desktop + mobile copies); keep only the desktop copy.
MOBILE_ONLY = re.compile(r"elementor-hidden-desktop|elementor-tab-mobile-title")
# Accordion / tab titles (FAQ questions, faculty department names) become headings, so each
# question-and-answer is chunked on its own instead of several being mixed together.
PANEL_TITLE = re.compile(r"(^|\s)(elementor-tab-title|e-n-accordion-item-title|e-n-tab-title|accordion-button|"
                         r"accordion-header|panel-title|elementor-toggle-title)(\s|$)")
HIDDEN_STYLE = re.compile(r"display\s*:\s*none|visibility\s*:\s*hidden|font-size\s*:\s*0|opacity\s*:\s*0(?![.\d])", re.I)
NOISE_LINES = re.compile(
    r"^(view all|read more|know more|click here|learn more|more|apply now|download|home|previous|next|"
    r"share|share this|follow us|back to top|menu|close|search|load more|view email|view profile|"
    r"search results|›|»|«|‹|\||-)\s*[:.!»›]*$",
    re.IGNORECASE,
)
BLOCK_TAGS = {"p", "div", "section", "article", "main", "li", "ul", "ol", "br", "tr", "table", "dl", "dt", "dd",
              "blockquote", "pre", "figure", "figcaption", "address", "h1", "h2", "h3", "h4", "h5", "h6"}
# Inline tags whose content must not glue onto the next element ("Dr. A ShahDr. B Rao").
SPACED_INLINE = {"a", "span", "label", "small", "time", "abbr", "cite"}

EMAIL = re.compile(r"[\w.+-]+@[\w-]+(?:\.[\w-]+)+")
TEXT_URL = re.compile(r"https?://[^\s<>\"')\]]+", re.I)
LINK_ATTRS = ("data-href", "data-url", "data-link", "data-file", "data-pdf", "data-pdf-url", "data-src-pdf")
ONCLICK_URL = re.compile(r"""(?:window\.open|location(?:\.href)?\s*=|location\.assign)\s*\(?\s*['"]([^'"]+)['"]""")


@dataclass
class ExtractedPage:
    title: str
    text: str
    links: list[tuple[str, str]] = field(default_factory=list)  # (url, anchor/link text)
    section: str = ""  # breadcrumb, e.g. "Admissions › UG Admissions"
    footer: str = ""  # facts from the site footer (indexed once site-wide, not per page)
    meta_modified: str | None = None
    meta_published: str | None = None
    lang: str | None = None


# ---------------------------------------------------------------- link discovery

def _viewer_target(url: str) -> str | None:
    """PDF viewers embed the real file as ?file=… / ?url=… (pdf.js, Google viewer, flipbooks)."""
    qs = parse_qs(urlparse(url).query)
    for key in ("file", "url", "pdf", "src"):
        for v in qs.get(key, []):
            v = unquote(v)
            if v.lower().split("?")[0].endswith((".pdf", ".docx")) or "drive.google.com" in v:
                return v
    return None


def discover_links(soup: BeautifulSoup, base: str) -> list[tuple[str, str]]:
    found: dict[str, str] = {}

    def add(raw: str | None, text: str = "") -> None:
        if not raw:
            return
        for candidate in (raw, _viewer_target(raw)):
            u = normalize_url(candidate, base) if candidate else None
            if u and (u not in found or (text and not found[u])):
                found[u] = text.strip()[:300]

    for a in soup.find_all("a", href=True):
        add(a["href"], a.get_text(" ", strip=True) or a.get("title", "") or a.get("aria-label", ""))
    for t in soup.find_all(["iframe", "embed", "source"]):
        add(t.get("src"), t.get("title", ""))
    for t in soup.find_all("object"):
        add(t.get("data"), t.get("title", ""))
    for attr in LINK_ATTRS:
        for t in soup.find_all(attrs={attr: True}):
            add(t[attr], t.get_text(" ", strip=True))
    for t in soup.find_all(attrs={"onclick": True}):
        m = ONCLICK_URL.search(t["onclick"])
        if m:
            add(m.group(1), t.get_text(" ", strip=True))
    return list(found.items())


# ---------------------------------------------------------------- cleaning helpers

def _is_boilerplate(tag: Tag) -> bool:
    if not isinstance(tag, Tag) or tag.attrs is None:
        return False
    ident = " ".join(tag.get("class", []) or []) + " " + (tag.get("id") or "")
    if MOBILE_ONLY.search(ident):
        return True
    if TOGGLE_PANEL.search(ident):
        return False
    if tag.has_attr("hidden") or tag.get("aria-hidden") == "true":
        return True
    if HIDDEN_STYLE.search(tag.get("style", "") or ""):
        return True
    role = (tag.get("role") or "").lower()
    if role in ("navigation", "banner", "contentinfo", "search", "dialog", "alert"):
        return True
    return bool(ident.strip()) and bool(BOILERPLATE_HINTS.search(ident))


def _reveal_contacts(soup: BeautifulSoup) -> None:
    """Make e-mail addresses / phone numbers stored in attributes part of the visible text."""
    def has_contact_attr(t: Tag) -> bool:
        return any(
            (n.startswith("data-") and "mail" in n)
            or (n == "href" and isinstance(v, str) and v.lower().startswith(("mailto:", "tel:")))
            for n, v in (t.attrs or {}).items()
        )

    for t in soup.find_all(has_contact_attr):
        visible = t.get_text(" ", strip=True)
        additions = []
        for name, value in list(t.attrs.items()):
            if not isinstance(value, str):
                continue
            if name.startswith("data-") and "mail" in name and EMAIL.fullmatch(value.strip()):
                additions.append(f"Email: {value.strip()}")
            elif name == "href" and value.lower().startswith("mailto:"):
                addr = unquote(value[7:].split("?")[0]).strip()
                if EMAIL.fullmatch(addr) and addr not in visible:
                    additions.append(f"Email: {addr}")
            elif name == "href" and value.lower().startswith("tel:"):
                num = unquote(value[4:]).strip()
                digits = re.sub(r"\D", "", num)
                if len(digits) >= 6 and digits not in re.sub(r"\D", "", visible):
                    additions.append(f"Phone: {num}")
        if additions:
            new = soup.new_tag("span")
            new.string = " " + " ".join(additions) + " "
            # Buttons are removed later, so put the text next to them rather than inside.
            (t.insert_after(new) if t.name == "button" else t.append(new))


_FOOTER_NOISE = re.compile(r"copyright|all rights reserved|website design|designed by|powered by|atlas softweb|©", re.I)


def footer_facts(soup: BeautifulSoup) -> str:
    """Facts from the site footer (leadership names + roles, address, phone, email), without its
    navigation link lists. The footer is the same on every page, so the crawler indexes it once."""
    foot = soup.find("footer") or soup.find(class_=re.compile(r"footer-section|site-footer"))
    if foot is None:
        return ""
    foot = copy.copy(foot)
    for t in foot(["script", "style", "svg", "img", "form", "button", "iframe"]):
        t.decompose()
    for a in foot.find_all("a"):
        href = (a.get("href") or "").lower()
        if href.startswith(("mailto:", "tel:")):
            a.unwrap()  # keep the address / number text
        else:
            a.decompose()  # navigation links carry no facts
    for li in foot.find_all("li"):  # <li><span class="name">X</span><span class="designation">Y</span></li>
        spans = [s.get_text(" ", strip=True) for s in li.find_all("span", recursive=False)]
        if len(spans) >= 2 and all(spans):
            li.clear()
            li.append(" — ".join(spans))
    lines: list[str] = []
    for line in foot.get_text("\n", strip=True).splitlines():
        line = re.sub(r"\s+", " ", line).strip()
        if len(line) < 3 or _FOOTER_NOISE.search(line) or line in lines:
            continue
        lines.append(line)
    # A footer made only of short menu headings (no names, numbers or addresses) holds no facts.
    return "\n".join(lines) if re.search(r"—|\d{5,}|@", "\n".join(lines)) else ""


def _breadcrumb(soup: BeautifulSoup, url: str, title: str) -> str:
    crumbs: list[str] = []
    bc = soup.find(class_=re.compile(r"breadcrumb", re.I)) or soup.find(attrs={"aria-label": re.compile("breadcrumb", re.I)})
    if bc:
        items = [re.sub(r"\s+", " ", x.get_text(" ", strip=True)) for x in bc.find_all(["a", "span", "li"])]
        for it in items:
            if it and it not in crumbs and it.lower() not in ("home", "»", "›", "/", ">"):
                crumbs.append(it)
    path = urlparse(url).path.strip("/").split("/")
    if not crumbs and path and path[0] == "author":
        return "Faculty profile"
    if not crumbs:  # derive from the URL path: /admissions/ug-admissions/ug-science/
        crumbs = [p.replace("-", " ").title() for p in path[:-1] if p and not p.isdigit()]
    crumbs = [c for c in crumbs if c.lower() != (title or "").lower()]
    return " › ".join(crumbs[:4])


def _cell_text(cell: Tag) -> str:
    return re.sub(r"\s+", " ", cell.get_text(" ", strip=True))


def table_to_text(table: Tag) -> str:
    """Linearise a table: each row becomes 'Header: value; Header: value'."""
    rows = [r for r in table.find_all("tr") if r.find_parent("table") is table]
    if not rows:
        return ""
    grid = [[_cell_text(c) for c in r.find_all(["th", "td"])] for r in rows]
    grid = [r for r in grid if any(r)]
    if not grid:
        return ""
    first_is_header = bool(rows[0].find("th")) or (len(grid) > 1 and all(c and not re.search(r"\d{3,}", c) for c in grid[0]))
    lines = []
    if first_is_header and len(grid) > 1:
        headers = grid[0]
        for r in grid[1:]:
            pairs = []
            for i, v in enumerate(r):
                if not v:
                    continue
                h = headers[i] if i < len(headers) and headers[i] else ""
                pairs.append(f"{h}: {v}" if h and h != v else v)
            if pairs:
                lines.append("; ".join(pairs))
    else:
        lines = [" | ".join(c for c in r if c) for r in grid]
    return "\n".join(lines)


_HEADING_TAGS = ("h1", "h2", "h3", "h4", "h5", "h6")


def _card_line(tag: Tag) -> str | None:
    """A profile/staff card: a block made only of short headings, e.g.
    <div><h2>Dr. Pravida Raja A.C.</h2><h4>Head, Assistant Professor</h4></div>.
    Rendered as one line ("Name — Position") so the role stays attached to the right person
    instead of becoming separate empty headings."""
    if tag.name in _HEADING_TAGS or tag.name not in ("div", "li", "section", "article", "a"):
        return None
    kids = [c for c in tag.children if isinstance(c, Tag)]
    stray_text = "".join(str(c) for c in tag.children if isinstance(c, NavigableString)).strip()
    if len(kids) < 2 or stray_text or not all(c.name in _HEADING_TAGS for c in kids):
        return None
    parts = [_cell_text(c) for c in kids]
    parts = [p for p in parts if p]
    if not parts or sum(len(p) for p in parts) > 200:
        return None
    return " — ".join(parts)


def _render(node: Tag, out: list[str]) -> None:
    """Walk the DOM, emitting markdown-ish lines."""
    for child in node.children:
        if isinstance(child, Comment):
            continue
        if isinstance(child, NavigableString):
            s = str(child)
            if s.strip():
                out.append(s)
            continue
        if not isinstance(child, Tag):
            continue
        name = child.name.lower()
        card = _card_line(child)
        if card:
            out.append(f"\n{card}\n")
        elif name == "summary" or PANEL_TITLE.search(" ".join(child.get("class", []) or [])):
            text = _cell_text(child)
            if text:
                out.append(f"\n\n#### {text}\n\n")
        elif name in ("h1", "h2", "h3", "h4", "h5", "h6"):
            text = _cell_text(child)
            if text:
                out.append(f"\n\n{'#' * int(name[1])} {text}\n\n")
        elif name == "table":
            text = table_to_text(child)
            if text:
                out.append(f"\n\n{text}\n\n")
        elif name == "li":
            out.append("\n- ")
            _render(child, out)
            out.append("\n")
        elif name == "br":
            out.append("\n")
        elif name == "img":
            alt = (child.get("alt") or "").strip()
            if len(alt) > 25:  # meaningful alt text only (not "logo", "image1")
                out.append(f"\n{alt}\n")
        elif name in BLOCK_TAGS:
            out.append("\n")
            _render(child, out)
            out.append("\n")
        else:
            _render(child, out)
            if name in SPACED_INLINE:
                out.append(" ")


def clean_lines(text: str) -> str:
    # Some WordPress content is double-encoded ("Chemistry &amp;amp; Zoology" in the HTML).
    if "&" in text:
        text = html.unescape(text)
    lines = []
    prev = None
    for raw in text.splitlines():
        line = re.sub(r"[ \t ​]+", " ", raw).strip()
        if not line or NOISE_LINES.match(line):
            if lines and lines[-1] != "":
                lines.append("")
            continue
        if line == prev:  # consecutive duplicates (e.g. repeated card titles)
            continue
        if line in ("-", "- -"):
            continue
        lines.append(line)
        prev = line
    return re.sub(r"\n{3,}", "\n\n", "\n".join(lines)).strip()


def extract_html(html: str, url: str) -> ExtractedPage:
    soup = BeautifulSoup(html, "lxml")

    title = ""
    og = soup.find("meta", property="og:title")
    if og and og.get("content"):
        title = og["content"].strip()
    elif soup.title and soup.title.string:
        title = soup.title.string.strip()
    title = re.sub(r"\s*[|–-]\s*St\.? Xavier.*$", "", title, flags=re.I).strip() or title
    title = re.sub(r",?\s*Author at .*$", "", title, flags=re.I).strip() or title  # faculty profile pages

    def meta(prop: str) -> str | None:
        m = soup.find("meta", property=prop) or soup.find("meta", attrs={"name": prop})
        return m.get("content") if m and m.get("content") else None

    meta_modified = meta("article:modified_time") or meta("og:updated_time")
    meta_published = meta("article:published_time")
    lang = soup.html.get("lang") if soup.html else None

    # Links and breadcrumb are read before cleaning so menus still feed the crawl frontier.
    links = discover_links(soup, url)
    section = _breadcrumb(soup, url, title)
    footer = footer_facts(soup)

    _reveal_contacts(soup)
    for t in soup(REMOVE_TAGS):
        t.decompose()
    for c in soup.find_all(string=lambda s: isinstance(s, Comment)):
        c.extract()
    for t in soup.find_all(_is_boilerplate):
        if t.name not in ("html", "body", "main"):
            t.decompose()

    root = (
        soup.find("main")
        or soup.find(attrs={"data-elementor-type": "wp-page"})
        or soup.find(attrs={"data-elementor-type": "single-post"})
        or soup.find("article")
        or soup.find(id="content")
        or soup.body
        or soup
    )
    parts: list[str] = []
    _render(root, parts)
    text = clean_lines("".join(parts))

    # URLs written as plain text inside the content ("visit https://…") are links too.
    known = {u for u, _ in links}
    for m in TEXT_URL.finditer(text):
        u = normalize_url(m.group(0).rstrip(".,;:"))
        if u and u not in known:
            links.append((u, ""))
            known.add(u)

    return ExtractedPage(title=title, text=text, links=links, section=section, footer=footer,
                         meta_modified=meta_modified,
                         meta_published=meta_published, lang=lang)
