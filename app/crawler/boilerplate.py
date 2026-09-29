"""Site-wide boilerplate removal.

Per-page cleaning removes obvious menus/footers, but themes also repeat blocks
("Subscribe to Newsletter", quick-link sidebars, contact strips) inside the
content area. A line that appears on a large share of pages is boilerplate:
it is stripped from every page and kept exactly once, in a single "common site
information" source, so contact numbers etc. remain searchable without
polluting every page's chunks.
"""
from __future__ import annotations

import re
from collections import Counter

COMMON_SOURCE_URL = "site://common-information"
COMMON_SOURCE_TITLE = "St. Xavier's College Ahmedabad: leadership (Rector, Director, Principal) and contact"

_HEADING = re.compile(r"^#{1,6}\s+")


def _norm(line: str) -> str:
    return re.sub(r"\s+", " ", _HEADING.sub("", line)).strip().lower()


def find_boilerplate(page_texts: list[str], min_pages: int = 6, min_share: float = 0.15) -> set[str]:
    """Normalised lines present on at least `min_pages` pages and `min_share` of all pages."""
    if not page_texts:
        return set()
    counts: Counter[str] = Counter()
    for text in page_texts:
        counts.update({_norm(l) for l in text.splitlines() if len(_norm(l)) >= 3})
    threshold = max(min_pages, int(len(page_texts) * min_share))
    return {line for line, n in counts.items() if n >= threshold}


def strip_boilerplate(text: str, boilerplate: set[str]) -> str:
    if not boilerplate:
        return text
    kept = [l for l in text.splitlines() if _norm(l) not in boilerplate]
    # Drop headings left with nothing under them.
    out: list[str] = []
    for i, line in enumerate(kept):
        if _HEADING.match(line):
            rest = next((k for k in kept[i + 1:] if k.strip()), "")
            if not rest or (_HEADING.match(rest) and len(_HEADING.match(rest).group(0)) <= len(_HEADING.match(line).group(0))):
                continue
        out.append(line)
    return re.sub(r"\n{3,}", "\n\n", "\n".join(out)).strip()


def common_information(page_texts: list[str], boilerplate: set[str]) -> str:
    """The boilerplate lines, in first-seen order, as one text (indexed once)."""
    seen: set[str] = set()
    lines: list[str] = []
    for text in page_texts:
        for l in text.splitlines():
            n = _norm(l)
            if n in boilerplate and n not in seen:
                seen.add(n)
                lines.append(_HEADING.sub("", l).strip())
    return "\n".join(lines)


def site_information(footer: str, repeated: str) -> str:
    """Text of the single site-wide source: footer facts (leadership, address, phone, email) first,
    then other lines repeated across many pages."""
    nl = chr(10)
    parts = []
    if footer:
        people = [l for l in footer.splitlines() if " — " in l]
        other = [l for l in footer.splitlines() if " — " not in l]
        if people:
            roles = [f"{l.split(' — ', 1)[1]}: {l.split(' — ', 1)[0]}" for l in people]
            parts.append("## College leadership" + nl * 2 + nl.join(roles))
        if other:
            parts.append("## College address and contact" + nl * 2 + nl.join(other))
    if repeated:
        parts.append("## Other information shown across the website" + nl * 2 + repeated)
    return (nl * 2).join(parts)
