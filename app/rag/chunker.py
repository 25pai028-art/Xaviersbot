"""Heading-aware chunking (~400–600 tokens with overlap).

Text uses markdown-style `#` headings (produced by the extractors). Each chunk
carries the document title and its heading path so a chunk is understandable
on its own, e.g. "Fee Structure > B.Com > Semester 1".
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from functools import lru_cache
from typing import Callable

_HEADING = re.compile(r"^(#{1,6})\s+(.*)$")
_SENTENCE_END = re.compile(r"(?<=[.!?।॥])\s+")


@dataclass
class Chunk:
    text: str  # what gets embedded / shown to the LLM (includes title + headings)
    heading: str
    index: int


@lru_cache
def _tokenizer():
    try:
        from transformers import AutoTokenizer

        from app.config import get_settings

        name = get_settings().embedding_model
        try:
            return AutoTokenizer.from_pretrained(name, local_files_only=True)
        except OSError:
            return AutoTokenizer.from_pretrained(name)
    except Exception:
        return None


def count_tokens(text: str) -> int:
    tok = _tokenizer()
    if tok is not None:
        return len(tok.encode(text, add_special_tokens=False))
    # Fallback approximation: ~4 chars/token for Latin, ~2 for Indic scripts.
    non_ascii = sum(1 for c in text if ord(c) > 127)
    return int((len(text) - non_ascii) / 4 + non_ascii / 2) + 1


def _split_long(text: str, limit: int, counter: Callable[[str], int]) -> list[str]:
    """Split an oversized paragraph by sentences, then by words."""
    if counter(text) <= limit:
        return [text]
    pieces, cur = [], ""
    for sent in _SENTENCE_END.split(text):
        candidate = f"{cur} {sent}".strip()
        if cur and counter(candidate) > limit:
            pieces.append(cur)
            cur = sent
        else:
            cur = candidate
    if cur:
        pieces.append(cur)
    out = []
    for p in pieces:
        if counter(p) <= limit:
            out.append(p)
            continue
        words, buf = p.split(), []
        for w in words:
            buf.append(w)
            if counter(" ".join(buf)) >= limit:
                out.append(" ".join(buf))
                buf = []
        if buf:
            out.append(" ".join(buf))
    return out


def _sections(text: str) -> list[tuple[str, list[str]]]:
    """[(heading path, [paragraphs])]."""
    path: list[tuple[int, str]] = []
    sections: list[tuple[str, list[str]]] = []
    paras: list[str] = []
    buf: list[str] = []

    def flush_para():
        if buf:
            paras.append("\n".join(buf).strip())
            buf.clear()

    def flush_section() -> bool:
        flush_para()
        if paras:
            sections.append((" > ".join(h for _, h in path), paras.copy()))
            paras.clear()
            return True
        return False

    for line in text.splitlines():
        m = _HEADING.match(line.strip())
        if m:
            had_content = flush_section()
            level = len(m.group(1))
            # A heading with nothing under it (e.g. a name on a staff card) is still information:
            # keep its text as a line of the parent section instead of silently dropping it.
            if not had_content and path and level <= path[-1][0]:
                parent = " > ".join(h for _, h in path[:-1])
                if sections and sections[-1][0] == parent and sections[-1][1][-1].startswith("• "):
                    sections[-1][1][-1] += "\n• " + path[-1][1]
                else:
                    sections.append((parent, ["• " + path[-1][1]]))
            path = [(lv, h) for lv, h in path if lv < level] + [(level, m.group(2).strip())]
            continue
        if not line.strip():
            flush_para()
        else:
            buf.append(line.strip())
    flush_section()
    return sections


def chunk_text(
    text: str,
    title: str = "",
    target_tokens: int = 500,
    overlap_tokens: int = 60,
    counter: Callable[[str], int] = count_tokens,
) -> list[Chunk]:
    chunks: list[Chunk] = []

    def emit(heading: str, body: list[str]) -> None:
        body_text = "\n\n".join(body).strip()
        if len(body_text) < 8:  # short is fine (a name, a phone number); empty is not
            return
        header = title if not heading else f"{title} — {heading}" if title else heading
        chunks.append(Chunk(text=f"{header}\n\n{body_text}" if header else body_text, heading=heading, index=len(chunks)))

    for heading, paras in _sections(text):
        units: list[str] = []
        for p in paras:
            units += _split_long(p, target_tokens, counter)
        cur: list[str] = []
        cur_tokens = 0
        for u in units:
            ut = counter(u)
            if cur and cur_tokens + ut > target_tokens:
                emit(heading, cur)
                # Overlap: carry trailing units up to `overlap_tokens`.
                carry: list[str] = []
                carried = 0
                for prev in reversed(cur):
                    pt = counter(prev)
                    if carried + pt > overlap_tokens:
                        break
                    carry.insert(0, prev)
                    carried += pt
                cur, cur_tokens = carry, carried
            cur.append(u)
            cur_tokens += ut
        if cur:
            emit(heading, cur)

    # Merge tiny fragments into the previous chunk. A short section with its own heading
    # (e.g. one course's fee line) stays separate unless it is nearly empty.
    merged: list[Chunk] = []
    for c in chunks:
        size = counter(c.text)
        tiny = size < 25 or (size < 60 and merged and merged[-1].heading == c.heading)
        if merged and tiny and counter(merged[-1].text) + size <= target_tokens:
            merged[-1] = Chunk(text=merged[-1].text + "\n\n" + c.text, heading=merged[-1].heading, index=merged[-1].index)
        else:
            merged.append(Chunk(text=c.text, heading=c.heading, index=len(merged)))
    return merged
