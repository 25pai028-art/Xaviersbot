"""Hybrid retrieval: bge-m3 vector search + BM25 keyword search, fused with Reciprocal Rank Fusion,
graded for relevance, freshness-aware, with same-section neighbour expansion.

Freshness: for ordinary questions a newer source only breaks ties. For time-sensitive questions
(fees, deadlines, notices, exams, events…) hits are ranked by relevance plus a recency bonus, and an
old source is dropped when a source from a later academic year covers the same question, so a 2025 fee
notice is never used when 2026-27 fee dates exist, but is still used (with a warning) when nothing
newer covers it.
"""
from __future__ import annotations

import math
import time
from dataclasses import dataclass
from datetime import datetime, timezone

from app.config import get_settings
from app.rag import vectorstore
from app.rag.bm25 import bm25_index, tokenize
from app.rag.chunker import count_tokens
from app.rag.vectorstore import Hit

RRF_K = 60
CANDIDATES = 12
CANDIDATES_TIME_SENSITIVE = 24  # look wider so the newest notice gets a chance to be found
# Tie-breaker between near-equal hits (a newer notice beats an older one on the same topic). A hit that
# ranks well in BOTH vector and keyword search scores ~0.015 more than one found by only one of them,
# so freshness can never lift a one-sided hit above it.
FRESHNESS_WEIGHT = 0.0015
# Time-sensitive questions: added to the relevance grade (grades of on-topic vs off-topic hits differ by
# ~0.05-0.15), so a newer document on the same topic wins but a newer off-topic page does not.
FRESHNESS_WEIGHT_TIME_SENSITIVE = 0.03
SAME_TOPIC_GRADE_MARGIN = 0.03  # "about as relevant" when deciding that a newer source replaces an old one
FRESH_DAYS = 180  # fully "current"
STALE_DAYS = 540  # ~18 months: old for fees/deadlines/events
DROP_OLD_IF_NEWER_WITHIN_DAYS = 365  # drop stale sources when one this recent exists
MAX_CHUNKS_PER_SOURCE = 2  # so one long page cannot fill every slot
KEYWORD_WEIGHT = 0.15  # how much exact query-term coverage adds to the relevance grade
# Ordinary questions: a college web page beats a PDF that ranks a few places higher (a hit found by both
# vector and keyword search still wins by ~0.015). Pages are written for visitors; PDFs are often reports,
# minutes or forms that merely mention the topic.
PAGE_BONUS = 0.001


@dataclass
class Retrieval:
    hits: list[Hit]  # best first, graded and within the context budget
    best_score: float  # relevance grade of the best hit (0..1+)
    relevant: bool  # passed the similarity threshold
    time_sensitive: bool = False


def _cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    return float(dot / (na * nb)) if na and nb else 0.0


def keyword_coverage(query: str, text: str) -> float:
    """Share of the query's content words that literally occur in the chunk."""
    terms = set(tokenize(query))
    if not terms:
        return 0.0
    words = set(tokenize(text))
    return len(terms & words) / len(terms)


def _is_home_page(url: str) -> bool:
    return url.rstrip("/").count("/") <= 2  # https://sxca.edu.in/ : its date changes daily, not its content


def content_age_days(meta: dict) -> float | None:
    """Age of what a source says. Uses the older of its date and its academic year
    (a 2022-23 report uploaded in 2026 is still 2022-23 information). None if undated."""
    if _is_home_page(meta.get("url", "")):
        return None
    ts = float(meta.get("date_ts") or 0)
    ay = str(meta.get("academic_year") or "")
    if ay[:4].isdigit() and 2000 <= int(ay[:4]) <= 2100:
        ay_ts = datetime(int(ay[:4]), 6, 1, tzinfo=timezone.utc).timestamp()  # Indian academic year starts ~June
        ts = min(ts, ay_ts) if ts else ay_ts
    if not ts:
        return None
    return max(0.0, (time.time() - ts) / 86400)


def recency(meta: dict, time_sensitive: bool) -> float:
    """Ranking bonus in [-1, 1]; undated sources (faculty pages, rules) are neutral (0)."""
    age = content_age_days(meta)
    if age is None:
        return 0.0
    if not time_sensitive:
        return max(0.0, 1.0 - age / 730)  # gentle: 1.0 today → 0 at two years
    if age <= FRESH_DAYS:
        return 1.0
    if age >= STALE_DAYS:
        return -1.0
    return 1.0 - 2.0 * (age - FRESH_DAYS) / (STALE_DAYS - FRESH_DAYS)


def hybrid_search(query: str, query_embedding: list[float], k: int | None = None,
                  grade_query: str | None = None, time_sensitive: bool = False) -> Retrieval:
    """`query` (may include expanded abbreviations) drives search; `grade_query` (the user's words)
    is used for keyword coverage so the expansion doesn't dilute it."""
    s = get_settings()
    k = k or s.retrieval_top_k
    grade_query = grade_query or query
    n = CANDIDATES_TIME_SENSITIVE if time_sensitive else CANDIDATES
    weight = FRESHNESS_WEIGHT_TIME_SENSITIVE if time_sensitive else FRESHNESS_WEIGHT

    dense = vectorstore.query(query_embedding, n)
    keyword = bm25_index.search(query, n)

    fused: dict[str, float] = {}
    for rank, h in enumerate(dense):
        fused[h.chunk_id] = fused.get(h.chunk_id, 0) + 1 / (RRF_K + rank + 1)
    for rank, (cid, _score) in enumerate(keyword):
        fused[cid] = fused.get(cid, 0) + 1 / (RRF_K + rank + 1)

    by_id = {h.chunk_id: h for h in dense}
    missing = [cid for cid in fused if cid not in by_id]
    for cid, (text, meta, emb) in vectorstore.get_chunks(missing, with_embeddings=True).items():
        by_id[cid] = Hit(chunk_id=cid, text=text, metadata=meta, score=_cosine(query_embedding, emb or []))

    candidates: list[Hit] = []
    for cid, rrf in fused.items():
        h = by_id.get(cid)
        if h is None:
            continue
        cov = keyword_coverage(grade_query, h.text)
        h.signals = {"cosine": round(h.score, 3), "keyword_coverage": round(cov, 2), "rrf": round(rrf, 4)}
        h.signals["grade"] = round(h.score + KEYWORD_WEIGHT * cov, 3)
        age = content_age_days(h.metadata)
        h.signals["age_days"] = None if age is None else int(age)
        fresh = recency(h.metadata, time_sensitive)
        if time_sensitive:
            h.signals["rank_score"] = h.signals["grade"] + weight * fresh
        else:
            page = PAGE_BONUS if h.metadata.get("content_type") == "html" else 0.0
            h.signals["rank_score"] = rrf + weight * fresh + page
        candidates.append(h)

    candidates.sort(key=lambda h: h.signals["rank_score"], reverse=True)
    best = max((h.signals["grade"] for h in candidates), default=0.0)
    threshold = s.similarity_threshold
    good = [h for h in candidates if h.signals["grade"] >= threshold * 0.9]
    if time_sensitive:
        good = drop_outdated(good, threshold)
    good = _diverse(good)[: k * 2]
    return Retrieval(hits=_budget(_with_neighbours(good[:k], k), s.max_context_tokens),
                     best_score=best, relevant=best >= threshold, time_sensitive=time_sensitive)


def content_year(meta: dict) -> int | None:
    """Academic year (its starting calendar year) a source belongs to: its stated academic year, else the
    academic year of its date (June–May). None for undated sources and the home page."""
    if _is_home_page(meta.get("url", "")):
        return None
    ay = str(meta.get("academic_year") or "")
    if ay[:4].isdigit() and 2000 <= int(ay[:4]) <= 2100:
        return int(ay[:4])
    ts = float(meta.get("date_ts") or 0)
    if not ts:
        return None
    d = datetime.fromtimestamp(ts, tz=timezone.utc)
    return d.year if d.month >= 6 else d.year - 1


def drop_outdated(hits: list[Hit], threshold: float = 0.5) -> list[Hit]:
    """For time-sensitive questions, remove a source that a newer one replaces. An old source that nothing
    newer replaces is kept; the answer then says how old it is.

    A source is replaced when
    - a source from a LATER academic year matches at least as many of the question's words and is itself
      relevant (the 2025 fee notice vs the 2026-27 calendar that also gives the fee-payment date; a more
      specific old document often *scores* higher, so relevance alone can't decide), or
    - it is older than ~18 months and a source from the last year is about as relevant."""
    recent = [h for h in hits if h.signals.get("age_days") is not None
              and h.signals["age_days"] <= DROP_OLD_IF_NEWER_WITHIN_DAYS]
    years = {h.chunk_id: content_year(h.metadata) for h in hits}

    def replaced(old: Hit) -> bool:
        old_year, old_cov = years[old.chunk_id], old.signals.get("keyword_coverage", 0)
        if old_year is not None and any(
                years[n.chunk_id] is not None and years[n.chunk_id] > old_year
                and n.signals.get("keyword_coverage", 0) >= old_cov and n.signals.get("grade", 0) >= threshold
                for n in hits):
            return True
        age = old.signals.get("age_days")
        if age is None or age < STALE_DAYS:
            return False
        return any(r.signals.get("grade", 0) >= old.signals.get("grade", 0) - SAME_TOPIC_GRADE_MARGIN for r in recent)

    return [h for h in hits if not replaced(h)]


def _diverse(hits: list[Hit]) -> list[Hit]:
    per_source: dict = {}
    out = []
    for h in hits:
        src = h.metadata.get("source_id")
        if per_source.get(src, 0) >= MAX_CHUNKS_PER_SOURCE:
            continue
        per_source[src] = per_source.get(src, 0) + 1
        out.append(h)
    return out


def _with_neighbours(hits: list[Hit], k: int) -> list[Hit]:
    """Add the next chunk of the best hit's section, so answers are not cut off mid-list/table."""
    if not hits:
        return hits
    top = hits[0]
    src, idx = top.metadata.get("source_id"), top.metadata.get("chunk_index")
    if src is None or idx is None:
        return hits
    nxt_id = f"{src}:{int(idx) + 1}"
    if any(h.chunk_id == nxt_id for h in hits):
        return hits
    nxt = vectorstore.get_chunks([nxt_id]).get(nxt_id)
    if not nxt or (nxt[1].get("heading") or "") != (top.metadata.get("heading") or ""):
        return hits
    neighbour = Hit(chunk_id=nxt_id, text=nxt[0], metadata=nxt[1], score=top.score, signals={"neighbour_of": top.chunk_id})
    return [top, neighbour] + hits[1:k]


def _budget(hits: list[Hit], max_tokens: int) -> list[Hit]:
    """Keep hits in order until the context budget is used (prompt reading time on CPU grows with it)."""
    kept, used = [], 0
    for h in hits:
        n = count_tokens(h.text)
        if kept and used + n > max_tokens:
            continue  # a later, shorter hit may still fit
        kept.append(h)
        used += n
    return kept
