"""Hybrid retrieval: bge-m3 vector search + BM25 keyword search, fused with Reciprocal Rank Fusion,
graded for relevance, with a small freshness boost and same-section neighbour expansion."""
from __future__ import annotations

import math
import time
from dataclasses import dataclass

from app.config import get_settings
from app.rag import vectorstore
from app.rag.bm25 import bm25_index, tokenize
from app.rag.chunker import count_tokens
from app.rag.vectorstore import Hit

RRF_K = 60
CANDIDATES = 12
# Tie-breaker between near-equal hits (a newer notice beats an older one on the same topic). A hit that
# ranks well in BOTH vector and keyword search scores ~0.015 more than one found by only one of them,
# so freshness can never lift a one-sided hit above it.
FRESHNESS_WEIGHT = 0.0015
MAX_CHUNKS_PER_SOURCE = 2  # so one long page cannot fill every slot
KEYWORD_WEIGHT = 0.15  # how much exact query-term coverage adds to the relevance grade


@dataclass
class Retrieval:
    hits: list[Hit]  # best first, graded and within the context budget
    best_score: float  # relevance grade of the best hit (0..1+)
    relevant: bool  # passed the similarity threshold


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


def _freshness(meta: dict) -> float:
    ts = meta.get("date_ts") or 0
    if not ts:
        return 0.0
    age_years = (time.time() - ts) / (365 * 86400)
    return max(0.0, 1.0 - age_years / 2)  # 1.0 today → 0 at two years old


def hybrid_search(query: str, query_embedding: list[float], k: int | None = None,
                  grade_query: str | None = None) -> Retrieval:
    """`query` (may include expanded abbreviations) drives search; `grade_query` (the user's words)
    is used for keyword coverage so the expansion doesn't dilute it."""
    s = get_settings()
    k = k or s.retrieval_top_k
    grade_query = grade_query or query

    dense = vectorstore.query(query_embedding, CANDIDATES)
    keyword = bm25_index.search(query, CANDIDATES)

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
        h.signals["rank_score"] = rrf + FRESHNESS_WEIGHT * _freshness(h.metadata)
        candidates.append(h)

    candidates.sort(key=lambda h: h.signals["rank_score"], reverse=True)
    best = max((h.signals["grade"] for h in candidates), default=0.0)
    threshold = s.similarity_threshold
    good = _diverse([h for h in candidates if h.signals["grade"] >= threshold * 0.9])[: k * 2]
    return Retrieval(hits=_budget(_with_neighbours(good[:k], k), s.max_context_tokens),
                     best_score=best, relevant=best >= threshold)


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
