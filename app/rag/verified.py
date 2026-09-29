"""Matching questions against the college's official (verified) answers.

Each official answer has a main question plus optional alternative phrasings. Their
embeddings are cached in memory and rebuilt automatically when an admin edits them.
"""
from __future__ import annotations

import threading
from dataclasses import dataclass

from sqlalchemy import func, select

from app.db.models import VerifiedAnswer
from app.db.session import session_scope
from app.rag.embeddings import embed_texts


@dataclass
class VerifiedMatch:
    id: int
    question: str
    answer: str
    source_url: str | None
    score: float  # cosine similarity between the user's question and the closest phrasing


class VerifiedIndex:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._stamp: tuple | None = None
        self._rows: list[tuple[int, str, str, str | None]] = []  # (id, question, answer, url)
        self._vectors: list[tuple[int, list[float]]] = []  # (row index, embedding) per phrasing

    def _current_stamp(self) -> tuple:
        with session_scope() as db:
            return tuple(db.execute(
                select(func.count(VerifiedAnswer.id), func.max(VerifiedAnswer.updated_at))
                .where(VerifiedAnswer.active.is_(True))).one())

    def _ensure(self) -> None:
        stamp = self._current_stamp()
        if stamp == self._stamp:
            return
        with self._lock:
            if stamp == self._stamp:
                return
            with session_scope() as db:
                rows = db.scalars(select(VerifiedAnswer).where(VerifiedAnswer.active.is_(True))).all()
                data = [(r.id, r.question, r.answer, r.source_url,
                         [r.question] + [q.strip() for q in (r.alt_questions or "").splitlines() if q.strip()])
                        for r in rows]
            phrasings = [(i, p) for i, d in enumerate(data) for p in d[4]]
            vectors = embed_texts([p for _, p in phrasings]) if phrasings else []
            self._rows = [d[:4] for d in data]
            self._vectors = [(i, v) for (i, _), v in zip(phrasings, vectors)]
            self._stamp = stamp

    def best(self, query_embedding: list[float]) -> VerifiedMatch | None:
        self._ensure()
        if not self._vectors:
            return None
        # Embeddings are normalised, so the dot product is the cosine similarity.
        idx, score = max(((i, sum(a * b for a, b in zip(query_embedding, v))) for i, v in self._vectors),
                         key=lambda t: t[1])
        rid, q, a, url = self._rows[idx]
        return VerifiedMatch(id=rid, question=q, answer=a, source_url=url, score=float(score))

    def invalidate(self) -> None:
        self._stamp = None


verified_index = VerifiedIndex()


def record_use(answer_id: int) -> None:
    with session_scope() as db:
        row = db.get(VerifiedAnswer, answer_id)
        if row:
            row.times_used += 1
