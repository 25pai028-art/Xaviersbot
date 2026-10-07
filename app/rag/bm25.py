"""In-memory BM25 keyword index over all chunks.

Vector search understands meaning but is weak on exact tokens: course codes
(BCA, B.Com), people's names, phone numbers, fee amounts. BM25 catches those.
The index is rebuilt automatically when the number of chunks changes.
"""
from __future__ import annotations

import logging
import re
import threading

from app.rag import vectorstore

log = logging.getLogger(__name__)

STOPWORDS = set("""a an the of in on at to for from by with and or is are was were be been being do does did
what which who whom whose when where why how can could should would will shall may might i me my we our you your
it its this that these those there here about as into than then please tell know give college xavier xaviers
st sxca ahmedabad sir madam mam maam miss mr mrs ms dr prof""".split())
_DOTTED = re.compile(r"\b([a-z])\.(?=[a-z]\b|[a-z]{2,}\b)")  # b.com → bcom, b.sc → bsc, m.a → ma
_TOKEN = re.compile(r"[a-z0-9]+|[^\W\d_a-z]+", re.UNICODE)


_HONORIFICS = {"mr", "mrs", "ms", "miss", "dr", "fr", "prof", "sr", "sj", "rev", "br"}
_ADDRESSES_PERSON = re.compile(r"\b(sir|madam|ma'?am|mam|miss|teacher|professor|prof|dr|mr|mrs|ms|fr|"
                               r"who is|who's|contact|email|e-mail)\b", re.I)


def _name_tokens(text: str) -> list[str]:
    """Lower-case name words without titles: "Dr. Fr. David K Roy, SJ" -> ['david', 'k', 'roy']."""
    return [w for w in re.findall(r"[a-z]+", text.lower()) if w not in _HONORIFICS]


def tokenize(text: str) -> list[str]:
    text = _DOTTED.sub(r"\1", text.lower())
    text = re.sub(r"(?<=\d)[,\s](?=\d{3}\b)", "", text)  # 20,000 → 20000
    return [t for t in _TOKEN.findall(text) if t not in STOPWORDS and (len(t) > 1 or t.isdigit())]


class BM25Index:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._bm25 = None
        self._ids: list[str] = []
        self._size = -1
        # Faculty profile pages (/author/…): name tokens -> chunk ids, to recognise people named in questions.
        self._people: dict[tuple[str, ...], list[str]] = {}

    def _ensure(self) -> None:
        size = vectorstore.count()
        if size == self._size:
            return
        with self._lock:
            if size == self._size:
                return
            from rank_bm25 import BM25Okapi

            ids, corpus, people = [], [], {}
            for cid, text, meta in vectorstore.all_chunks():
                ids.append(cid)
                corpus.append(tokenize(text))
                if "/author/" in str(meta.get("url", "")):
                    name = tuple(_name_tokens(str(meta.get("title", ""))))
                    if len(name) >= 2:
                        people.setdefault(name, []).append(cid)
            self._bm25 = BM25Okapi(corpus) if corpus else None
            self._ids, self._people, self._size = ids, people, size
            log.info("BM25 keyword index built over %d chunks", size)

    def warmup(self) -> None:
        """Build the index now (about 30 s for ~33,000 chunks on a laptop) instead of during the first question."""
        self._ensure()

    def search(self, query: str, k: int) -> list[tuple[str, float]]:
        """[(chunk_id, bm25 score)] best first; only chunks sharing at least one query term."""
        self._ensure()
        terms = tokenize(query)
        if self._bm25 is None or not terms:
            return []
        scores = self._bm25.get_scores(terms)
        order = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:k]
        return [(self._ids[i], float(scores[i])) for i in order if scores[i] > 0]

    def people_in(self, query: str) -> list[tuple[str, list[str]]]:
        """Faculty named in the question: [(full name, profile chunk ids)]. A full name ("Nisarg Vyas") always
        counts; a first name alone ("Nisarg sir") counts when it is rare (at most 3 people) and the question
        addresses a person (sir, madam, dr, prof, who is…)."""
        self._ensure()
        q = set(_name_tokens(query))
        full = [(" ".join(n), ids) for n, ids in self._people.items() if set(n) <= q]
        if full:
            return full
        if not _ADDRESSES_PERSON.search(query):
            return []
        for first in q:
            same = [(" ".join(n), ids) for n, ids in self._people.items() if n[0] == first]
            if 0 < len(same) <= 3:
                return same
        return []

    def invalidate(self) -> None:
        self._size = -1


bm25_index = BM25Index()
