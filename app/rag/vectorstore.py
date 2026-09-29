"""Persistent ChromaDB collection holding chunk embeddings + metadata."""
from __future__ import annotations

from dataclasses import dataclass, field
from functools import lru_cache

from app.config import get_settings

COLLECTION = "sxca_chunks"


@dataclass
class Hit:
    chunk_id: str
    text: str
    score: float  # cosine similarity to the query, 0..1 (higher is better)
    metadata: dict
    signals: dict = field(default_factory=dict)  # retrieval diagnostics (ranks, keyword coverage, …)


@lru_cache
def _collection():
    import chromadb
    from chromadb.config import Settings as ChromaSettings

    s = get_settings()
    client = chromadb.PersistentClient(path=str(s.chroma_dir), settings=ChromaSettings(anonymized_telemetry=False))
    return client.get_or_create_collection(COLLECTION, metadata={"hnsw:space": "cosine"})


def add_chunks(ids: list[str], texts: list[str], embeddings: list[list[float]], metadatas: list[dict]) -> None:
    if not ids:
        return
    col = _collection()
    for i in range(0, len(ids), 500):
        col.upsert(
            ids=ids[i:i + 500], documents=texts[i:i + 500],
            embeddings=embeddings[i:i + 500], metadatas=metadatas[i:i + 500],
        )


def delete_source(source_id: int) -> None:
    _collection().delete(where={"source_id": source_id})


def query(embedding: list[float], k: int, where: dict | None = None) -> list[Hit]:
    col = _collection()
    if col.count() == 0:
        return []
    res = col.query(query_embeddings=[embedding], n_results=min(k, col.count()), where=where,
                    include=["documents", "metadatas", "distances"])
    hits = []
    for cid, doc, meta, dist in zip(res["ids"][0], res["documents"][0], res["metadatas"][0], res["distances"][0]):
        hits.append(Hit(chunk_id=cid, text=doc, score=1.0 - float(dist), metadata=meta or {}))
    return hits


def get_chunks(ids: list[str], with_embeddings: bool = False) -> dict[str, tuple[str, dict, list[float] | None]]:
    """{chunk_id: (text, metadata, embedding|None)} for the given ids (missing ids are skipped)."""
    if not ids:
        return {}
    include = ["documents", "metadatas"] + (["embeddings"] if with_embeddings else [])
    res = _collection().get(ids=ids, include=include)
    embs = res.get("embeddings") if with_embeddings else None
    out = {}
    for i, cid in enumerate(res["ids"]):
        out[cid] = (res["documents"][i], res["metadatas"][i] or {},
                    list(embs[i]) if embs is not None else None)
    return out


def all_chunks(batch: int = 2000):
    """Yield (chunk_id, text, metadata) for every chunk (used to build the keyword index)."""
    col = _collection()
    total = col.count()
    for offset in range(0, total, batch):
        res = col.get(include=["documents", "metadatas"], limit=batch, offset=offset)
        yield from zip(res["ids"], res["documents"], res["metadatas"])


def count() -> int:
    return _collection().count()
