"""Turns stored Sources into chunks + embeddings in the vector store."""
from __future__ import annotations

from app.config import get_settings
from app.db.models import Source
from app.rag import vectorstore
from app.rag.chunker import Chunk, chunk_text
from app.rag.embeddings import embed_texts


def chunk_header(src: Source) -> str:
    """'Admissions › UG Admissions › UG Science' — where the text sits on the site."""
    if src.section and src.title and src.title not in src.section:
        return f"{src.section} › {src.title}"
    return src.title or src.section or ""


def chunk_metadata(src: Source, heading: str, index: int) -> dict:
    # Chroma metadata values must be str/int/float/bool (no None).
    return {
        "source_id": src.id,
        "url": src.url,
        "title": src.title or "",
        "section": src.section or "",
        "parent_url": src.parent_url or "",
        "heading": heading or "",
        "chunk_index": index,
        "source_type": src.source_type,
        "content_type": src.content_type,
        "language": src.language or "en",
        "academic_year": src.academic_year or "",
        "date": src.published_at.date().isoformat() if src.published_at else "",
        "date_ts": int(src.published_at.timestamp()) if src.published_at else 0,
    }


def make_chunks(src: Source) -> list[Chunk]:
    s = get_settings()
    return chunk_text(src.text, title=chunk_header(src), target_tokens=s.chunk_target_tokens,
                      overlap_tokens=s.chunk_overlap_tokens)


def index_sources(sources: list[Source]) -> dict[int, int]:
    """(Re)index several sources with one batched embedding call. Returns {source_id: chunk_count}."""
    batch: list[tuple[Source, Chunk]] = []
    for src in sources:
        vectorstore.delete_source(src.id)
        batch += [(src, c) for c in make_chunks(src)]
    counts = {src.id: 0 for src in sources}
    if not batch:
        return counts
    texts = [c.text for _, c in batch]
    embeddings = embed_texts(texts)
    vectorstore.add_chunks(
        ids=[f"{src.id}:{c.index}" for src, c in batch],
        texts=texts,
        embeddings=embeddings,
        metadatas=[chunk_metadata(src, c.heading, c.index) for src, c in batch],
    )
    for src, _ in batch:
        counts[src.id] += 1
    return counts


def index_source(src: Source) -> int:
    return index_sources([src])[src.id]


def remove_source(src: Source) -> None:
    vectorstore.delete_source(src.id)
