"""Local bge-m3 embeddings (sentence-transformers). Always local, whatever LLM is active,
so switching the LLM never requires re-indexing."""
from __future__ import annotations

import logging
import threading
from functools import lru_cache

from app.config import get_settings

log = logging.getLogger(__name__)
_lock = threading.Lock()


@lru_cache
def _model():
    from sentence_transformers import SentenceTransformer

    s = get_settings()
    try:
        # Use the cached copy without contacting Hugging Face (faster, works offline).
        model = SentenceTransformer(s.embedding_model, device=s.embedding_device, local_files_only=True)
        log.info("Loaded embedding model %s from local cache", s.embedding_model)
    except OSError:
        log.info("Downloading embedding model %s (~2.3 GB, first run only)…", s.embedding_model)
        model = SentenceTransformer(s.embedding_model, device=s.embedding_device)
    model.max_seq_length = s.embedding_max_seq_length
    return model


def embed_texts(texts: list[str], show_progress: bool = False) -> list[list[float]]:
    s = get_settings()
    with _lock:  # the model is not re-entrant on CPU; serialise callers
        vecs = _model().encode(
            texts,
            batch_size=s.embedding_batch_size,
            normalize_embeddings=True,
            show_progress_bar=show_progress,
            convert_to_numpy=True,
        )
    return vecs.tolist()


def embed_query(text: str) -> list[float]:
    return embed_texts([text])[0]
