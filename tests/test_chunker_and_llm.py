from app.llm.base import ThinkingStreamFilter, strip_thinking
from app.rag.chunker import chunk_text


def words(n: int, w: str = "word") -> str:
    return " ".join(f"{w}{i}" for i in range(n))


def approx(text: str) -> int:  # deterministic token counter for tests (1 token per word)
    return len(text.split())


def test_chunks_keep_heading_path_and_title():
    text = "# Fees\n\n## B.Com\n\n" + words(30) + "\n\n## BBA\n\n" + words(30, "bba")
    chunks = chunk_text(text, title="Fee Structure 2026-27", target_tokens=100, counter=approx)
    assert len(chunks) == 2
    assert chunks[0].text.startswith("Fee Structure 2026-27 — Fees > B.Com")
    assert chunks[1].heading == "Fees > BBA"


def test_long_sections_split_with_overlap():
    paras = "\n\n".join(words(40, f"p{i}w") for i in range(10))
    chunks = chunk_text(paras, title="T", target_tokens=100, overlap_tokens=45, counter=approx)
    assert len(chunks) > 3
    assert all(approx(c.text) <= 100 + 45 + 5 for c in chunks)
    # the last paragraph of a chunk reappears at the start of the next one
    first_tail = chunks[0].text.split("\n\n")[-1]
    assert first_tail in chunks[1].text


def test_oversized_paragraph_is_split():
    chunks = chunk_text(words(1000), title="T", target_tokens=200, counter=approx)
    assert len(chunks) >= 5


def test_strip_thinking():
    assert strip_thinking("<think>let me reason</think>The answer is 5.") == "The answer is 5."
    assert strip_thinking("reasoning...</think>Final") == "Final"
    assert strip_thinking("Answer only") == "Answer only"


def test_thinking_stream_filter_handles_split_tags():
    f = ThinkingStreamFilter()
    pieces = ["Hel", "lo <th", "ink>secret ", "stuff</thi", "nk> world"]
    out = "".join(f.feed(p) for p in pieces) + f.flush()
    assert out == "Hello  world"
