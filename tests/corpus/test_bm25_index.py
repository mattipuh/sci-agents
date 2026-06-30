import pytest
from corpus.bm25_index import BM25Index
from corpus.retriever import PaperChunk


def _chunk(chunk_id: str, text: str) -> PaperChunk:
    return PaperChunk(
        chunk_id=chunk_id, paper_id="2301.00001", title="T",
        authors=["A"], year=2023, journal="", section="results",
        text=text, score=0.0, source_url="url",
    )


def test_search_returns_relevant_chunks(tmp_path):
    idx = BM25Index(str(tmp_path))
    idx.add_chunks("ai_ml_systems", [
        _chunk("id-0", "transformer attention mechanism neural network"),
        _chunk("id-1", "random forest decision tree ensemble method"),
        _chunk("id-2", "attention head query key value matrix"),
    ])
    results = idx.search("ai_ml_systems", "attention transformer", top_k=2)
    assert len(results) == 2
    ids = {c.chunk_id for c, _ in results}
    assert ids & {"id-0", "id-2"}


def test_search_unknown_domain_returns_empty(tmp_path):
    idx = BM25Index(str(tmp_path))
    assert idx.search("unknown_domain", "query", top_k=5) == []


def test_scores_are_non_negative(tmp_path):
    idx = BM25Index(str(tmp_path))
    idx.add_chunks("mathematics", [_chunk("id-0", "stochastic optimisation convergence")])
    results = idx.search("mathematics", "stochastic", top_k=1)
    assert all(score >= 0.0 for _, score in results)


def test_save_and_load(tmp_path):
    idx = BM25Index(str(tmp_path))
    idx.add_chunks("ai_ml_systems", [_chunk("id-0", "neural network backpropagation")])
    idx.save("ai_ml_systems")

    idx2 = BM25Index(str(tmp_path))
    assert idx2.load("ai_ml_systems") is True
    results = idx2.search("ai_ml_systems", "neural network", top_k=1)
    assert results[0][0].chunk_id == "id-0"


def test_load_returns_false_when_not_found(tmp_path):
    idx = BM25Index(str(tmp_path))
    assert idx.load("nonexistent") is False


def test_add_chunks_accumulates_across_calls(tmp_path):
    idx = BM25Index(str(tmp_path))
    idx.add_chunks("ai_ml_systems", [_chunk("id-0", "first chunk")])
    idx.add_chunks("ai_ml_systems", [_chunk("id-1", "second chunk")])
    results = idx.search("ai_ml_systems", "chunk", top_k=5)
    ids = {c.chunk_id for c, _ in results}
    assert "id-0" in ids and "id-1" in ids
