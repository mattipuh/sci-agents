from unittest.mock import MagicMock, patch

import pytest

from corpus.retriever import PaperChunk
from corpus.vector_store import VectorStore


def _chunk(chunk_id="2301.00001-results-0"):
    return PaperChunk(
        chunk_id=chunk_id, paper_id="2301.00001", title="Test Paper",
        authors=["Smith, J."], year=2023, journal="NeurIPS",
        section="results", text="Our results show improvement.",
        score=0.0, source_url="https://arxiv.org/abs/2301.00001",
    )


@pytest.fixture
def mock_client():
    with patch("corpus.vector_store.QdrantClient") as cls:
        client = MagicMock()
        cls.return_value = client
        client.get_collections.return_value = MagicMock(collections=[])
        yield client


def test_upsert_calls_qdrant_with_correct_collection(mock_client):
    vs = VectorStore(url="http://localhost:6333")
    vs.upsert("ai_ml_systems", [_chunk()], [[0.1] * 768])
    mock_client.upsert.assert_called_once()
    assert mock_client.upsert.call_args[1]["collection_name"] == "domain_ai_ml_systems"


def test_search_returns_paper_chunks(mock_client):
    chunk = _chunk()
    hit = MagicMock()
    hit.score = 0.95
    hit.payload = {
        "chunk_id": chunk.chunk_id, "paper_id": chunk.paper_id,
        "title": chunk.title, "authors": chunk.authors, "year": chunk.year,
        "journal": chunk.journal, "section": chunk.section,
        "text": chunk.text, "source_url": chunk.source_url,
    }
    mock_client.search.return_value = [hit]

    vs = VectorStore(url="http://localhost:6333")
    results = vs.search("ai_ml_systems", [0.1] * 768, top_k=5)

    assert len(results) == 1
    returned_chunk, score = results[0]
    assert returned_chunk.chunk_id == chunk.chunk_id
    assert returned_chunk.score == 0.95
    assert score == 0.95


def test_search_uses_correct_collection_name(mock_client):
    mock_client.search.return_value = []
    vs = VectorStore(url="http://localhost:6333")
    vs.search("mathematics", [0.1] * 768, top_k=5)
    assert mock_client.search.call_args[1]["collection_name"] == "domain_mathematics"
