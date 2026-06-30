from unittest.mock import MagicMock, patch

from corpus.retriever import PaperChunk

_PAPER = {
    "arxiv_id": "1706.03762", "title": "Attention Is All You Need",
    "authors": ["Vaswani, A."], "year": 2017,
    "abstract": "We propose transformers.",
    "pdf_url": "https://arxiv.org/pdf/1706.03762",
    "source_url": "https://arxiv.org/abs/1706.03762",
    "comment": "NeurIPS 2017",
}

_CHUNK = PaperChunk(
    chunk_id="1706.03762-results-0", paper_id="1706.03762",
    title="Attention Is All You Need", authors=["Vaswani, A."], year=2017,
    journal="", section="results", text="Our model achieves 28.4 BLEU.",
    score=0.0, source_url="https://arxiv.org/abs/1706.03762",
)


def _make_retriever():
    from corpus.arxiv_retriever import ArxivRetriever
    r = ArxivRetriever.__new__(ArxivRetriever)
    r._cache = MagicMock()
    r._vector_store = MagicMock()
    r._bm25 = MagicMock()
    return r


def test_retrieve_returns_chunks():
    r = _make_retriever()
    r._cache.is_indexed.return_value = False
    r._cache.get_quality.return_value = None
    r._vector_store.search.return_value = [(_CHUNK, 0.9)]
    r._bm25.search.return_value = [(_CHUNK, 1.5)]

    with (
        patch("corpus.arxiv_retriever.arxiv_search", return_value=[_PAPER]),
        patch("corpus.arxiv_retriever.fetch_pdf", return_value="/tmp/paper.pdf"),
        patch("corpus.arxiv_retriever.chunk_pdf", return_value=[_CHUNK]),
        patch("corpus.arxiv_retriever.embed_documents", return_value=[[0.1] * 768]),
        patch("corpus.arxiv_retriever.embed_query", return_value=[0.1] * 768),
        patch("corpus.arxiv_retriever._fetch_quality_metadata", return_value=None),
    ):
        results = r.retrieve("transformer attention", "ai_ml_systems", top_k=5)

    assert len(results) >= 1
    assert all(isinstance(c, PaperChunk) for c in results)


def test_retrieve_skips_ingestion_for_cached_papers():
    r = _make_retriever()
    r._cache.is_indexed.return_value = True
    r._cache.get_quality.return_value = None
    r._vector_store.search.return_value = [(_CHUNK, 0.9)]
    r._bm25.search.return_value = []

    with (
        patch("corpus.arxiv_retriever.arxiv_search", return_value=[_PAPER]),
        patch("corpus.arxiv_retriever.fetch_pdf") as mock_fetch,
        patch("corpus.arxiv_retriever.embed_query", return_value=[0.1] * 768),
    ):
        r.retrieve("attention", "ai_ml_systems", top_k=5)

    mock_fetch.assert_not_called()


def test_retrieve_filters_by_min_year():
    r = _make_retriever()
    r._cache.is_indexed.return_value = False
    r._vector_store.search.return_value = []
    r._bm25.search.return_value = []

    old_paper = {**_PAPER, "year": 2010}
    with (
        patch("corpus.arxiv_retriever.arxiv_search", return_value=[old_paper]),
        patch("corpus.arxiv_retriever.fetch_pdf") as mock_fetch,
        patch("corpus.arxiv_retriever.embed_query", return_value=[0.1] * 768),
    ):
        r.retrieve("attention", "ai_ml_systems", top_k=5, min_year=2015)

    mock_fetch.assert_not_called()


def test_retrieve_scores_journal_articles_higher():
    r = _make_retriever()
    r._cache.is_indexed.return_value = True

    journal_chunk = PaperChunk("a-0", "2301.00001", "A", [], 2023, "", "results",
                               "text a", 0.0, "url")
    preprint_chunk = PaperChunk("b-0", "2301.00002", "B", [], 2023, "", "results",
                                "text b", 0.0, "url")

    def get_quality(arxiv_id):
        if arxiv_id == "2301.00001":
            return {"publication_type": "JournalArticle", "venue": "Nature",
                    "citation_count": 500, "influential_citation_count": 50}
        return {"publication_type": "Preprint", "venue": "",
                "citation_count": 0, "influential_citation_count": 0}

    r._cache.get_quality.side_effect = get_quality
    r._vector_store.search.return_value = [(journal_chunk, 0.8), (preprint_chunk, 0.85)]
    r._bm25.search.return_value = [(journal_chunk, 1.2), (preprint_chunk, 1.5)]

    with (
        patch("corpus.arxiv_retriever.arxiv_search", return_value=[]),
        patch("corpus.arxiv_retriever.embed_query", return_value=[0.1] * 768),
    ):
        results = r.retrieve("query", "ai_ml_systems", top_k=2)

    assert len(results) == 2
    assert results[0].paper_id == "2301.00001"  # journal article ranked first
