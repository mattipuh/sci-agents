from unittest.mock import MagicMock, patch

import numpy as np


def _mock_model(dim: int = 768):
    m = MagicMock()
    def encode(texts, **kw):
        n = len(texts) if isinstance(texts, list) else 1
        return np.random.rand(n, dim).astype(np.float32)
    m.encode.side_effect = encode
    return m


def test_embed_query_returns_768_floats():
    with patch("corpus.embedder._get_model", return_value=_mock_model()):
        from corpus.embedder import embed_query
        result = embed_query("attention mechanisms")
    assert len(result) == 768
    assert all(isinstance(v, float) for v in result)


def test_embed_documents_returns_list_of_vectors():
    with patch("corpus.embedder._get_model", return_value=_mock_model()):
        from corpus.embedder import embed_documents
        result = embed_documents(["text one", "text two"])
    assert len(result) == 2
    assert all(len(v) == 768 for v in result)


def test_embed_query_uses_search_query_prefix():
    m = _mock_model()
    with patch("corpus.embedder._get_model", return_value=m):
        from corpus.embedder import embed_query
        embed_query("hello world")
    assert m.encode.call_args[0][0] == ["search_query: hello world"]


def test_embed_documents_uses_search_document_prefix():
    m = _mock_model()
    with patch("corpus.embedder._get_model", return_value=m):
        from corpus.embedder import embed_documents
        embed_documents(["chunk one"])
    assert m.encode.call_args[0][0] == ["search_document: chunk one"]


def test_embed_documents_empty_returns_empty():
    with patch("corpus.embedder._get_model", return_value=_mock_model()):
        from corpus.embedder import embed_documents
        assert embed_documents([]) == []
