from unittest.mock import MagicMock, patch

from corpus.sources.arxiv import ARXIV_CATEGORIES, search

SAMPLE_ATOM = """<?xml version="1.0" encoding="UTF-8"?>
<feed xmlns="http://www.w3.org/2005/Atom"
      xmlns:arxiv="http://arxiv.org/schemas/atom">
  <entry>
    <id>http://arxiv.org/abs/1706.03762v5</id>
    <title>Attention Is All You Need</title>
    <author><name>Vaswani, Ashish</name></author>
    <author><name>Shazeer, Noam</name></author>
    <published>2017-06-12T00:00:00Z</published>
    <summary>The dominant sequence transduction models...</summary>
    <link href="https://arxiv.org/pdf/1706.03762v5"
          rel="related" type="application/pdf"/>
    <arxiv:comment>Published in NeurIPS 2017</arxiv:comment>
  </entry>
</feed>"""


def _mock_response(text):
    r = MagicMock()
    r.text = text
    r.raise_for_status = MagicMock()
    return r


def test_search_parses_paper_fields():
    with patch("httpx.get", return_value=_mock_response(SAMPLE_ATOM)):
        results = search("attention transformer", "ai_ml_systems", max_results=1)
    assert len(results) == 1
    p = results[0]
    assert p["arxiv_id"] == "1706.03762"
    assert p["title"] == "Attention Is All You Need"
    assert p["authors"] == ["Vaswani, Ashish", "Shazeer, Noam"]
    assert p["year"] == 2017
    assert "dominant sequence" in p["abstract"]
    assert p["pdf_url"] == "https://arxiv.org/pdf/1706.03762v5"
    assert p["source_url"] == "https://arxiv.org/abs/1706.03762"
    assert p["comment"] == "Published in NeurIPS 2017"


def test_search_uses_domain_categories():
    with patch("httpx.get", return_value=_mock_response("<feed/>")) as mock_get:
        search("transformers", "ai_ml_systems")
    params = mock_get.call_args[1]["params"]
    search_query = params["search_query"]
    assert "cs.AI" in search_query or "cs.LG" in search_query


def test_search_handles_empty_feed():
    with patch("httpx.get", return_value=_mock_response("<feed/>")):
        assert search("nothing", "ai_ml_systems") == []


def test_categories_cover_required_domains():
    for domain in ["ai_ml_systems", "mathematics", "ops_research"]:
        assert domain in ARXIV_CATEGORIES
        assert len(ARXIV_CATEGORIES[domain]) >= 1
