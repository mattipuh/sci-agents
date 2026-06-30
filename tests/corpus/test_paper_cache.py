import pytest
from corpus.paper_cache import PaperCache


@pytest.fixture
def cache(tmp_path):
    return PaperCache(db_path=str(tmp_path / "test.db"))


def _mark(cache, arxiv_id="2301.00001", **kwargs):
    defaults = dict(
        domain="ai_ml_systems", title="Test Paper", authors=["Smith, J."],
        year=2023, abstract="Abstract.", source_url="https://arxiv.org/abs/2301.00001",
        publication_type="JournalArticle", venue="NeurIPS",
        citation_count=42, influential_citation_count=5,
    )
    defaults.update(kwargs)
    cache.mark_indexed(arxiv_id=arxiv_id, **defaults)


def test_is_indexed_returns_false_for_unknown(cache):
    assert cache.is_indexed("2301.00001") is False


def test_mark_indexed_then_is_indexed(cache):
    _mark(cache)
    assert cache.is_indexed("2301.00001") is True


def test_get_quality_returns_none_for_unknown(cache):
    assert cache.get_quality("9999.00000") is None


def test_get_quality_returns_stored_metadata(cache):
    _mark(cache, publication_type="Conference", venue="ICML",
          citation_count=10, influential_citation_count=2)
    q = cache.get_quality("2301.00001")
    assert q == {"publication_type": "Conference", "venue": "ICML",
                 "citation_count": 10, "influential_citation_count": 2}


def test_list_indexed_returns_all_papers(cache):
    _mark(cache, "2301.00001", title="Paper A")
    _mark(cache, "2301.00002", title="Paper B")
    papers = cache.list_indexed()
    assert len(papers) == 2
    assert {p["arxiv_id"] for p in papers} == {"2301.00001", "2301.00002"}


def test_mark_indexed_is_idempotent(cache):
    _mark(cache)
    _mark(cache, citation_count=99)  # should upsert without error
    assert cache.is_indexed("2301.00001") is True
    assert cache.get_quality("2301.00001")["citation_count"] == 99
