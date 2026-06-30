from unittest.mock import MagicMock, patch

from corpus.chunker import _chunk_text, _detect_section, chunk_pdf

PAPER_TEXT = """Abstract
This paper proposes a novel attention mechanism.

Introduction
Neural networks have become ubiquitous in modern AI.
Attention has revolutionised sequence modelling.

Methods
We train a transformer on WMT 2014 using 8 attention heads.
Training uses the Adam optimiser with a warmup schedule.

Results
Our model achieves 28.4 BLEU on English-German translation.
This outperforms all prior single-model results by over 2 BLEU.

Discussion
The results confirm that attention alone is sufficient for sequence tasks.
"""


def _mock_pdf(pages_text: list[str]):
    mock_pdf = MagicMock()
    pages = []
    for text in pages_text:
        page = MagicMock()
        page.extract_text.return_value = text
        pages.append(page)
    mock_pdf.pages = pages
    mock_pdf.__enter__ = lambda s: s
    mock_pdf.__exit__ = MagicMock(return_value=False)
    return mock_pdf


def test_chunk_pdf_detects_sections(tmp_path):
    pdf_path = str(tmp_path / "paper.pdf")
    open(pdf_path, "w").close()
    with patch("pdfplumber.open", return_value=_mock_pdf([PAPER_TEXT])):
        chunks = chunk_pdf(pdf_path, "1706.03762", "Attention", ["Vaswani"], 2017,
                           "https://arxiv.org/abs/1706.03762")
    sections = {c.section for c in chunks}
    assert sections & {"abstract", "introduction", "methods", "results", "discussion"}


def test_chunk_pdf_sets_metadata(tmp_path):
    pdf_path = str(tmp_path / "paper.pdf")
    open(pdf_path, "w").close()
    with patch("pdfplumber.open", return_value=_mock_pdf([PAPER_TEXT])):
        chunks = chunk_pdf(pdf_path, "1706.03762", "Attention", ["A", "B"], 2017,
                           "https://arxiv.org/abs/1706.03762")
    assert len(chunks) >= 1
    for c in chunks:
        assert c.paper_id == "1706.03762"
        assert c.title == "Attention"
        assert c.authors == ["A", "B"]
        assert c.year == 2017
        assert c.score == 0.0
        assert c.chunk_id.startswith("1706.03762-")


def test_chunk_pdf_returns_empty_for_blank_pdf(tmp_path):
    pdf_path = str(tmp_path / "paper.pdf")
    open(pdf_path, "w").close()
    with patch("pdfplumber.open", return_value=_mock_pdf([""])):
        chunks = chunk_pdf(pdf_path, "0000.00000", "T", [], 2020, "url")
    assert chunks == []


def test_detect_section_recognises_headings():
    assert _detect_section("Abstract") == "abstract"
    assert _detect_section("Introduction") == "introduction"
    assert _detect_section("2. Methods and Materials") == "methods"
    assert _detect_section("3. Results") == "results"
    assert _detect_section("Discussion and Conclusion") == "discussion"
    assert _detect_section("Conclusion") == "discussion"
    assert _detect_section("Related Work") == "introduction"
    assert _detect_section("Random heading xyz") is None


def test_chunk_text_splits_long_text():
    long_text = "word " * 700
    chunks = _chunk_text(long_text, max_tokens=512, overlap_tokens=50)
    assert len(chunks) > 1
    for c in chunks:
        assert len(c.split()) <= 515  # small tolerance


def test_chunk_text_returns_single_chunk_for_short_text():
    short = "short text here"
    chunks = _chunk_text(short, max_tokens=512, overlap_tokens=50)
    assert chunks == [short]
