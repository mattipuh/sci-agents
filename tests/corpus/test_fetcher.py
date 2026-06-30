from pathlib import Path
from unittest.mock import MagicMock, patch

from corpus.fetcher import fetch_pdf


def _mock_resp(content: bytes):
    r = MagicMock()
    r.content = content
    r.raise_for_status = MagicMock()
    return r


def test_fetch_saves_and_returns_path(tmp_path):
    fake_pdf = b"%PDF-1.4 fake"
    with patch("httpx.get", return_value=_mock_resp(fake_pdf)):
        result = fetch_pdf("2301.00001", "https://arxiv.org/pdf/2301.00001",
                           str(tmp_path))
    assert result is not None
    assert Path(result).read_bytes() == fake_pdf
    assert "2301.00001" in Path(result).name


def test_fetch_returns_none_on_failure(tmp_path):
    with patch("httpx.get", side_effect=Exception("timeout")):
        result = fetch_pdf("2301.00001", "https://arxiv.org/pdf/2301.00001",
                           str(tmp_path))
    assert result is None


def test_fetch_skips_download_if_already_cached(tmp_path):
    pdf_dir = tmp_path / "pdfs"
    pdf_dir.mkdir()
    existing = pdf_dir / "2301.00001.pdf"
    existing.write_bytes(b"cached")
    with patch("httpx.get") as mock_get:
        result = fetch_pdf("2301.00001", "https://arxiv.org/pdf/2301.00001",
                           str(tmp_path))
    mock_get.assert_not_called()
    assert result == str(existing)
