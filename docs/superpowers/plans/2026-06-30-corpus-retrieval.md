# Corpus Retrieval — arXiv Full-Text Pipeline — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace live `web_search` with a local arXiv full-text pipeline — discover papers via arXiv API, download PDFs, chunk by section, embed with nomic-embed-text-v1.5, store in Qdrant, retrieve via BM25+vector RRF fusion with Semantic Scholar quality scoring.

**Architecture:** On first `retrieve(query, domain)` call, `ArxivRetriever` fetches candidate papers from arXiv, downloads PDFs, chunks them section-aware, embeds with a local sentence-transformers model, and upserts to Qdrant + BM25. Subsequent calls hit the local index directly (<500ms). A SQLite cache tracks indexed papers and their quality metadata. The existing `DomainAgent` accepts an optional `retriever` and injects retrieved chunks as a `[CONTEXT]` block instead of calling `web_search`.

**Tech Stack:** `httpx`, `pdfplumber`, `sentence-transformers` (nomic-embed-text-v1.5), `qdrant-client`, `rank-bm25`, `sqlite3` (stdlib), `pytest`

## Global Constraints

- Python 3.12 (matches Dockerfile `python:3.12-slim`)
- In-scope domains: `ai_ml_systems`, `mathematics`, `ops_research`
- `PaperChunk` and `Retriever` are defined in `corpus/retriever.py` — never redefine them
- Cache root: `CORPUS_CACHE_DIR` env var, default `./cache`
- Qdrant URL: `QDRANT_URL` env var, default `http://localhost:6333`
- Semantic Scholar quality lookup: 2s timeout, skip on failure (non-fatal)
- Web search fallback: only when retriever returns <3 chunks
- All tests go under `tests/` with `__init__.py` at each package level
- Run tests with: `pytest tests/ -v`

---

## File Map

**New files:**
```
corpus/
  paper_cache.py          — SQLite: tracks indexed paper_ids + quality metadata
  sources/__init__.py     — empty package marker
  sources/arxiv.py        — arXiv Atom API search → list[dict]
  fetcher.py              — download PDF to local cache, return path
  chunker.py              — PDF → list[PaperChunk] (section-aware)
  embedder.py             — nomic-embed-text-v1.5 wrapper (lazy import)
  vector_store.py         — Qdrant wrapper (upsert + search)
  bm25_index.py           — BM25 per domain with disk persistence
  arxiv_retriever.py      — implements Retriever, ties all components together

tests/
  __init__.py
  corpus/
    __init__.py
    test_paper_cache.py
    test_fetcher.py
    test_chunker.py
    test_embedder.py
    test_vector_store.py
    test_bm25_index.py
    test_arxiv_retriever.py
    sources/
      __init__.py
      test_arxiv.py
  agents/
    __init__.py
    test_base_agent_retriever.py
```

**Modified files:**
- `docker-compose.yml` — add Qdrant service + `qdrant_data` volume
- `Dockerfile` — pre-download embedding model
- `requirements.txt` — add new dependencies
- `agents/base_agent.py` — accept optional `retriever`, inject context block, fallback to web search

---

### Task 1: Infrastructure

**Files:**
- Modify: `docker-compose.yml`
- Modify: `Dockerfile`
- Modify: `requirements.txt`
- Create: `corpus/sources/__init__.py`
- Create: `tests/__init__.py`, `tests/corpus/__init__.py`, `tests/corpus/sources/__init__.py`, `tests/agents/__init__.py`

**Interfaces:**
- Produces: Qdrant running at `http://localhost:6333`; `nomic-embed-text-v1.5` cached in the container image; all new packages importable

- [ ] **Step 1: Add Qdrant service to docker-compose.yml**

Replace the entire contents of `docker-compose.yml`:
```yaml
services:
  sciagent:
    build: .
    ports:
      - "8080:8080"
    env_file:
      - .env
    environment:
      - LOG_LEVEL=INFO
    depends_on:
      - qdrant

  qdrant:
    image: qdrant/qdrant:latest
    ports:
      - "6333:6333"
    volumes:
      - qdrant_data:/qdrant/storage

volumes:
  qdrant_data:
```

- [ ] **Step 2: Add new Python dependencies to requirements.txt**

Append to `requirements.txt`:
```
# Corpus retrieval
httpx>=0.27.0
pdfplumber>=0.11.0
sentence-transformers>=3.0.0
qdrant-client>=1.9.0
rank-bm25>=0.2.2
defusedxml>=0.7.1
```

- [ ] **Step 3: Pre-download embedding model in Dockerfile**

Replace `Dockerfile` contents:
```dockerfile
FROM python:3.12-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('nomic-ai/nomic-embed-text-v1.5', trust_remote_code=True)"

COPY . .

EXPOSE 8080

CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8080"]
```

- [ ] **Step 4: Create package markers**

Create these empty files:
- `corpus/sources/__init__.py`
- `tests/__init__.py`
- `tests/corpus/__init__.py`
- `tests/corpus/sources/__init__.py`
- `tests/agents/__init__.py`

- [ ] **Step 5: Verify Qdrant starts (requires Docker)**

```bash
docker compose up qdrant -d
curl -s http://localhost:6333/healthz
```

Expected output contains `"title":"qdrant - vector search engine"`. (Skip this step if Docker is not available locally; CI will catch it.)

- [ ] **Step 6: Commit**

```bash
git add docker-compose.yml Dockerfile requirements.txt \
  corpus/sources/__init__.py \
  tests/__init__.py tests/corpus/__init__.py \
  tests/corpus/sources/__init__.py tests/agents/__init__.py
git commit -m "feat: add Qdrant, sentence-transformers, and corpus infrastructure"
```

---

### Task 2: PaperCache

**Files:**
- Create: `corpus/paper_cache.py`
- Create: `tests/corpus/test_paper_cache.py`

**Interfaces:**
- Consumes: nothing
- Produces:
  ```python
  class PaperCache:
      def __init__(self, db_path: str | None = None) -> None: ...
      def is_indexed(self, arxiv_id: str) -> bool: ...
      def mark_indexed(
          self, arxiv_id: str, domain: str, title: str,
          authors: list[str], year: int, abstract: str,
          source_url: str, publication_type: str, venue: str,
          citation_count: int, influential_citation_count: int,
      ) -> None: ...
      def get_quality(self, arxiv_id: str) -> dict | None:
          # returns {"publication_type": str, "venue": str,
          #          "citation_count": int, "influential_citation_count": int}
          # or None if not found
      def list_indexed(self) -> list[dict]:
          # returns [{"arxiv_id": str, "domain": str, "title": str, "year": int}, ...]
  ```

- [ ] **Step 1: Write failing tests**

Create `tests/corpus/test_paper_cache.py`:
```python
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
```

- [ ] **Step 2: Run to confirm failure**

```bash
pytest tests/corpus/test_paper_cache.py -v
```

Expected: `ModuleNotFoundError: No module named 'corpus.paper_cache'`

- [ ] **Step 3: Implement corpus/paper_cache.py**

```python
import json
import os
import sqlite3


class PaperCache:
    def __init__(self, db_path: str | None = None) -> None:
        if db_path is None:
            cache_dir = os.getenv("CORPUS_CACHE_DIR", "./cache")
            os.makedirs(cache_dir, exist_ok=True)
            db_path = os.path.join(cache_dir, "papers.db")
        self._db_path = db_path
        self._init_db()

    def _connect(self) -> sqlite3.Connection:
        return sqlite3.connect(self._db_path)

    def _init_db(self) -> None:
        with self._connect() as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS papers (
                    arxiv_id TEXT PRIMARY KEY,
                    domain TEXT NOT NULL,
                    title TEXT NOT NULL,
                    authors TEXT NOT NULL,
                    year INTEGER NOT NULL,
                    abstract TEXT NOT NULL,
                    source_url TEXT NOT NULL,
                    publication_type TEXT NOT NULL DEFAULT '',
                    venue TEXT NOT NULL DEFAULT '',
                    citation_count INTEGER NOT NULL DEFAULT 0,
                    influential_citation_count INTEGER NOT NULL DEFAULT 0,
                    indexed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)

    def is_indexed(self, arxiv_id: str) -> bool:
        with self._connect() as conn:
            return conn.execute(
                "SELECT 1 FROM papers WHERE arxiv_id = ?", (arxiv_id,)
            ).fetchone() is not None

    def mark_indexed(
        self,
        arxiv_id: str,
        domain: str,
        title: str,
        authors: list[str],
        year: int,
        abstract: str,
        source_url: str,
        publication_type: str,
        venue: str,
        citation_count: int,
        influential_citation_count: int,
    ) -> None:
        with self._connect() as conn:
            conn.execute(
                """INSERT OR REPLACE INTO papers
                   (arxiv_id, domain, title, authors, year, abstract, source_url,
                    publication_type, venue, citation_count, influential_citation_count)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (arxiv_id, domain, title, json.dumps(authors), year, abstract,
                 source_url, publication_type, venue, citation_count,
                 influential_citation_count),
            )

    def get_quality(self, arxiv_id: str) -> dict | None:
        with self._connect() as conn:
            row = conn.execute(
                """SELECT publication_type, venue, citation_count, influential_citation_count
                   FROM papers WHERE arxiv_id = ?""",
                (arxiv_id,),
            ).fetchone()
        if row is None:
            return None
        return {
            "publication_type": row[0],
            "venue": row[1],
            "citation_count": row[2],
            "influential_citation_count": row[3],
        }

    def list_indexed(self) -> list[dict]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT arxiv_id, domain, title, year FROM papers ORDER BY indexed_at DESC"
            ).fetchall()
        return [{"arxiv_id": r[0], "domain": r[1], "title": r[2], "year": r[3]}
                for r in rows]
```

- [ ] **Step 4: Run to confirm passing**

```bash
pytest tests/corpus/test_paper_cache.py -v
```

Expected: 6 tests PASS

- [ ] **Step 5: Commit**

```bash
git add corpus/paper_cache.py tests/corpus/test_paper_cache.py
git commit -m "feat: add PaperCache SQLite for tracking indexed arXiv papers"
```

---

### Task 3: arXiv Source

**Files:**
- Create: `corpus/sources/arxiv.py`
- Create: `tests/corpus/sources/test_arxiv.py`

**Interfaces:**
- Consumes: nothing
- Produces:
  ```python
  ARXIV_CATEGORIES: dict[str, list[str]]

  def search(query: str, domain: str, max_results: int = 10) -> list[dict]:
      # Each dict: {
      #   "arxiv_id": str,       # e.g. "1706.03762"
      #   "title": str,
      #   "authors": list[str],
      #   "year": int,
      #   "abstract": str,
      #   "pdf_url": str,        # direct PDF link
      #   "source_url": str,     # abstract page: https://arxiv.org/abs/{id}
      #   "comment": str,        # e.g. "Published in NeurIPS 2023"
      # }
  ```

- [ ] **Step 1: Write failing tests**

Create `tests/corpus/sources/test_arxiv.py`:
```python
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
    url = mock_get.call_args[0][0]
    assert "cs.AI" in url or "cs.LG" in url


def test_search_handles_empty_feed():
    with patch("httpx.get", return_value=_mock_response("<feed/>")):
        assert search("nothing", "ai_ml_systems") == []


def test_categories_cover_required_domains():
    for domain in ["ai_ml_systems", "mathematics", "ops_research"]:
        assert domain in ARXIV_CATEGORIES
        assert len(ARXIV_CATEGORIES[domain]) >= 1
```

- [ ] **Step 2: Run to confirm failure**

```bash
pytest tests/corpus/sources/test_arxiv.py -v
```

Expected: `ModuleNotFoundError: No module named 'corpus.sources.arxiv'`

- [ ] **Step 3: Implement corpus/sources/arxiv.py**

```python
import time
from urllib.parse import quote_plus

import defusedxml.ElementTree as ET
import httpx

ARXIV_CATEGORIES: dict[str, list[str]] = {
    "ai_ml_systems":       ["cs.AI", "cs.LG", "cs.CL", "cs.CV", "cs.SE"],
    "mathematics":         ["math.OC", "math.CO", "math.ST", "cs.GT"],
    "ops_research":        ["math.OC", "cs.GT", "econ.GN"],
    # Option C extension:
    "finance":             ["q-fin.RM", "q-fin.GN", "econ.GN"],
    "business_economics":  ["econ.GN", "econ.TH"],
    "strategy_game_theory":["cs.GT", "econ.TH"],
    "venture_finance":     ["q-fin.GN", "econ.GN"],
}

_NS = {"atom": "http://www.w3.org/2005/Atom",
       "arxiv": "http://arxiv.org/schemas/atom"}
_BASE = "https://export.arxiv.org/api/query"


def search(query: str, domain: str, max_results: int = 10) -> list[dict]:
    categories = ARXIV_CATEGORIES.get(domain, ["cs.AI"])
    cat_filter = " OR ".join(f"cat:{c}" for c in categories)
    q = f"({quote_plus(query)}) AND ({quote_plus(cat_filter)})"
    url = f"{_BASE}?search_query={q}&max_results={max_results}&sortBy=relevance"

    for attempt in range(3):
        try:
            resp = httpx.get(url, timeout=15.0)
            resp.raise_for_status()
            return _parse(resp.text)
        except Exception:
            if attempt == 2:
                return []
            time.sleep(2 ** attempt)
    return []


def _parse(xml_text: str) -> list[dict]:
    try:
        root = ET.fromstring(xml_text)
    except ET.ParseError:
        return []
    results = []
    for entry in root.findall("atom:entry", _NS):
        try:
            results.append(_parse_entry(entry))
        except Exception:
            continue
    return results


def _parse_entry(entry: ET.Element) -> dict:
    raw_id = entry.findtext("atom:id", "", _NS).rstrip("/").split("/")[-1]
    arxiv_id = raw_id.rsplit("v", 1)[0] if "v" in raw_id and raw_id.split("v")[-1].isdigit() else raw_id

    title = (entry.findtext("atom:title", "", _NS) or "").strip().replace("\n", " ")
    year = int((entry.findtext("atom:published", "2000", _NS) or "2000")[:4])
    abstract = (entry.findtext("atom:summary", "", _NS) or "").strip().replace("\n", " ")
    comment = (entry.findtext("arxiv:comment", "", _NS) or "").strip()
    authors = [
        (a.findtext("atom:name", "", _NS) or "").strip()
        for a in entry.findall("atom:author", _NS)
    ]
    pdf_url = next(
        (lnk.get("href", "") for lnk in entry.findall("atom:link", _NS)
         if lnk.get("type") == "application/pdf"),
        "",
    )
    return {
        "arxiv_id": arxiv_id,
        "title": title,
        "authors": authors,
        "year": year,
        "abstract": abstract,
        "pdf_url": pdf_url,
        "source_url": f"https://arxiv.org/abs/{arxiv_id}",
        "comment": comment,
    }
```

- [ ] **Step 4: Run to confirm passing**

```bash
pytest tests/corpus/sources/test_arxiv.py -v
```

Expected: 4 tests PASS

- [ ] **Step 5: Commit**

```bash
git add corpus/sources/arxiv.py tests/corpus/sources/test_arxiv.py
git commit -m "feat: add arXiv API source with domain category mapping"
```

---

### Task 4: PDF Fetcher

**Files:**
- Create: `corpus/fetcher.py`
- Create: `tests/corpus/test_fetcher.py`

**Interfaces:**
- Consumes: nothing
- Produces:
  ```python
  def fetch_pdf(arxiv_id: str, pdf_url: str,
                cache_dir: str | None = None) -> str | None:
      # Returns local path to saved PDF on success, None on failure.
      # Returns existing path immediately if already downloaded.
  ```

- [ ] **Step 1: Write failing tests**

Create `tests/corpus/test_fetcher.py`:
```python
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
```

- [ ] **Step 2: Run to confirm failure**

```bash
pytest tests/corpus/test_fetcher.py -v
```

Expected: `ModuleNotFoundError: No module named 'corpus.fetcher'`

- [ ] **Step 3: Implement corpus/fetcher.py**

```python
import logging
import os
import time

import httpx

log = logging.getLogger(__name__)


def fetch_pdf(arxiv_id: str, pdf_url: str,
              cache_dir: str | None = None) -> str | None:
    if cache_dir is None:
        cache_dir = os.getenv("CORPUS_CACHE_DIR", "./cache")
    pdf_dir = os.path.join(cache_dir, "pdfs")
    os.makedirs(pdf_dir, exist_ok=True)
    dest = os.path.join(pdf_dir, f"{arxiv_id}.pdf")
    if os.path.exists(dest):
        return dest
    for attempt in range(3):
        try:
            resp = httpx.get(pdf_url, timeout=30.0, follow_redirects=True)
            resp.raise_for_status()
            with open(dest, "wb") as f:
                f.write(resp.content)
            return dest
        except Exception as exc:
            log.warning("PDF download attempt %d failed for %s: %s",
                        attempt + 1, arxiv_id, exc)
            if attempt < 2:
                time.sleep(2 ** attempt)
    return None
```

- [ ] **Step 4: Run to confirm passing**

```bash
pytest tests/corpus/test_fetcher.py -v
```

Expected: 3 tests PASS

- [ ] **Step 5: Commit**

```bash
git add corpus/fetcher.py tests/corpus/test_fetcher.py
git commit -m "feat: add PDF fetcher with local cache and retry"
```

---

### Task 5: Section-Aware Chunker

**Files:**
- Create: `corpus/chunker.py`
- Create: `tests/corpus/test_chunker.py`

**Interfaces:**
- Consumes: `PaperChunk` from `corpus.retriever`
- Produces:
  ```python
  def chunk_pdf(
      pdf_path: str, arxiv_id: str, title: str,
      authors: list[str], year: int, source_url: str,
  ) -> list[PaperChunk]:
      # Returns PaperChunk list with section, text, chunk_id set.
      # score = 0.0 (retriever sets it). journal = "" (quality layer sets it).
      # Returns [] if PDF unreadable or yields no chunks above MIN_TOKENS.

  # Also exported for testing:
  def _detect_section(heading: str) -> str | None: ...
  def _chunk_text(text: str, max_tokens: int, overlap_tokens: int) -> list[str]: ...
  ```

- [ ] **Step 1: Write failing tests**

Create `tests/corpus/test_chunker.py`:
```python
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
```

- [ ] **Step 2: Run to confirm failure**

```bash
pytest tests/corpus/test_chunker.py -v
```

Expected: `ModuleNotFoundError: No module named 'corpus.chunker'`

- [ ] **Step 3: Implement corpus/chunker.py**

```python
import logging
import re

import pdfplumber

from corpus.retriever import PaperChunk

log = logging.getLogger(__name__)

_MAX_TOKENS = 512
_OVERLAP_TOKENS = 50
_MIN_TOKENS = 100

_SECTION_PATTERNS = [
    (re.compile(r"^\s*abstract\s*$", re.I),           "abstract"),
    (re.compile(r"^\s*\d*\.?\s*introduction\b", re.I), "introduction"),
    (re.compile(r"^\s*\d*\.?\s*related work", re.I),  "introduction"),
    (re.compile(r"^\s*\d*\.?\s*method",        re.I), "methods"),
    (re.compile(r"^\s*\d*\.?\s*result",        re.I), "results"),
    (re.compile(r"^\s*\d*\.?\s*discussion",    re.I), "discussion"),
    (re.compile(r"^\s*\d*\.?\s*conclusion",    re.I), "discussion"),
]


def chunk_pdf(
    pdf_path: str,
    arxiv_id: str,
    title: str,
    authors: list[str],
    year: int,
    source_url: str,
) -> list[PaperChunk]:
    try:
        full_text = _extract_text(pdf_path)
    except Exception as exc:
        log.warning("Failed to parse PDF %s: %s", arxiv_id, exc)
        return []

    if not full_text.strip():
        return []

    chunks: list[PaperChunk] = []
    idx = 0
    for section_name, section_text in _split_sections(full_text):
        for text in _chunk_text(section_text, _MAX_TOKENS, _OVERLAP_TOKENS):
            if len(text.split()) < _MIN_TOKENS:
                continue
            chunks.append(PaperChunk(
                chunk_id=f"{arxiv_id}-{section_name}-{idx}",
                paper_id=arxiv_id,
                title=title,
                authors=authors,
                year=year,
                journal="",
                section=section_name,
                text=text.strip(),
                score=0.0,
                source_url=source_url,
            ))
            idx += 1
    return chunks


def _extract_text(pdf_path: str) -> str:
    with pdfplumber.open(pdf_path) as pdf:
        return "\n".join(page.extract_text() or "" for page in pdf.pages)


def _split_sections(text: str) -> list[tuple[str, str]]:
    sections: list[tuple[str, str]] = []
    current_section = "body"
    current_lines: list[str] = []
    for line in text.split("\n"):
        detected = _detect_section(line.strip())
        if detected and len(line.strip()) < 80:
            if current_lines:
                sections.append((current_section, "\n".join(current_lines)))
            current_section = detected
            current_lines = []
        else:
            current_lines.append(line)
    if current_lines:
        sections.append((current_section, "\n".join(current_lines)))
    return sections


def _detect_section(heading: str) -> str | None:
    for pattern, label in _SECTION_PATTERNS:
        if pattern.match(heading):
            return label
    return None


def _chunk_text(text: str, max_tokens: int, overlap_tokens: int) -> list[str]:
    words = text.split()
    if not words:
        return []
    if len(words) <= max_tokens:
        return [text]
    step = max_tokens - overlap_tokens
    return [
        " ".join(words[start: start + max_tokens])
        for start in range(0, len(words), step)
        if start < len(words)
    ]
```

- [ ] **Step 4: Run to confirm passing**

```bash
pytest tests/corpus/test_chunker.py -v
```

Expected: 7 tests PASS

- [ ] **Step 5: Commit**

```bash
git add corpus/chunker.py tests/corpus/test_chunker.py
git commit -m "feat: add section-aware PDF chunker"
```

---

### Task 6: Embedder

**Files:**
- Create: `corpus/embedder.py`
- Create: `tests/corpus/test_embedder.py`

**Interfaces:**
- Consumes: nothing
- Produces:
  ```python
  def embed_query(text: str) -> list[float]:
      # Prefixes "search_query: ", returns 768-dim vector

  def embed_documents(texts: list[str]) -> list[list[float]]:
      # Prefixes each with "search_document: ", returns list of 768-dim vectors
  ```

- [ ] **Step 1: Write failing tests**

Create `tests/corpus/test_embedder.py`:
```python
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
```

- [ ] **Step 2: Run to confirm failure**

```bash
pytest tests/corpus/test_embedder.py -v
```

Expected: `ModuleNotFoundError: No module named 'corpus.embedder'`

- [ ] **Step 3: Implement corpus/embedder.py**

```python
from __future__ import annotations

_MODEL_NAME = "nomic-ai/nomic-embed-text-v1.5"
_model = None


def _get_model():
    global _model
    if _model is None:
        from sentence_transformers import SentenceTransformer
        _model = SentenceTransformer(_MODEL_NAME, trust_remote_code=True)
    return _model


def embed_query(text: str) -> list[float]:
    vecs = _get_model().encode(
        [f"search_query: {text}"], normalize_embeddings=True
    )
    return vecs[0].tolist()


def embed_documents(texts: list[str]) -> list[list[float]]:
    if not texts:
        return []
    prefixed = [f"search_document: {t}" for t in texts]
    vecs = _get_model().encode(prefixed, normalize_embeddings=True, batch_size=32)
    return [v.tolist() for v in vecs]
```

- [ ] **Step 4: Run to confirm passing**

```bash
pytest tests/corpus/test_embedder.py -v
```

Expected: 5 tests PASS

- [ ] **Step 5: Commit**

```bash
git add corpus/embedder.py tests/corpus/test_embedder.py
git commit -m "feat: add embedder using nomic-embed-text-v1.5 with lazy model load"
```

---

### Task 7: Vector Store

**Files:**
- Create: `corpus/vector_store.py`
- Create: `tests/corpus/test_vector_store.py`

**Interfaces:**
- Consumes: `PaperChunk` from `corpus.retriever`
- Produces:
  ```python
  class VectorStore:
      def __init__(self, url: str | None = None) -> None: ...
      def upsert(self, domain: str, chunks: list[PaperChunk],
                 embeddings: list[list[float]]) -> None: ...
      def search(self, domain: str, query_embedding: list[float],
                 top_k: int) -> list[tuple[PaperChunk, float]]:
          # Returns (chunk, cosine_score) sorted by score descending.
          # chunk.score is set to the cosine score.
  ```

- [ ] **Step 1: Write failing tests**

Create `tests/corpus/test_vector_store.py`:
```python
from unittest.mock import MagicMock, patch

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


import pytest


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
```

- [ ] **Step 2: Run to confirm failure**

```bash
pytest tests/corpus/test_vector_store.py -v
```

Expected: `ModuleNotFoundError: No module named 'corpus.vector_store'`

- [ ] **Step 3: Implement corpus/vector_store.py**

```python
import logging
import os
import uuid

from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams

from corpus.retriever import PaperChunk

log = logging.getLogger(__name__)
_DIM = 768


def _col(domain: str) -> str:
    return f"domain_{domain}"


class VectorStore:
    def __init__(self, url: str | None = None) -> None:
        url = url or os.getenv("QDRANT_URL", "http://localhost:6333")
        self._client = QdrantClient(url=url)
        self._ready: set[str] = set()

    def _ensure(self, domain: str) -> None:
        name = _col(domain)
        if name in self._ready:
            return
        existing = {c.name for c in self._client.get_collections().collections}
        if name not in existing:
            self._client.create_collection(
                collection_name=name,
                vectors_config=VectorParams(size=_DIM, distance=Distance.COSINE),
            )
        self._ready.add(name)

    def upsert(self, domain: str, chunks: list[PaperChunk],
               embeddings: list[list[float]]) -> None:
        self._ensure(domain)
        points = [
            PointStruct(
                id=str(uuid.uuid4()),
                vector=emb,
                payload={
                    "chunk_id": c.chunk_id, "paper_id": c.paper_id,
                    "title": c.title, "authors": c.authors, "year": c.year,
                    "journal": c.journal, "section": c.section,
                    "text": c.text, "source_url": c.source_url,
                },
            )
            for c, emb in zip(chunks, embeddings)
        ]
        self._client.upsert(collection_name=_col(domain), points=points)

    def search(self, domain: str, query_embedding: list[float],
               top_k: int) -> list[tuple[PaperChunk, float]]:
        self._ensure(domain)
        hits = self._client.search(
            collection_name=_col(domain),
            query_vector=query_embedding,
            limit=top_k,
        )
        results = []
        for hit in hits:
            p = hit.payload
            chunk = PaperChunk(
                chunk_id=p["chunk_id"], paper_id=p["paper_id"],
                title=p["title"], authors=p["authors"], year=p["year"],
                journal=p["journal"], section=p["section"],
                text=p["text"], score=hit.score, source_url=p["source_url"],
            )
            results.append((chunk, hit.score))
        return results
```

- [ ] **Step 4: Run to confirm passing**

```bash
pytest tests/corpus/test_vector_store.py -v
```

Expected: 3 tests PASS

- [ ] **Step 5: Commit**

```bash
git add corpus/vector_store.py tests/corpus/test_vector_store.py
git commit -m "feat: add Qdrant vector store wrapper"
```

---

### Task 8: BM25 Index

**Files:**
- Create: `corpus/bm25_index.py`
- Create: `tests/corpus/test_bm25_index.py`

**Interfaces:**
- Consumes: `PaperChunk` from `corpus.retriever`
- Produces:
  ```python
  class BM25Index:
      def __init__(self, cache_dir: str | None = None) -> None: ...
      def add_chunks(self, domain: str, chunks: list[PaperChunk]) -> None: ...
      def search(self, domain: str, query: str,
                 top_k: int) -> list[tuple[PaperChunk, float]]:
          # Returns (chunk, bm25_score) sorted descending. [] if domain unknown.
      def save(self, domain: str) -> None: ...
      def load(self, domain: str) -> bool:
          # True if loaded from disk, False if not found.
  ```

- [ ] **Step 1: Write failing tests**

Create `tests/corpus/test_bm25_index.py`:
```python
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
```

- [ ] **Step 2: Run to confirm failure**

```bash
pytest tests/corpus/test_bm25_index.py -v
```

Expected: `ModuleNotFoundError: No module named 'corpus.bm25_index'`

- [ ] **Step 3: Implement corpus/bm25_index.py**

```python
import logging
import os
import pickle

from rank_bm25 import BM25Okapi

from corpus.retriever import PaperChunk

log = logging.getLogger(__name__)


class BM25Index:
    def __init__(self, cache_dir: str | None = None) -> None:
        self._dir = cache_dir or os.getenv("CORPUS_CACHE_DIR", "./cache")
        self._indexes: dict[str, BM25Okapi] = {}
        self._chunks: dict[str, list[PaperChunk]] = {}

    def add_chunks(self, domain: str, chunks: list[PaperChunk]) -> None:
        all_chunks = self._chunks.get(domain, []) + chunks
        self._chunks[domain] = all_chunks
        self._indexes[domain] = BM25Okapi([c.text.lower().split() for c in all_chunks])

    def search(self, domain: str, query: str,
               top_k: int) -> list[tuple[PaperChunk, float]]:
        if domain not in self._indexes:
            return []
        scores = self._indexes[domain].get_scores(query.lower().split())
        ranked = sorted(zip(self._chunks[domain], scores),
                        key=lambda x: x[1], reverse=True)
        return [(c, float(s)) for c, s in ranked[:top_k]]

    def save(self, domain: str) -> None:
        os.makedirs(self._dir, exist_ok=True)
        path = os.path.join(self._dir, f"bm25_{domain}.pkl")
        with open(path, "wb") as f:
            pickle.dump({"index": self._indexes.get(domain),
                         "chunks": self._chunks.get(domain, [])}, f)

    def load(self, domain: str) -> bool:
        path = os.path.join(self._dir, f"bm25_{domain}.pkl")
        if not os.path.exists(path):
            return False
        try:
            with open(path, "rb") as f:
                data = pickle.load(f)
            self._indexes[domain] = data["index"]
            self._chunks[domain] = data["chunks"]
            return True
        except Exception as exc:
            log.warning("Failed to load BM25 index for %s: %s", domain, exc)
            return False
```

- [ ] **Step 4: Run to confirm passing**

```bash
pytest tests/corpus/test_bm25_index.py -v
```

Expected: 6 tests PASS

- [ ] **Step 5: Commit**

```bash
git add corpus/bm25_index.py tests/corpus/test_bm25_index.py
git commit -m "feat: add per-domain BM25 index with disk persistence"
```

---

### Task 9: ArxivRetriever

**Files:**
- Create: `corpus/arxiv_retriever.py`
- Create: `tests/corpus/test_arxiv_retriever.py`

**Interfaces:**
- Consumes: `PaperCache`, `search` (arxiv), `fetch_pdf`, `chunk_pdf`, `embed_query`, `embed_documents`, `VectorStore`, `BM25Index`, `Retriever`, `PaperChunk`
- Produces:
  ```python
  class ArxivRetriever(Retriever):
      def retrieve(self, query: str, domain: str,
                   top_k: int = 8, min_year: int | None = None) -> list[PaperChunk]:
          # Fetches + indexes new papers, then runs hybrid RRF retrieval.
          # Chunks are sorted by final_score = rrf_score × pub_mult × citation_boost.
  ```

- [ ] **Step 1: Write failing tests**

Create `tests/corpus/test_arxiv_retriever.py`:
```python
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
```

- [ ] **Step 2: Run to confirm failure**

```bash
pytest tests/corpus/test_arxiv_retriever.py -v
```

Expected: `ModuleNotFoundError: No module named 'corpus.arxiv_retriever'`

- [ ] **Step 3: Implement corpus/arxiv_retriever.py**

```python
import logging
import math

import httpx

from corpus.bm25_index import BM25Index
from corpus.chunker import chunk_pdf
from corpus.embedder import embed_documents, embed_query
from corpus.fetcher import fetch_pdf
from corpus.paper_cache import PaperCache
from corpus.retriever import PaperChunk, Retriever
from corpus.sources.arxiv import search as arxiv_search
from corpus.vector_store import VectorStore

log = logging.getLogger(__name__)

_TOP_VENUES: dict[str, list[str]] = {
    "ai_ml_systems": ["NeurIPS", "ICML", "ICLR", "CVPR", "ACL", "EMNLP", "ICCV"],
    "mathematics":   ["Annals of Mathematics", "Journal of the ACM", "SIAM", "Inventiones"],
    "ops_research":  ["Operations Research", "Management Science", "EJOR",
                      "Mathematical Programming"],
}


class ArxivRetriever(Retriever):
    def __init__(self) -> None:
        self._cache = PaperCache()
        self._vector_store = VectorStore()
        self._bm25 = BM25Index()
        for domain in ["ai_ml_systems", "mathematics", "ops_research"]:
            self._bm25.load(domain)

    def retrieve(
        self, query: str, domain: str,
        top_k: int = 8, min_year: int | None = None,
    ) -> list[PaperChunk]:
        candidates = arxiv_search(query, domain, max_results=10)
        if min_year:
            candidates = [p for p in candidates if p["year"] >= min_year]

        for paper in candidates:
            if not self._cache.is_indexed(paper["arxiv_id"]):
                self._ingest(paper, domain)

        query_vec = embed_query(query)
        vector_hits = self._vector_store.search(domain, query_vec, top_k=20)
        bm25_hits = self._bm25.search(domain, query, top_k=20)

        return self._rrf_fuse(vector_hits, bm25_hits, domain)[:top_k]

    def _ingest(self, paper: dict, domain: str) -> None:
        arxiv_id = paper["arxiv_id"]
        pdf_path = fetch_pdf(arxiv_id, paper["pdf_url"])
        if pdf_path:
            chunks = chunk_pdf(
                pdf_path=pdf_path, arxiv_id=arxiv_id,
                title=paper["title"], authors=paper["authors"],
                year=paper["year"], source_url=paper["source_url"],
            )
        else:
            chunks = []

        if chunks:
            embeddings = embed_documents([c.text for c in chunks])
            self._vector_store.upsert(domain, chunks, embeddings)
            self._bm25.add_chunks(domain, chunks)
            self._bm25.save(domain)
        else:
            log.warning("No chunks for %s — storing abstract only", arxiv_id)

        quality = _fetch_quality_metadata(arxiv_id)
        self._cache.mark_indexed(
            arxiv_id=arxiv_id, domain=domain,
            title=paper["title"], authors=paper["authors"],
            year=paper["year"], abstract=paper["abstract"],
            source_url=paper["source_url"],
            publication_type=quality.get("publication_type", "") if quality else "",
            venue=quality.get("venue", "") if quality else "",
            citation_count=quality.get("citation_count", 0) if quality else 0,
            influential_citation_count=quality.get("influential_citation_count", 0)
            if quality else 0,
        )

    def _rrf_fuse(
        self,
        vector_hits: list[tuple[PaperChunk, float]],
        bm25_hits: list[tuple[PaperChunk, float]],
        domain: str,
    ) -> list[PaperChunk]:
        K = 60
        rrf: dict[str, float] = {}
        by_id: dict[str, PaperChunk] = {}

        for rank, (chunk, _) in enumerate(vector_hits):
            rrf[chunk.chunk_id] = rrf.get(chunk.chunk_id, 0.0) + 1.0 / (K + rank + 1)
            by_id[chunk.chunk_id] = chunk

        for rank, (chunk, _) in enumerate(bm25_hits):
            rrf[chunk.chunk_id] = rrf.get(chunk.chunk_id, 0.0) + 1.0 / (K + rank + 1)
            by_id[chunk.chunk_id] = chunk

        top_venues = _TOP_VENUES.get(domain, [])
        result: list[PaperChunk] = []
        for cid, base_score in sorted(rrf.items(), key=lambda x: x[1], reverse=True):
            chunk = by_id[cid]
            quality = self._cache.get_quality(chunk.paper_id)
            final = base_score * _quality_multiplier(quality, top_venues)
            result.append(PaperChunk(
                chunk_id=chunk.chunk_id, paper_id=chunk.paper_id,
                title=chunk.title, authors=chunk.authors, year=chunk.year,
                journal=chunk.journal, section=chunk.section,
                text=chunk.text, score=final, source_url=chunk.source_url,
            ))
        return result


def _quality_multiplier(quality: dict | None, top_venues: list[str]) -> float:
    if quality is None:
        return 1.0
    pub_type = quality.get("publication_type", "")
    venue = quality.get("venue", "")
    influential = quality.get("influential_citation_count", 0)

    if pub_type == "JournalArticle":
        pub_mult = 1.3
    elif any(v.lower() in venue.lower() for v in top_venues if v):
        pub_mult = 1.2
    else:
        pub_mult = 1.0

    citation_boost = min(1.0 + math.log(1 + influential) / 7.0, 1.5)
    return pub_mult * citation_boost


def _fetch_quality_metadata(arxiv_id: str) -> dict | None:
    url = (
        f"https://api.semanticscholar.org/graph/v1/paper/arXiv:{arxiv_id}"
        "?fields=citationCount,influentialCitationCount,publicationTypes,venue"
    )
    try:
        resp = httpx.get(url, timeout=2.0)
        resp.raise_for_status()
        data = resp.json()
        types = data.get("publicationTypes") or []
        return {
            "publication_type": types[0] if types else "Preprint",
            "venue": data.get("venue") or "",
            "citation_count": data.get("citationCount") or 0,
            "influential_citation_count": data.get("influentialCitationCount") or 0,
        }
    except Exception:
        return None
```

- [ ] **Step 4: Run to confirm passing**

```bash
pytest tests/corpus/test_arxiv_retriever.py -v
```

Expected: 4 tests PASS

- [ ] **Step 5: Run all corpus tests**

```bash
pytest tests/corpus/ -v
```

Expected: all tests PASS

- [ ] **Step 6: Commit**

```bash
git add corpus/arxiv_retriever.py tests/corpus/test_arxiv_retriever.py
git commit -m "feat: add ArxivRetriever with hybrid RRF retrieval and quality scoring"
```

---

### Task 10: Agent Integration

**Files:**
- Modify: `agents/base_agent.py`
- Create: `tests/agents/test_base_agent_retriever.py`

**Interfaces:**
- Consumes: `ArxivRetriever` / `Retriever` from corpus layer
- Produces: `DomainAgent.__init__` accepts `retriever: Retriever | None = None`; when ≥3 chunks retrieved, uses `[CONTEXT]` block and omits `tools`; when <3 chunks, falls back to `web_search`

- [ ] **Step 1: Write failing tests**

Create `tests/agents/test_base_agent_retriever.py`:
```python
from unittest.mock import MagicMock, patch

from corpus.retriever import PaperChunk


def _chunk(n=0):
    return PaperChunk(
        chunk_id=f"1706.03762-results-{n}", paper_id="1706.03762",
        title="Attention Is All You Need", authors=["Vaswani, A."], year=2017,
        journal="NeurIPS 2017", section="results",
        text="Our model achieves 28.4 BLEU on English-German translation.",
        score=0.91, source_url="https://arxiv.org/abs/1706.03762",
    )


def _mock_llm_response(text="test response"):
    r = MagicMock()
    r.content = [MagicMock(text=(
        f'{{"content": "{text}", "citations": [], "confidence": 0.8, "dissent": null}}'
    ))]
    r.stop_reason = "end_turn"
    r.usage = MagicMock(input_tokens=100, output_tokens=50)
    return r


def _make_agent(retriever=None):
    with patch.object(
        __import__("agents.base_agent", fromlist=["DomainAgent"]).DomainAgent,
        "_load_persona", return_value="test persona"
    ):
        from agents.base_agent import DomainAgent
        client = MagicMock()
        client.messages.create.return_value = _mock_llm_response()
        return DomainAgent("ai_ml_systems", client, retriever=retriever), client


def test_agent_injects_context_block_when_retriever_returns_chunks():
    mock_retriever = MagicMock()
    mock_retriever.retrieve.return_value = [_chunk(i) for i in range(3)]

    agent, client = _make_agent(retriever=mock_retriever)
    agent.respond("What is attention?", [])

    message_content = client.messages.create.call_args[1]["messages"][0]["content"]
    assert "[CONTEXT" in message_content
    assert "Attention Is All You Need" in message_content


def test_agent_omits_tools_when_retriever_returns_enough_chunks():
    mock_retriever = MagicMock()
    mock_retriever.retrieve.return_value = [_chunk(i) for i in range(3)]

    agent, client = _make_agent(retriever=mock_retriever)
    agent.respond("What is attention?", [])

    call_kwargs = client.messages.create.call_args[1]
    assert "tools" not in call_kwargs


def test_agent_falls_back_to_web_search_when_fewer_than_3_chunks():
    mock_retriever = MagicMock()
    mock_retriever.retrieve.return_value = [_chunk()]  # only 1

    agent, client = _make_agent(retriever=mock_retriever)
    agent.respond("What is attention?", [])

    call_kwargs = client.messages.create.call_args[1]
    tools = call_kwargs.get("tools", [])
    assert any(t.get("type") == "web_search_20250305" for t in tools)


def test_agent_uses_web_search_when_no_retriever():
    agent, client = _make_agent(retriever=None)
    agent.respond("What is attention?", [])

    call_kwargs = client.messages.create.call_args[1]
    tools = call_kwargs.get("tools", [])
    assert any(t.get("type") == "web_search_20250305" for t in tools)
```

- [ ] **Step 2: Run to confirm failure**

```bash
pytest tests/agents/test_base_agent_retriever.py -v
```

Expected: `TypeError: DomainAgent.__init__() got an unexpected keyword argument 'retriever'`

- [ ] **Step 3a: Add import and retriever parameter to DomainAgent.__init__**

In `agents/base_agent.py`, add to imports:
```python
from corpus.retriever import Retriever
```

Change `DomainAgent.__init__` signature (line 70):
```python
def __init__(self, domain: str, client: anthropic.Anthropic,
             model: str = "claude-sonnet-4-6",
             retriever: Retriever | None = None):
```

Add at the end of `__init__` body:
```python
        self._retriever = retriever
```

- [ ] **Step 3b: Add `_format_chunks` helper at module level**

Add before the `DomainAgent` class definition:
```python
def _format_chunks(chunks: list) -> str:
    lines = ["[CONTEXT — retrieved literature]\n"]
    for i, chunk in enumerate(chunks, 1):
        authors_str = ", ".join(chunk.authors[:2])
        if len(chunk.authors) > 2:
            authors_str += " et al."
        lines.append(
            f"[{i}] {authors_str} ({chunk.year}) — \"{chunk.title}\"\n"
            f"Source: {chunk.source_url} | Section: {chunk.section}"
            + (f" | {chunk.journal}" if chunk.journal else "")
            + f"\n\"{chunk.text}\"\n"
        )
    return "\n".join(lines)
```

- [ ] **Step 3c: Replace retrieval logic in `respond()`**

In `respond()`, find the line that builds `user_message` (currently starting with `scope = DOMAIN_SEARCH_SCOPES...`). Replace the entire block from `scope =` down to and including the `response = self.client.messages.create(...)` call with:

```python
        # Try retriever first; fall back to web_search if <3 chunks
        context_block = ""
        use_web_search = True
        if self._retriever is not None:
            try:
                chunks = self._retriever.retrieve(
                    query=question, domain=self.domain, top_k=8
                )
                if len(chunks) >= 3:
                    context_block = _format_chunks(chunks)
                    use_web_search = False
            except Exception as exc:
                log.warning("Retriever failed for %s, falling back to web search: %s",
                            self.domain, exc)

        scope = DOMAIN_SEARCH_SCOPES.get(self.domain, "site:arxiv.org")

        json_schema = """{
  "content": "Your substantive response (200-400 words, flowing paragraphs, inline [Author, Year] citations)",
  "citations": [
    {"title": "Paper title", "year": 2023, "url": "https://...", "doi": "10.xxx/xxx"}
  ],
  "confidence": 0.8,
  "dissent": "If you disagree with a prior agent, state it here. Otherwise null."
}"""

        if context_block:
            user_message = f"""Research question: {question}
{prior_context}{company_context_str}{docs_context}

{context_block}

Provide your expert response grounded in the retrieved literature above.

Your response MUST follow this exact JSON structure:
{json_schema}

Provide ONLY the JSON object, no other text."""

            response = self.client.messages.create(
                model=self.model,
                max_tokens=4096,
                system=self._persona,
                messages=[{"role": "user", "content": user_message}],
            )
        else:
            user_message = f"""Research question: {question}
{prior_context}{company_context_str}{docs_context}

Search for relevant literature using the web search tool (scoped to {self.domain} literature: {scope}).
Then provide your expert response.

Your response MUST follow this exact JSON structure:
{json_schema}

Provide ONLY the JSON object, no other text."""

            response = self.client.messages.create(
                model=self.model,
                max_tokens=4096,
                system=self._persona,
                tools=[{"type": "web_search_20250305", "name": "web_search", "max_uses": 3}],
                messages=[{"role": "user", "content": user_message}],
            )
```

Leave the rest of `respond()` (usage extraction, `_extract_json`, `return AgentTurn`) unchanged.

- [ ] **Step 4: Run to confirm passing**

```bash
pytest tests/agents/test_base_agent_retriever.py -v
```

Expected: 4 tests PASS

- [ ] **Step 5: Run full test suite**

```bash
pytest tests/ -v
```

Expected: all tests PASS

- [ ] **Step 6: Commit**

```bash
git add agents/base_agent.py tests/agents/test_base_agent_retriever.py
git commit -m "feat: integrate ArxivRetriever into DomainAgent with web search fallback"
```

---

## Self-Review

Checking spec sections against plan tasks:

- ✅ arXiv API search with category mapping → Task 3
- ✅ PDF download + local cache → Task 4
- ✅ Section-aware chunking (Abstract/Intro/Methods/Results/Discussion) → Task 5
- ✅ nomic-embed-text-v1.5 with search_query/search_document prefixes → Task 6
- ✅ Qdrant vector store, one collection per domain → Task 7
- ✅ BM25 index per domain with disk persistence → Task 8
- ✅ RRF fusion (k=60) → Task 9
- ✅ Semantic Scholar quality lookup (2s timeout, non-fatal) → Task 9
- ✅ Quality multiplier = pub_mult × citation_boost → Task 9
- ✅ min_year filter → Task 9
- ✅ Web search fallback when <3 chunks → Task 10
- ✅ [CONTEXT] block format with section + source URL → Task 10
- ✅ Qdrant in docker-compose + embedding model pre-download → Task 1
- ✅ Exponential backoff on arXiv API errors → Task 3
- ✅ PDF download failure → empty chunks, still mark indexed → Task 9
- ✅ SQLite tracks indexed papers + quality fields → Task 2
- ✅ BM25 load at startup per domain → Task 9 (`__init__`)
- ✅ CORPUS_CACHE_DIR and QDRANT_URL env vars → Tasks 2, 4, 7, 8
