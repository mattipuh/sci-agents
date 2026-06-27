# Layer 1: Corpus

Parent context: see `../CLAUDE.md`

## Responsibility

This layer owns everything from raw paper → searchable index.
It exposes one clean interface upward: `retrieve(query, domain, top_k)`.

Nothing above this layer touches Qdrant, BM25, or PDFs directly.

---

## Files to build here

```
corpus/
├── CLAUDE.md
├── retriever.py        ← main interface (what agents import)
├── stub_retriever.py   ← fake retriever for DEMO_MODE=true
├── ingest.py           ← CLI: ingest a directory of PDFs (not needed for demo)
├── chunker.py          ← section-aware PDF chunking logic
├── embedder.py         ← wraps embedding model, batches calls
└── sources.py          ← open-access source configs (PMC, arXiv, OpenAlex)
```

---

## The retriever interface (build this first)

```python
# retriever.py
from dataclasses import dataclass

@dataclass
class PaperChunk:
    chunk_id: str
    paper_id: str          # DOI or arXiv ID
    title: str
    authors: list[str]
    year: int
    journal: str
    section: str           # "abstract" | "introduction" | "methods" | "results" | "discussion"
    text: str              # the actual chunk text
    score: float           # retrieval score (higher = more relevant)
    source_url: str        # link to full text (Unpaywall / PMC / arXiv)

class Retriever:
    def retrieve(
        self,
        query: str,
        domain: str,           # e.g. "mathematics", "industrial_management"
        top_k: int = 8,
        min_year: int | None = None,
    ) -> list[PaperChunk]:
        ...
```

Agents call `retriever.retrieve(query, domain)` and get back `list[PaperChunk]`.
They never call Qdrant or BM25 directly.

---

## Stub retriever (build this for demo)

`stub_retriever.py` returns hardcoded `PaperChunk` objects for each domain.
Load from `corpus/stubs/*.json` — one file per domain.

Format for stub JSON:
```json
[
  {
    "chunk_id": "stub-math-001",
    "paper_id": "arXiv:2301.00001",
    "title": "Stochastic job shop scheduling: a review",
    "authors": ["Smith, J.", "Aho, R."],
    "year": 2022,
    "journal": "European Journal of Operational Research",
    "section": "results",
    "text": "Our experiments across 14 benchmark instances show that...",
    "score": 0.91,
    "source_url": "https://arxiv.org/abs/2301.00001"
  }
]
```

Provide at least 5 stub chunks per domain for the demo domains:
- `mathematics` — stochastic optimisation, scheduling theory
- `industrial_management` — job shop scheduling, lean manufacturing
- `biology` — (reserve for future panel)
- `finance` — regulatory compliance, operational risk (Velvoite angle)

---

## Real retriever (skip for demo, document the design)

When `DEMO_MODE=false`, the real retriever:

1. **Vector search**: embed the query with the same model used at ingest,
   query Qdrant collection `domain_{domain_name}`, get top 20 by cosine similarity

2. **BM25 search**: tokenize query, search in-memory BM25 index for the same
   domain collection, get top 20 by BM25 score

3. **Hybrid fusion**: reciprocal rank fusion (RRF) to merge the two result lists

4. **Filter**: apply `min_year` filter if provided

5. **Return**: top `top_k` after fusion, as `list[PaperChunk]`

RRF formula: `score(d) = sum(1 / (k + rank_i(d)))` where k=60 (standard)

---

## Chunking strategy (for real ingest)

Do NOT use fixed-size character chunks. Use section-aware chunking:

1. Parse PDF with `pymupdf` or `pdfplumber`
2. Detect section headings (Abstract / Introduction / Methods / Results / Discussion)
3. Split each section into chunks of ~400 tokens with 50-token overlap
4. Tag each chunk with its `section` field — agents use this to weight evidence
   (results > abstract, methods > introduction for empirical claims)

Max chunk size: 512 tokens. Min chunk size: 100 tokens (discard smaller fragments).

---

## Domain taxonomy

Keep domain names as snake_case strings. Canonical list:

```python
DOMAINS = [
    "mathematics",
    "industrial_management",
    "biology",
    "chemistry",
    "physics",
    "computer_science",
    "finance",
    "engineering",
    "medicine",
    "economics",
]
```

A domain name maps 1:1 to a Qdrant collection name: `domain_mathematics` etc.

---

## Testing this layer

```bash
# With DEMO_MODE=true — should always work
python -c "
from corpus.stub_retriever import StubRetriever
r = StubRetriever()
chunks = r.retrieve('stochastic scheduling makespan', 'mathematics')
for c in chunks:
    print(c.title, c.score)
"
```

Expected: 3–5 chunks, all with `domain=mathematics`, sorted by score descending.
