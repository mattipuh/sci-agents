# Corpus Retrieval Design — arXiv Full-Text Pipeline (Option B → C)

**Date:** 2026-06-30  
**Status:** Approved  
**Scope:** Replace live web_search with pre-retrieved full-text chunks for `ai_ml_systems`, `mathematics`, `ops_research`. Extension path to Option C (Semantic Scholar + Unpaywall) documented but not implemented in this phase.

---

## Problem

The current `DomainAgent` uses Anthropic's `web_search_20250305` tool at generation time. This means:
- Agents spend 3 tool calls searching instead of reasoning
- Retrieved content is snippets, not full paper sections
- Citations are non-deterministic and sometimes hallucinated
- Every panel run pays web search token overhead

The goal is research quality: agents reason over actual paper sections (Methods, Results, Discussion), not abstracts or search snippets.

---

## Architecture

### Component map

```
corpus/
├── retriever.py              ← existing Retriever interface (unchanged)
├── arxiv_retriever.py        ← implements Retriever using the pipeline below
├── sources/
│   ├── arxiv.py              ← arXiv API: search by query+domain → paper metadata
│   ├── semantic_scholar.py   ← (C, future) richer discovery + broader domains
│   └── unpaywall.py          ← (C, future) open-access PDF URL resolver
├── fetcher.py                ← downloads PDF bytes from URL
├── chunker.py                ← PDF → section-aware PaperChunk list
├── embedder.py               ← local sentence-transformers model, batch embed
├── vector_store.py           ← Qdrant wrapper (upsert, search by cosine similarity)
├── bm25_index.py             ← in-memory BM25 per domain, persisted to disk
└── paper_cache.py            ← SQLite: tracks indexed paper_ids + quality metadata
```

The `Retriever` interface (`retriever.py`) is unchanged. `DomainAgent` imports `ArxivRetriever` instead of using web search.

---

## Data Flow

### On-demand fetch + cache

```
retrieve(query, domain, top_k=8)
│
├─ 1. sources/arxiv.py — search arXiv API
│     map domain → arXiv categories (see below)
│     returns: [{arxiv_id, title, authors, year, abstract, pdf_url, comment}]
│     fetch top 10 candidates
│
├─ 2. paper_cache.py — check SQLite
│     already indexed → skip to step 6
│     new paper → continue
│
├─ 3. fetcher.py — download PDF
│     GET https://arxiv.org/pdf/{arxiv_id}.pdf
│     store at cache/pdfs/{arxiv_id}.pdf
│
├─ 4. chunker.py — section-aware chunking
│     pdfplumber parses PDF
│     detect headings: Abstract / Introduction / Methods / Results / Discussion
│     chunk each section: ~400 tokens, 50-token overlap
│     min chunk: 100 tokens (discard smaller fragments)
│     max chunk: 512 tokens
│     tag each chunk with section field
│
├─ 5. embedder.py + vector_store.py — index (synchronous, on critical path)
│     embed chunks with nomic-embed-text-v1.5 (local, batch)
│     upsert into Qdrant collection domain_{domain}
│     fetch quality metadata from Semantic Scholar paper lookup API
│       → 2s timeout; skip quality multiplier on timeout/error (non-fatal)
│     store arxiv_id as indexed in SQLite with quality fields
│
└─ 6. hybrid retrieval
      vector search: embed query → Qdrant top 20 by cosine similarity
      BM25 search: tokenize query → in-memory index top 20
      RRF fusion: score(chunk) = Σ 1/(60 + rank_i) × quality_multiplier
      return top_k PaperChunk objects
```

**First-query latency:** ~3–8s for fetching + indexing 5 new papers.  
**Subsequent queries (cache hit):** <500ms.  
Panel runs already take 30–60s total — first-run warmup is acceptable.

---

## Domain → arXiv Category Mapping

```python
ARXIV_CATEGORIES = {
    "ai_ml_systems":   ["cs.AI", "cs.LG", "cs.CL", "cs.CV", "cs.SE"],
    "mathematics":     ["math.OC", "math.CO", "math.ST", "cs.GT"],
    "ops_research":    ["math.OC", "cs.GT", "econ.GN"],
    # Option C extension — add these when implementing Semantic Scholar source:
    "finance":         ["q-fin.RM", "q-fin.GN", "econ.GN"],
    "business_economics": ["econ.GN", "econ.TH"],
    "strategy_game_theory": ["cs.GT", "econ.TH"],
    "venture_finance": ["q-fin.GN", "econ.GN"],
}
```

---

## Quality Scoring

arXiv papers are preprints and not peer-reviewed by default. We supplement arXiv metadata with a single Semantic Scholar paper lookup at index time:

```
GET https://api.semanticscholar.org/graph/v1/paper/arXiv:{id}
    ?fields=citationCount,influentialCitationCount,publicationTypes,venue
```

This is one free API call per new paper, result cached in SQLite.

### Quality multiplier applied at RRF scoring

```
final_score = rrf_score × publication_multiplier × citation_boost

publication_multiplier:
  JournalArticle → 1.3
  Conference (known top venue) → 1.2
  Preprint only → 1.0

citation_boost:
  min(log(1 + influential_citation_count) / 7.0, 1.5)
  (7.0 ≈ log(1 + 1000) — a reasonable midpoint for influential citations;
   caps boost at 1.5× so blockbuster papers don't dominate;
   recent papers with few citations are not penalized)
```

Top venues per domain (used for conference classification):

```python
TOP_VENUES = {
    "ai_ml_systems": ["NeurIPS", "ICML", "ICLR", "CVPR", "ACL", "EMNLP"],
    "mathematics":   ["Annals of Mathematics", "Journal of the ACM", "SIAM"],
    "ops_research":  ["Operations Research", "Management Science", "EJOR"],
}
```

### UI surface

Each citation in the agent response includes a quality badge:
```
"Peer-reviewed · NeurIPS 2023 · 847 citations"
```

---

## Embedding Model

**Model:** `nomic-ai/nomic-embed-text-v1.5` (via sentence-transformers)
- 768-dim embeddings
- Runs on CPU (~100ms per batch of 32 chunks)
- ~550MB, cached at container build time
- No API cost

**Qdrant collection config:** cosine similarity, 768 dimensions, one collection per domain: `domain_ai_ml_systems`, `domain_mathematics`, `domain_ops_research`.

---

## PDF Parsing

**Library:** `pdfplumber` (over pymupdf)
- Better section heading detection on academic papers
- Cleaner text extraction for two-column layouts
- Slower than pymupdf; indexing is on the critical path for new papers but acceptable given 3–8s first-query budget

**Section detection:** regex match on common academic heading patterns. Papers without detectable sections fall back to 512-token fixed chunks tagged as `section: body`.

---

## Agent Integration

### What changes in `base_agent.py`

```python
# Before
tools=[{"type": "web_search_20250305", "name": "web_search", "max_uses": 3}]

# After
chunks = retriever.retrieve(query=question, domain=self.domain, top_k=8)
context_block = _format_chunks(chunks)
# injected into user_message as [CONTEXT] block; no tools parameter
```

### Chunk format injected into prompt

```
[CONTEXT — retrieved literature]

[1] Smith & Jones (2023) — "Attention is All You Need" (arXiv:1706.03762)
Section: results | NeurIPS 2017 | 90,241 citations
"Our model achieves 28.4 BLEU on the WMT 2014 English-to-German translation task..."

[2] Brown et al. (2020) — "Language Models are Few-Shot Learners" (arXiv:2005.14165)
Section: methods | NeurIPS 2020 | 12,847 citations
"We train GPT-3, an autoregressive language model with 175 billion parameters..."
```

The `_base.txt` persona already instructs agents to "ground every claim in the papers retrieved for you (provided below as [CONTEXT])". This fulfills that contract properly.

### Web search fallback

Keep `web_search` available but only activate it if `retriever.retrieve()` returns fewer than 3 chunks (cold domain, no cached papers yet). Preserves graceful degradation on first run.

---

## Infrastructure Changes

### docker-compose.yml addition

```yaml
qdrant:
  image: qdrant/qdrant:latest
  ports: ["6333:6333"]
  volumes: ["qdrant_data:/qdrant/storage"]
```

### Dockerfile addition

```dockerfile
RUN python -c "from sentence_transformers import SentenceTransformer; \
    SentenceTransformer('nomic-ai/nomic-embed-text-v1.5')"
```

### New dependencies (requirements.txt)

```
qdrant-client
sentence-transformers
pdfplumber
rank-bm25
httpx
```

---

## Error Handling

| Situation | Handling |
|---|---|
| arXiv API rate limit (3 req/s) | Exponential backoff, max 3 retries |
| PDF download fails | Skip paper, log warning, continue with remaining candidates |
| PDF parsing yields <3 chunks | Store abstract as single chunk, `section: abstract` |
| Qdrant unavailable at startup | Fall back to web_search tool, log error |
| Semantic Scholar quality lookup fails | Skip quality multiplier, use rrf_score only |
| arXiv categories return <5 results | Relax category filter, retry with broader search |

---

## BM25 Index

- `rank-bm25` library, one index per domain
- Built in-memory at startup from cached chunk texts in Qdrant
- Persisted as `cache/bm25_{domain}.pkl` so restarts don't rebuild from scratch
- RRF formula: `score(chunk) = Σ 1/(60 + rank_i)` across vector + BM25 results

---

## Testing

```bash
# Smoke test — retrieves, indexes, returns chunks
python -c "
from corpus.arxiv_retriever import ArxivRetriever
r = ArxivRetriever()
chunks = r.retrieve('transformer attention mechanisms', 'ai_ml_systems', top_k=5)
for c in chunks:
    print(c.title, c.section, c.score)
"
# Run again immediately — should be <500ms (cache hit)

# Quality metadata check
python -c "
from corpus.paper_cache import PaperCache
cache = PaperCache()
papers = cache.list_indexed()
for p in papers[:3]:
    print(p['arxiv_id'], p['publication_type'], p['citation_count'])
"
```

---

## Option C Extension Path

When ready to add Semantic Scholar as a discovery source and Unpaywall for broader full-text access:

1. Add `corpus/sources/semantic_scholar.py` — search 200M+ papers, richer ranking signals
2. Add `corpus/sources/unpaywall.py` — resolves open-access PDF URLs beyond arXiv
3. `ArxivRetriever` becomes `CorpusRetriever` with a configurable `sources: list[Source]`
4. `paper_cache.py` gains a `source` column (`arxiv` / `semantic_scholar` / `unpaywall`)
5. Chunker, embedder, Qdrant, BM25, RRF — all unchanged
6. Business/finance domains (`finance`, `business_economics`, `venture_finance`) activate with Semantic Scholar as their primary discovery source

The `Retriever` interface stays identical throughout. Agents are unaware of which sources are active.

### Tiered access model (commercial)

- **Open access tier** (included): arXiv + Unpaywall-resolved PDFs
- **Premium tier**: Semantic Scholar Academic Graph API (SciAgent pays, bundles into higher subscription tier)
- **Enterprise / BYOK (future)**: institutional API keys for Elsevier/Springer TDM APIs

---

## Out of Scope

- Ingestion of manually uploaded PDFs (future feature)
- Persistent user-specific corpora
- Full Semantic Scholar integration (Option C)
- Payment/subscription infrastructure for tiered access
