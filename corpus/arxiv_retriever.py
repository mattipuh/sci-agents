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
