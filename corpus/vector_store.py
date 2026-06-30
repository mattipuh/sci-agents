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
