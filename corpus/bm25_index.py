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
        return [(c, max(0.0, float(s))) for c, s in ranked[:top_k]]

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
            # Safe: pickle files are created only by our own save() method in trusted cache_dir
            with open(path, "rb") as f:
                data = pickle.load(f)
            self._indexes[domain] = data["index"]
            self._chunks[domain] = data["chunks"]
            return True
        except Exception as exc:
            log.warning("Failed to load BM25 index for %s: %s", domain, exc)
            return False
