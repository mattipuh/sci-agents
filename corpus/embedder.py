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
