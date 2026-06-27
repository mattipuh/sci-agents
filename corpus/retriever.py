from dataclasses import dataclass


@dataclass
class PaperChunk:
    chunk_id: str
    paper_id: str
    title: str
    authors: list[str]
    year: int
    journal: str
    section: str
    text: str
    score: float
    source_url: str


class Retriever:
    """Interface for paper retrieval. Subclass for real corpus; demo uses web search instead."""

    def retrieve(
        self,
        query: str,
        domain: str,
        top_k: int = 8,
        min_year: int | None = None,
    ) -> list[PaperChunk]:
        raise NotImplementedError
