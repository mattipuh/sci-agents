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
