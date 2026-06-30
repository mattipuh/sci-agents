import logging
import re

import pdfplumber

from corpus.retriever import PaperChunk

log = logging.getLogger(__name__)

_MAX_TOKENS = 512
_OVERLAP_TOKENS = 50
_MIN_TOKENS = 100

_SECTION_PATTERNS = [
    (re.compile(r"^\s*abstract\s*$", re.I),            "abstract"),
    (re.compile(r"^\s*\d*\.?\s*introduction\b", re.I), "introduction"),
    (re.compile(r"^\s*\d*\.?\s*related work", re.I),   "introduction"),
    (re.compile(r"^\s*\d*\.?\s*method",        re.I),  "methods"),
    (re.compile(r"^\s*\d*\.?\s*result",        re.I),  "results"),
    (re.compile(r"^\s*\d*\.?\s*discussion",    re.I),  "discussion"),
    (re.compile(r"^\s*\d*\.?\s*conclusion",    re.I),  "discussion"),
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
        section_chunks = _chunk_text(section_text, _MAX_TOKENS, _OVERLAP_TOKENS)
        for text in section_chunks:
            # Drop fragments only when a section was long enough to need splitting;
            # a sole small chunk from a short section is kept as-is.
            if len(section_chunks) > 1 and len(text.split()) < _MIN_TOKENS:
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
