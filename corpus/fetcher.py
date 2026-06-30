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
