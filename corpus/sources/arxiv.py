import time
from urllib.parse import quote_plus

import defusedxml.ElementTree as ET
import httpx

ARXIV_CATEGORIES: dict[str, list[str]] = {
    "ai_ml_systems":       ["cs.AI", "cs.LG", "cs.CL", "cs.CV", "cs.SE"],
    "mathematics":         ["math.OC", "math.CO", "math.ST", "cs.GT"],
    "ops_research":        ["math.OC", "cs.GT", "econ.GN"],
    # Option C extension:
    "finance":             ["q-fin.RM", "q-fin.GN", "econ.GN"],
    "business_economics":  ["econ.GN", "econ.TH"],
    "strategy_game_theory":["cs.GT", "econ.TH"],
    "venture_finance":     ["q-fin.GN", "econ.GN"],
}

_NS = {"atom": "http://www.w3.org/2005/Atom",
       "arxiv": "http://arxiv.org/schemas/atom"}
_BASE = "https://export.arxiv.org/api/query"


def search(query: str, domain: str, max_results: int = 10) -> list[dict]:
    categories = ARXIV_CATEGORIES.get(domain, ["cs.AI"])
    cat_filter = " OR ".join(f"cat:{c}" for c in categories)
    q = f"({quote_plus(query)}) AND ({quote_plus(cat_filter)})"
    url = f"{_BASE}?search_query={q}&max_results={max_results}&sortBy=relevance"

    for attempt in range(3):
        try:
            resp = httpx.get(url, timeout=15.0)
            resp.raise_for_status()
            return _parse(resp.text)
        except Exception:
            if attempt == 2:
                return []
            time.sleep(2 ** attempt)
    return []


def _parse(xml_text: str) -> list[dict]:
    try:
        root = ET.fromstring(xml_text)
    except ET.ParseError:
        return []
    results = []
    for entry in root.findall("atom:entry", _NS):
        try:
            results.append(_parse_entry(entry))
        except Exception:
            continue
    return results


def _parse_entry(entry) -> dict:
    raw_id = entry.findtext("atom:id", "", _NS).rstrip("/").split("/")[-1]
    arxiv_id = raw_id.rsplit("v", 1)[0] if "v" in raw_id and raw_id.split("v")[-1].isdigit() else raw_id

    title = (entry.findtext("atom:title", "", _NS) or "").strip().replace("\n", " ")
    year = int((entry.findtext("atom:published", "2000", _NS) or "2000")[:4])
    abstract = (entry.findtext("atom:summary", "", _NS) or "").strip().replace("\n", " ")
    comment = (entry.findtext("arxiv:comment", "", _NS) or "").strip()
    authors = [
        (a.findtext("atom:name", "", _NS) or "").strip()
        for a in entry.findall("atom:author", _NS)
    ]
    pdf_url = next(
        (lnk.get("href", "") for lnk in entry.findall("atom:link", _NS)
         if lnk.get("type") == "application/pdf"),
        "",
    )
    return {
        "arxiv_id": arxiv_id,
        "title": title,
        "authors": authors,
        "year": year,
        "abstract": abstract,
        "pdf_url": pdf_url,
        "source_url": f"https://arxiv.org/abs/{arxiv_id}",
        "comment": comment,
    }
