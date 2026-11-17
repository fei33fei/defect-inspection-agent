"""学术文献检索（为「算法来源 / 改进方式」提供依据）。

站点：
    arXiv             预印本，检索算法改进最常用
    Semantic Scholar  引文网络，可按被引数排序
    Crossref          DOI 元数据，期刊出处权威
    OpenAlex          开放学术图谱，覆盖期刊论文

返回的每条论文都带链接，最终会被写进项目报告的「算法来源」章节。
"""

from __future__ import annotations

import re
import xml.etree.ElementTree as ET
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional

from ..httpclient import get_json, get_text


@dataclass
class PaperCandidate:
    title: str
    source: str
    url: str = ""
    year: Optional[int] = None
    venue: str = ""
    authors: List[str] = field(default_factory=list)
    abstract: str = ""
    citations: Optional[int] = None
    doi: str = ""
    arxiv_id: str = ""
    raw: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @property
    def key(self) -> str:
        return f"{self.source}:{self.title[:60]}"


def _truncate(text: str, limit: int = 500) -> str:
    text = re.sub(r"\s+", " ", (text or "")).strip()
    return text[:limit]


def _keywords_to_query(keywords: List[str]) -> str:
    return " ".join(k.strip() for k in keywords if k and k.strip())


# ---------------------------------------------------------------------------
# arXiv
# ---------------------------------------------------------------------------

_ATOM = "{http://www.w3.org/2005/Atom}"


def search_arxiv(keywords: List[str], limit: int = 8, timeout: float = 30.0) -> List[PaperCandidate]:
    query = _keywords_to_query(keywords)
    if not query:
        return []
    text = get_text(
        "https://export.arxiv.org/api/query",
        params={
            "search_query": f"all:{query}",
            "start": 0,
            "max_results": limit,
            "sortBy": "relevance",
        },
        timeout=timeout,
    )
    if not text:
        return []
    try:
        root = ET.fromstring(text)
    except ET.ParseError:
        return []

    out: List[PaperCandidate] = []
    for entry in root.findall(f"{_ATOM}entry"):
        title = (entry.findtext(f"{_ATOM}title") or "").strip()
        if not title:
            continue
        summary = (entry.findtext(f"{_ATOM}summary") or "").strip()
        link = entry.findtext(f"{_ATOM}id") or ""
        published = (entry.findtext(f"{_ATOM}published") or "")[:4]
        authors = [
            (a.findtext(f"{_ATOM}name") or "").strip()
            for a in entry.findall(f"{_ATOM}author")
        ]
        year = int(published) if published.isdigit() else None
        arxiv_id = link.rsplit("/", 1)[-1] if link else ""
        out.append(
            PaperCandidate(
                title=re.sub(r"\s+", " ", title),
                source="arxiv",
                url=link,
                year=year,
                venue="arXiv 预印本",
                authors=authors[:8],
                abstract=_truncate(summary, 700),
                arxiv_id=arxiv_id,
                raw={"categories": [c.get("term") for c in entry.findall(f"{_ATOM}category")][:5]},
            )
        )
    return out


# ---------------------------------------------------------------------------
# Semantic Scholar
# ---------------------------------------------------------------------------


def search_semanticscholar(keywords: List[str], limit: int = 8, timeout: float = 30.0) -> List[PaperCandidate]:
    query = _keywords_to_query(keywords)
    if not query:
        return []
    data = get_json(
        "https://api.semanticscholar.org/graph/v1/paper/search",
        params={
            "query": query,
            "limit": limit,
            "fields": "title,abstract,year,venue,authors,url,externalIds,citationCount,openAccessPdf",
        },
        timeout=timeout,
        retries=1,
    )
    if not isinstance(data, dict):
        return []
    out: List[PaperCandidate] = []
    for item in data.get("data", []) or []:
        out.append(
            PaperCandidate(
                title=item.get("title", ""),
                source="semanticscholar",
                url=item.get("url", ""),
                year=item.get("year"),
                venue=item.get("venue", "") or "",
                authors=[a.get("name", "") for a in (item.get("authors") or [])][:8],
                abstract=_truncate(item.get("abstract") or ""),
                citations=item.get("citationCount"),
                doi=((item.get("externalIds") or {}).get("DOI")) or "",
                arxiv_id=((item.get("externalIds") or {}).get("ArXiv")) or "",
                raw={"openAccessPdf": (item.get("openAccessPdf") or {}).get("url", "")},
            )
        )
    return out


# ---------------------------------------------------------------------------
# Crossref
# ---------------------------------------------------------------------------


def search_crossref(keywords: List[str], limit: int = 8, timeout: float = 30.0) -> List[PaperCandidate]:
    query = _keywords_to_query(keywords)
    if not query:
        return []
    data = get_json(
        "https://api.crossref.org/works",
        params={"query": query, "rows": limit, "select": "title,abstract,issued,container-title,author,URL,DOI,is-referenced-by-count"},
        timeout=timeout,
        retries=1,
    )
    items = ((data or {}).get("message") or {}).get("items") or []
    out: List[PaperCandidate] = []
    for item in items:
        titles = item.get("title") or []
        title = titles[0] if titles else ""
        if not title:
            continue
        issued = ((item.get("issued") or {}).get("date-parts") or [[None]])[0]
        year = issued[0] if issued and isinstance(issued[0], int) else None
        out.append(
            PaperCandidate(
                title=title,
                source="crossref",
                url=item.get("URL", ""),
                year=year,
                venue=(item.get("container-title") or [""])[0],
                authors=[
                    " ".join(filter(None, [a.get("given"), a.get("family")]))
                    for a in (item.get("author") or [])
                ][:8],
                abstract=_truncate(re.sub(r"<[^>]+>", " ", item.get("abstract") or "")),
                citations=item.get("is-referenced-by-count"),
                doi=item.get("DOI", ""),
            )
        )
    return out


# ---------------------------------------------------------------------------
# OpenAlex
# ---------------------------------------------------------------------------


def search_openalex(keywords: List[str], limit: int = 8, timeout: float = 30.0) -> List[PaperCandidate]:
    query = _keywords_to_query(keywords)
    if not query:
        return []
    data = get_json(
        "https://api.openalex.org/works",
        params={"search": query, "per-page": limit},
        timeout=timeout,
        retries=1,
    )
    if not isinstance(data, dict):
        return []
    out: List[PaperCandidate] = []
    for item in data.get("results", []) or []:
        loc = (item.get("primary_location") or {}).get("source") or {}
        out.append(
            PaperCandidate(
                title=item.get("title", "") or "",
                source="openalex",
                url=item.get("doi") or item.get("id", ""),
                year=item.get("publication_year"),
                venue=loc.get("display_name", "") or "",
                authors=[a.get("author", {}).get("display_name", "") for a in (item.get("authorships") or [])][:8],
                abstract="",
                citations=item.get("cited_by_count"),
                doi=(item.get("doi") or "").replace("https://doi.org/", ""),
                raw={"type": item.get("type"), "is_oa": (item.get("open_access") or {}).get("is_oa")},
            )
        )
    return out


# ---------------------------------------------------------------------------
# 聚合
# ---------------------------------------------------------------------------

_RUNNERS = {
    "arxiv": search_arxiv,
    "semanticscholar": search_semanticscholar,
    "crossref": search_crossref,
    "openalex": search_openalex,
}


def search_all(
    keywords: List[str],
    limit: int = 8,
    sources: Optional[List[str]] = None,
    timeout: float = 30.0,
) -> List[PaperCandidate]:
    sources = sources or ["arxiv", "semanticscholar"]
    out: List[PaperCandidate] = []
    for name in sources:
        fn = _RUNNERS.get(name)
        if fn is None:
            continue
        try:
            out.extend(fn(keywords, limit=limit, timeout=timeout))
        except Exception:
            continue
    return _dedupe(out)


def _dedupe(items: List[PaperCandidate]) -> List[PaperCandidate]:
    """按标题去重，保留信息更完整的一条。"""
    best: Dict[str, PaperCandidate] = {}
    for item in items:
        key = re.sub(r"[^a-z0-9]", "", item.title.lower())[:80]
        if not key:
            continue
        cur = best.get(key)
        if cur is None or len(item.abstract) > len(cur.abstract) or (item.citations or 0) > (cur.citations or 0):
            best[key] = item
    return list(best.values())
