"""权威数据集站点检索。

覆盖的站点（均支持免密钥访问）：
    HuggingFace Datasets   最大的开源数据集托管平台
    Zenodo                 CERN 开放数据仓库，论文配套数据常见
    Kaggle                 竞赛与社区数据集（列表接口可能需要登录，失败即跳过）
    OpenDataLab            国内 AI 数据集平台（页面抓取，稳定性一般）

所有检索函数都遵循「失败返回空列表、不抛异常」的约定：
单个站点挂掉不应该让整条流水线中断。
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional

from ..httpclient import get_json, get_text


@dataclass
class DatasetCandidate:
    name: str
    source: str
    url: str = ""
    download_url: str = ""
    description: str = ""
    license: str = ""
    annotation_format: str = ""
    num_images: Optional[int] = None
    num_classes: Optional[int] = None
    classes: List[str] = field(default_factory=list)
    downloads: Optional[int] = None
    homepage: str = ""
    raw: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @property
    def key(self) -> str:
        return f"{self.source}:{self.name}"


def _keywords_to_query(keywords: List[str]) -> str:
    return " ".join(k.strip() for k in keywords if k and k.strip())


def _truncate(text: str, limit: int = 400) -> str:
    text = (text or "").replace("\r", " ").replace("\n", " ").strip()
    return text[:limit]


# ---------------------------------------------------------------------------
# HuggingFace
# ---------------------------------------------------------------------------


def search_huggingface(keywords: List[str], limit: int = 8, timeout: float = 30.0) -> List[DatasetCandidate]:
    out: List[DatasetCandidate] = []
    for kw in keywords:
        data = get_json(
            "https://huggingface.co/api/datasets",
            params={"search": kw, "limit": limit, "full": "true"},
            timeout=timeout,
        )
        if not isinstance(data, list):
            continue
        for item in data:
            card = (item.get("cardData") or {}) if isinstance(item, dict) else {}
            name = item.get("id", "")
            if not name:
                continue
            out.append(
                DatasetCandidate(
                    name=name,
                    source="huggingface",
                    url=f"https://huggingface.co/datasets/{name}",
                    download_url=f"https://huggingface.co/datasets/{name}/tree/main",
                    description=_truncate(card.get("pretty_name") or item.get("description") or ""),
                    license=str(card.get("license") or ""),
                    downloads=item.get("downloads"),
                    homepage=item.get("cardData", {}).get("homepage", "") if isinstance(item.get("cardData"), dict) else "",
                    raw={
                        "tags": item.get("tags", [])[:12],
                        "siblings": [s.get("rfilename") for s in (item.get("siblings") or [])][:20],
                        "lastModified": item.get("lastModified"),
                    },
                )
            )
    return _dedupe(out)


def list_huggingface_files(repo: str, timeout: float = 30.0) -> List[Dict[str, Any]]:
    """列出 HF 数据集仓库的文件树（含下载 URL）。"""
    url = f"https://huggingface.co/api/datasets/{repo}/tree/main"
    data = get_json(url, params={"recursive": "true"}, timeout=timeout)
    files: List[Dict[str, Any]] = []
    if isinstance(data, list):
        for item in data:
            if item.get("type") != "file":
                continue
            path = item.get("path", "")
            files.append(
                {
                    "path": path,
                    "size": item.get("size"),
                    "url": f"https://huggingface.co/datasets/{repo}/resolve/main/{path}",
                }
            )
    return files


# ---------------------------------------------------------------------------
# Zenodo
# ---------------------------------------------------------------------------


def search_zenodo(keywords: List[str], limit: int = 8, timeout: float = 30.0) -> List[DatasetCandidate]:
    out: List[DatasetCandidate] = []
    query = _keywords_to_query(keywords)
    if not query:
        return out
    data = get_json(
        "https://zenodo.org/api/records",
        params={"q": query, "size": limit, "type": "dataset"},
        timeout=timeout,
    )
    hits = (data or {}).get("hits", {}).get("hits", []) if isinstance(data, dict) else []
    for item in hits:
        meta = item.get("metadata", {}) or {}
        files = item.get("files", []) or []
        total_bytes = sum(f.get("size", 0) or 0 for f in files)
        out.append(
            DatasetCandidate(
                name=meta.get("title", "")[:120],
                source="zenodo",
                url=f"https://zenodo.org/records/{item.get('id')}",
                download_url=(files[0].get("links", {}) or {}).get("self", "") if files else "",
                description=_truncate(_strip_html(meta.get("description", ""))),
                license=(meta.get("license") or {}).get("id", "") if isinstance(meta.get("license"), dict) else str(meta.get("license", "")),
                num_images=None,
                raw={"doi": meta.get("doi"), "files": len(files), "size_bytes": total_bytes},
            )
        )
    return _dedupe(out)


def _strip_html(text: str) -> str:
    import re

    return re.sub(r"<[^>]+>", " ", text or "")


# ---------------------------------------------------------------------------
# Kaggle / OpenDataLab
# ---------------------------------------------------------------------------


def search_kaggle(keywords: List[str], limit: int = 8, timeout: float = 30.0) -> List[DatasetCandidate]:
    """Kaggle 数据集列表（未登录时通常 401/403，失败即跳过）。"""
    out: List[DatasetCandidate] = []
    query = _keywords_to_query(keywords)
    if not query:
        return out
    data = get_json(
        "https://www.kaggle.com/api/v1/datasets/list",
        params={"search": query, "pageSize": limit},
        timeout=timeout,
        retries=0,
    )
    if not isinstance(data, list):
        return out
    for item in data:
        ref = item.get("ref", "")
        if not ref:
            continue
        out.append(
            DatasetCandidate(
                name=item.get("title", ref),
                source="kaggle",
                url=f"https://www.kaggle.com/datasets/{ref}",
                description=_truncate(item.get("subtitle", "") or item.get("description", "")),
                num_images=item.get("totalBytes"),
                raw={"ref": ref, "usabilityRating": item.get("usabilityRating")},
            )
        )
    return _dedupe(out)


def search_opendatalab(keywords: List[str], limit: int = 8, timeout: float = 30.0) -> List[DatasetCandidate]:
    """OpenDataLab 公开搜索接口（若接口变更则自动跳过）。"""
    out: List[DatasetCandidate] = []
    query = _keywords_to_query(keywords)
    if not query:
        return out
    data = get_json(
        "https://opendatalab.com/api/v2/datasets",
        params={"keywords": query, "pageSize": limit},
        timeout=timeout,
        retries=0,
    )
    if not isinstance(data, list):
        return out
    for item in data:
        name = item.get("name") or item.get("id") or ""
        if not name:
            continue
        out.append(
            DatasetCandidate(
                name=str(name),
                source="opendatalab",
                url=f"https://opendatalab.com/OpenDataLab/{name}",
                description=_truncate(item.get("description", "")),
                raw={"id": item.get("id")},
            )
        )
    return _dedupe(out)


# ---------------------------------------------------------------------------
# 聚合
# ---------------------------------------------------------------------------


def search_all(
    keywords: List[str],
    limit: int = 8,
    sources: Optional[List[str]] = None,
    timeout: float = 30.0,
) -> List[DatasetCandidate]:
    """按 sources 顺序检索并合并去重。"""
    sources = sources or ["huggingface", "zenodo"]
    runners = {
        "huggingface": search_huggingface,
        "zenodo": search_zenodo,
        "kaggle": search_kaggle,
        "opendatalab": search_opendatalab,
    }
    out: List[DatasetCandidate] = []
    for name in sources:
        fn = runners.get(name)
        if fn is None:
            continue
        try:
            out.extend(fn(keywords, limit=limit, timeout=timeout))
        except Exception:
            continue
    return _dedupe(out)


def _dedupe(items: List[DatasetCandidate]) -> List[DatasetCandidate]:
    seen, out = set(), []
    for item in items:
        if item.key in seen:
            continue
        seen.add(item.key)
        out.append(item)
    return out


def guess_annotation_format(candidate: DatasetCandidate) -> str:
    """根据名称/描述/文件清单猜测标注格式。"""
    text = " ".join(
        filter(None, [candidate.name, candidate.description, " ".join(candidate.raw.get("siblings", []) if isinstance(candidate.raw.get("siblings"), list) else [])])
    ).lower()
    if ".xml" in text or "voc" in text or "pascal" in text:
        return "voc_xml"
    if "coco" in text or "instances_" in text:
        return "coco_json"
    if ".txt" in text or "yolo" in text:
        return "yolo_txt"
    return "未知（需人工确认）"
