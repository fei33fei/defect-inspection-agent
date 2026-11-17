"""GitHub 官方仓库检索与克隆。

流程：
    1. 先查内置的「已知官方仓库表」——这类项目（YOLOv8/v9/v10/v11、RT-DETR 等）
       官方仓库高度稳定，内置表比盲目搜索准得多；
    2. 表里没有，再走 GitHub 搜索 API（未认证 10 次/分钟，够用）；
    3. 克隆用 `git clone --depth 1`，失败会记录原因而不是中断流水线。

注意：GitHub 搜索 API 在未登录状态下有速率限制，
所以这里做了本地缓存（models/_search_cache.json），避免重复查询。
"""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from ..httpclient import get_json

# ---------------------------------------------------------------------------
# 已知官方仓库（名字 -> 仓库标识）
# ---------------------------------------------------------------------------
KNOWN_REPOS: Dict[str, Dict[str, str]] = {
    "yolov8": {"repo": "ultralytics/ultralytics", "desc": "YOLOv8 官方实现（含 v9/v10/v11 系列与大量模块）"},
    "yolov9": {"repo": "WongKinYiu/yolov9", "desc": "YOLOv9 官方实现"},
    "yolov10": {"repo": "THU-MIG/yolov10", "desc": "YOLOv10 官方实现（NMS-free）"},
    "yolov11": {"repo": "ultralytics/ultralytics", "desc": "YOLOv11（与 YOLOv8 同仓库）"},
    "yolov12": {"repo": "suliangxie/yolov12", "desc": "YOLOv12（注意力中心化）"},
    "yolov5": {"repo": "ultralytics/yolov5", "desc": "YOLOv5 官方实现"},
    "yolov6": {"repo": "meituan/YOLOv6", "desc": "YOLOv6 官方实现"},
    "yolov7": {"repo": "WongKinYiu/yolov7", "desc": "YOLOv7 官方实现"},
    "yolox": {"repo": "Megvii-BaseDetection/YOLOX", "desc": "YOLOX 官方实现"},
    "rtdetr": {"repo": "lyuwenyu/RT-DETR", "desc": "RT-DETR 官方实现"},
    "rt-detr": {"repo": "lyuwenyu/RT-DETR", "desc": "RT-DETR 官方实现"},
    "rtx": {"repo": "lyuwenyu/RT-DETR", "desc": "RT-DETR 官方实现"},
    "ultralytics": {"repo": "ultralytics/ultralytics", "desc": "Ultralytics 官方仓库"},
    "yolos": {"repo": "THU-MIG/yolos", "desc": "YOLOS 官方实现"},
    "damoyolo": {"repo": "tianweiy/DAMO-YOLO", "desc": "DAMO-YOLO 官方实现"},
    "gold_yolo": {"repo": "huawei-noah/Efficient-Repo/tree/master/YOLOv7", "desc": "Gold-YOLO"},
    "ppyoloe": {"repo": "PaddlePaddle/PaddleDetection", "desc": "PP-YOLOE（PaddleDetection）"},
    "deim": {"repo": "ShihuaHuang95/DEIM", "desc": "DEIM 官方实现"},
    "faster-rcnn": {"repo": "pytorch/vision", "desc": "torchvision（两阶段检测器实现）"},
    "mmdetection": {"repo": "open-mmlab/mmdetection", "desc": "MMDetection 官方实现"},
    "detectron2": {"repo": "facebookresearch/detectron2", "desc": "Detectron2 官方实现"},
    "yolo-se": {"repo": "ultralytics/ultralytics", "desc": "SE/CBAM 等注意力模块可基于 Ultralytics 实现"},
}


@dataclass
class RepoCandidate:
    full_name: str
    html_url: str
    description: str = ""
    stars: int = 0
    forks: int = 0
    license: str = ""
    language: str = ""
    updated_at: str = ""
    source: str = "github-search"
    raw: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def normalize_model_name(name: str) -> str:
    """把 'yolov8-se'、'YOLOv8'、'rt_detr' 归一化成检索关键词。"""
    n = (name or "").lower().strip()
    n = re.sub(r"[-_\s]+", "", n)
    n = re.sub(r"(attention|baseline|official|repo|weights|pt)$", "", n)
    return n


def resolve_repo(model_name: str) -> Optional[RepoCandidate]:
    """内置表优先：命中则返回官方仓库候选。"""
    key = normalize_model_name(model_name)
    entry = KNOWN_REPOS.get(key)
    if not entry:
        # 尝试前缀匹配，例如 'yolov8-se' -> 'yolov8'
        for known, val in KNOWN_REPOS.items():
            if key.startswith(known) or known.startswith(key):
                entry = val
                break
    if not entry:
        return None
    full = entry["repo"]
    owner, _, repo = full.partition("/")
    return RepoCandidate(
        full_name=full,
        html_url=f"https://github.com/{full}",
        description=entry.get("desc", ""),
        stars=0,
        source="known-registry",
        raw={"owner": owner, "repo": repo},
    )


def search_repositories(query: str, limit: int = 8, timeout: float = 30.0) -> List[RepoCandidate]:
    """GitHub 仓库搜索（未认证限流，失败返回空列表）。"""
    data = get_json(
        "https://api.github.com/search/repositories",
        params={"q": query, "sort": "stars", "order": "desc", "per_page": limit},
        headers={"Accept": "application/vnd.github+json"},
        timeout=timeout,
        retries=1,
    )
    items = (data or {}).get("items") if isinstance(data, dict) else None
    out: List[RepoCandidate] = []
    for item in items or []:
        lic = item.get("license") or {}
        out.append(
            RepoCandidate(
                full_name=item.get("full_name", ""),
                html_url=item.get("html_url", ""),
                description=item.get("description") or "",
                stars=item.get("stargazers_count", 0),
                forks=item.get("forks_count", 0),
                license=lic.get("spdx_id", "") if isinstance(lic, dict) else "",
                language=item.get("language") or "",
                updated_at=item.get("pushed_at", ""),
                source="github-search",
                raw={"default_branch": item.get("default_branch", "main")},
            )
        )
    return out


def get_repo(full_name: str, timeout: float = 30.0) -> Optional[RepoCandidate]:
    """查询单个仓库详情（拿 star / license 等元数据）。"""
    data = get_json(
        f"https://api.github.com/repos/{full_name}",
        headers={"Accept": "application/vnd.github+json"},
        timeout=timeout,
        retries=1,
    )
    if not isinstance(data, dict) or "full_name" not in data:
        return None
    lic = data.get("license") or {}
    return RepoCandidate(
        full_name=data.get("full_name", ""),
        html_url=data.get("html_url", ""),
        description=data.get("description") or "",
        stars=data.get("stargazers_count", 0),
        forks=data.get("forks_count", 0),
        license=lic.get("spdx_id", "") if isinstance(lic, dict) else "",
        language=data.get("language") or "",
        updated_at=data.get("pushed_at", ""),
        source="github-api",
        raw={"default_branch": data.get("default_branch", "main")},
    )


def clone_repo(repo_url: str, dest: Path, timeout: float = 1800.0) -> Dict[str, Any]:
    """浅克隆仓库到 dest。返回 {ok, dest, message}。

    处理两个常见坑：
    1. 全局配置里残留的代理（http.proxy=127.0.0.1:xxxx）可能没在运行 → 第一次失败后
       用 `-c http.proxy= -c https.proxy=` 清掉代理再试；
    2. 非交互环境下必须关掉凭据提示，否则会卡住等输入。
    """
    dest = Path(dest)
    if dest.exists() and any(dest.iterdir()):
        return {"ok": True, "dest": str(dest), "message": "已存在，跳过克隆"}
    if shutil.which("git") is None:
        return {"ok": False, "dest": str(dest), "message": "系统未安装 git，无法克隆仓库"}
    dest.parent.mkdir(parents=True, exist_ok=True)

    env = dict(os.environ)
    env["GIT_TERMINAL_PROMPT"] = "0"
    env["GCM_INTERACTIVE"] = "never"

    attempts = [
        ["git", "clone", "--depth", "1", repo_url, str(dest)],
        [
            "git",
            "-c",
            "http.proxy=",
            "-c",
            "https.proxy=",
            "clone",
            "--depth",
            "1",
            repo_url,
            str(dest),
        ],
    ]
    last_msg = ""
    for cmd in attempts:
        try:
            proc = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=timeout,
                encoding="utf-8",
                errors="replace",
                env=env,
            )
            if proc.returncode == 0:
                return {"ok": True, "dest": str(dest), "message": "克隆成功"}
            last_msg = (proc.stderr or proc.stdout or "克隆失败").strip()[:500]
        except subprocess.TimeoutExpired:
            last_msg = "克隆超时"
        except Exception as exc:  # noqa: BLE001
            last_msg = str(exc)
        # 第一次失败后清掉残留的半成品目录再试
        if dest.exists() and not any(dest.iterdir()):
            dest.rmdir()
    return {"ok": False, "dest": str(dest), "message": last_msg}


def clone_from_candidate(candidate: RepoCandidate, models_dir: Path, timeout: float = 1800.0) -> Dict[str, Any]:
    """按候选仓库克隆到 models_dir/<repo_name>。"""
    name = candidate.full_name.split("/")[-1]
    dest = Path(models_dir) / name
    res = clone_repo(candidate.html_url + ".git", dest, timeout=timeout)
    res["full_name"] = candidate.full_name
    res["html_url"] = candidate.html_url
    return res


def load_cache(path: Path) -> Dict[str, Any]:
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except Exception:
        return {}


def save_cache(path: Path, data: Dict[str, Any]) -> None:
    try:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
    except Exception:
        pass
