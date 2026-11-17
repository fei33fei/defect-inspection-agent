"""S3 · 数据集下载（对应工作流第 4 步）。

四种来源：
    local         本地已有数据集 —— 用硬链接（失败回退复制）收拢到工作区，
                 保证工作区自包含，同时不占额外磁盘；
    url           直链下载 —— 支持断点续传，记录 SHA256；
    huggingface   HF 数据集仓库 —— 走公开 API 列举文件再逐个下载；
    kaggle        需要账号与凭据，Agent 无法代持凭据，
                 这里只生成标准命令让人执行，执行完可再跑一次本阶段。

每次下载都会写 inventory.json：来源、文件、大小、校验和、失败原因。
"""

from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path
from typing import Any, Dict, List

from ..httpclient import download as http_download
from ..search import datasets as ds_search
from .base import StageContext, StageResult

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}
MAX_TOTAL_BYTES = 8 * 1024 ** 3  # 单次下载上限 8GB，防止误下巨型数据集


def _sha256(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        while True:
            block = fh.read(chunk)
            if not block:
                break
            h.update(block)
    return h.hexdigest()


def _mirror_tree(src: Path, dst: Path) -> Dict[str, Any]:
    """用硬链接把目录树镜像到工作区（跨盘或不支持时回退复制）。"""
    files = 0
    linked = 0
    copied = 0
    errors: List[str] = []
    for path in src.rglob("*"):
        if not path.is_file():
            continue
        rel = path.relative_to(src)
        target = dst / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists():
            continue
        try:
            import os

            os.link(path, target)
            linked += 1
        except (OSError, NotImplementedError):
            try:
                shutil.copy2(path, target)
                copied += 1
            except Exception as exc:  # noqa: BLE001
                errors.append(f"{rel}: {exc}")
                continue
        files += 1
    return {"files": files, "hardlinked": linked, "copied": copied, "errors": errors}


def run(ctx: StageContext) -> StageResult:
    res = StageResult()
    ws = ctx.ws
    cfg = ctx.cfg
    ws.data_raw.mkdir(parents=True, exist_ok=True)

    inventory: Dict[str, Any] = {
        "source": cfg.data.source,
        "local_dir": str(cfg.data.local_dir) if cfg.data.local_dir else "",
        "items": [],
        "errors": [],
    }

    if cfg.data.source == "local":
        src = cfg.resolve(cfg.data.local_dir) if cfg.data.local_dir else None
        if src is None or not src.exists():
            inventory["errors"].append(f"本地数据目录不存在: {src}")
            res.ok = False
        else:
            ctx.logger.info(f"收拢本地数据集: {src} -> {ws.data_raw}")
            info = _mirror_tree(src, ws.data_raw)
            inventory["mirror"] = info
            inventory["items"].append(
                {"kind": "local-mirror", "source": str(src), "dest": str(ws.data_raw), **info}
            )
            if info["errors"]:
                inventory["errors"].extend(info["errors"][:20])

    elif cfg.data.source == "url":
        total = 0
        for url in cfg.data.urls:
            name = url.split("?")[0].rstrip("/").split("/")[-1] or "download.bin"
            dest = ws.data_raw / name
            try:
                ctx.logger.info(f"下载 {url}")
                http_download(url, dest)
                size = dest.stat().st_size
                total += size
                inventory["items"].append(
                    {
                        "kind": "url",
                        "url": url,
                        "path": str(dest),
                        "bytes": size,
                        "sha256": _sha256(dest),
                    }
                )
            except Exception as exc:  # noqa: BLE001
                inventory["errors"].append(f"{url}: {exc}")
            if total > MAX_TOTAL_BYTES:
                inventory["errors"].append("超过单次下载上限，已停止")
                break
        if not inventory["items"]:
            res.ok = False

    elif cfg.data.source == "huggingface":
        repo = cfg.data.hf_repo
        files = ds_search.list_huggingface_files(repo, timeout=float(cfg.search.timeout))
        if not files:
            inventory["errors"].append(f"HF 仓库 {repo} 没有列出文件（离线或仓库不存在）")
            res.ok = False
        total = 0
        for f in files:
            if total > MAX_TOTAL_BYTES:
                break
            suffix = Path(f["path"]).suffix.lower()
            dest = ws.data_raw / f["path"]
            try:
                http_download(f["url"], dest)
                size = dest.stat().st_size
                total += size
                inventory["items"].append(
                    {
                        "kind": "huggingface",
                        "repo": repo,
                        "path": str(dest),
                        "bytes": size,
                        "sha256": _sha256(dest),
                        "is_image": suffix in IMAGE_SUFFIXES,
                    }
                )
            except Exception as exc:  # noqa: BLE001
                inventory["errors"].append(f"{f['path']}: {exc}")
        if not inventory["items"]:
            res.ok = False

    elif cfg.data.source == "kaggle":
        cmd = f"kaggle datasets download -d {cfg.data.kaggle_dataset} -p \"{ws.data_raw}\""
        inventory["manual_command"] = cmd
        inventory["errors"].append(
            "Kaggle 需要账号凭据，Agent 不代持凭据。请手动执行上面的命令，"
            "下载完成后重新运行本阶段（或直接进入预处理阶段）。"
        )
        res.ok = False

    # --- 清单落盘 ---
    inv_path = ws.datasets / "inventory.json"
    inv_path.write_text(json.dumps(inventory, indent=2, ensure_ascii=False), encoding="utf-8")

    report = ["# 数据来源与获取记录", "", f"- **来源类型**: {cfg.data.source}", ""]
    if cfg.data.local_dir:
        report.append(f"- **本地目录**: {cfg.resolve(cfg.data.local_dir)}")
    if cfg.data.hf_repo:
        report.append(f"- **HuggingFace 仓库**: {cfg.data.hf_repo}")
    report.append("")
    if inventory["items"]:
        report.append("| 类型 | 来源 | 目标 | 大小 |")
        report.append("| --- | --- | --- | --- |")
        for item in inventory["items"]:
            report.append(
                f"| {item.get('kind')} | {item.get('source') or item.get('repo') or item.get('url', '')} "
                f"| {item.get('dest') or item.get('path', '')} | {item.get('bytes', '-')} |"
            )
    else:
        report.append("未获取到数据文件。")
    report.append("")
    if inventory["errors"]:
        report.append("## 问题与注意事项")
        report.append("")
        for e in inventory["errors"][:30]:
            report.append(f"- {e}")
        report.append("")
    if inventory.get("manual_command"):
        report.append("## 需要手动执行的命令")
        report.append("")
        report.append("```bash")
        report.append(inventory["manual_command"])
        report.append("```")
        report.append("")

    src_report = ws.datasets / "source_report.md"
    src_report.write_text("\n".join(report), encoding="utf-8")

    res.add(inv_path)
    res.add(src_report)
    res.data = {
        "items": len(inventory["items"]),
        "errors": len(inventory["errors"]),
        "manual_command": inventory.get("manual_command", ""),
    }
    res.message = f"数据获取：{len(inventory['items'])} 项，{len(inventory['errors'])} 个问题"
    return res
