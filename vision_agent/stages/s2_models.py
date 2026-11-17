"""S2 · 官方模型仓库检索与克隆（对应工作流第 3 步）。

三层策略：
    1. 本地仓库优先：project_root 里已经有该模型的 YAML（你的 ultralytics 魔改版
       就带 yolov8-se / yolov8-ema / yolov8-cbam），直接复用，不再下载；
    2. 内置官方表：YOLOv8/v9/v10/v11、RT-DETR 这类仓库地址高度稳定，直接命中；
    3. GitHub 搜索 API：表里没有再搜，按 star 排序取第一个。

另外负责生成「组合变体」的模型 YAML（例如 SE+EMA+CBAM 三合一），
生成逻辑在 codegen.insert_modules_into_yaml，会同步重映射 head 对 backbone 的索引引用。
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

from ..codegen import find_base_yaml, generate_model_yaml, variant_name
from ..search import github
from .base import StageContext, StageResult


def run(ctx: StageContext) -> StageResult:
    res = StageResult()
    ws = ctx.ws
    cfg = ctx.cfg
    project_root = cfg.project_root

    rows: List[Dict[str, Any]] = []
    artifacts = []

    # --- 1. 本地已有的模型 ---
    wanted: List[str] = []
    if cfg.model.base:
        wanted.append(cfg.model.base)
    wanted.extend(cfg.model.variants)

    for name in wanted:
        local_yaml = find_base_yaml(project_root, name)
        if local_yaml is not None:
            rows.append(
                {
                    "model": name,
                    "origin": "local",
                    "path": str(local_yaml),
                    "repo": "",
                    "stars": None,
                    "note": "复用本地 ultralytics 仓库（含自定义模块）",
                }
            )
            ctx.logger.info(f"模型 {name}: 本地已存在 -> {local_yaml}")
            continue

        candidate = github.resolve_repo(name)
        origin = "known-registry"
        if candidate is None:
            query = f"{name} in:name,description"
            hits = github.search_repositories(query, limit=5)
            if not hits:
                rows.append({"model": name, "origin": "not-found", "path": "", "repo": "", "note": "GitHub 未检索到"})
                ctx.logger.warning(f"模型 {name}: GitHub 未检索到官方仓库")
                continue
            candidate = hits[0]
            origin = "github-search"

        detail = github.get_repo(candidate.full_name)
        if detail:
            candidate = detail

        clone = github.clone_from_candidate(candidate, ws.models)
        rows.append(
            {
                "model": name,
                "origin": origin,
                "path": clone.get("dest", "") if clone.get("ok") else "",
                "repo": candidate.html_url,
                "stars": candidate.stars,
                "license": candidate.license,
                "note": clone.get("message", ""),
            }
        )
        ctx.logger.info(f"模型 {name}: {candidate.html_url} -> {clone.get('message')}")

    # --- 2. 生成组合变体 YAML ---
    generated: List[Dict[str, Any]] = []
    for combo in cfg.model.auto_variants:
        modules = [m for m in combo.replace(" ", "").split("+") if m]
        if not modules:
            continue
        info = generate_model_yaml(
            project_root=project_root,
            base=cfg.model.base,
            modules=modules,
            out_dir=ws.model_configs,
            insert_after_c2f=cfg.model.insert_after_c2f,
        )
        generated.append(info)
        if info.get("ok"):
            ctx.logger.info(f"自动生成变体 {info['name']} -> {info['path']}")
            rows.append(
                {
                    "model": info["name"],
                    "origin": "generated",
                    "path": info["path"],
                    "repo": "",
                    "note": f"由 {cfg.model.base} 插入 {', '.join(info['modules'])} 生成",
                }
            )
            artifacts.append(Path(info["path"]))
        else:
            ctx.logger.warning(f"自动生成变体失败: {info.get('error')}")

    # --- 3. 报告 ---
    lines = ["# 模型来源清单", "", "| 模型 | 来源 | 路径 / 仓库 | 说明 |", "| --- | --- | --- | --- |"]
    for r in rows:
        loc = r.get("path") or r.get("repo") or "-"
        lines.append(f"| {r['model']} | {r['origin']} | {loc} | {r.get('note', '')} |")
    lines.append("")
    report = ws.models / "models_report.md"
    report.write_text("\n".join(lines), encoding="utf-8")
    artifacts.append(report)

    (ws.models / "models.json").write_text(
        json.dumps({"rows": rows, "generated": generated}, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    artifacts.append(ws.models / "models.json")

    res.artifacts.extend(artifacts)
    res.data = {"models": rows, "generated": generated}
    res.message = f"模型准备完成，共 {len(rows)} 项"
    return res
