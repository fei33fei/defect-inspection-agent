"""S0 · 任务受理与工作区初始化。

对应工作流第 1 步：用户给出「数据集类型 + 选用哪个模型 + 工作文件夹」，
Agent 据此建立工作区，把配置快照落盘，并生成一份「本次要做什么」的说明。
"""

from __future__ import annotations

import shutil
from pathlib import Path

from .base import StageContext, StageResult


def run(ctx: StageContext) -> StageResult:
    res = StageResult()
    ws = ctx.ws
    ws.ensure()

    # 配置快照（保证工作区自包含、可复现）
    if ctx.cfg.config_path.exists():
        shutil.copy2(ctx.cfg.config_path, ws.task_file)

    cfg = ctx.cfg
    lines = [
        f"# {cfg.task.name}",
        "",
        f"- **数据集类型**: {cfg.task.dataset_type or '-'}",
        f"- **目标数据集**: {cfg.task.dataset_name or '-'}",
        f"- **基线模型**: {cfg.model.base}",
        f"- **消融变体**: {', '.join(cfg.model.variants) or '-'}",
        f"- **自动组合变体**: {', '.join(cfg.model.auto_variants) or '-'}",
        f"- **类别数**: {len(cfg.task.classes)}",
        f"- **本地 ultralytics 仓库**: {cfg.project_root}",
        f"- **训练超参**: epochs={cfg.train.epochs} batch={cfg.train.batch} imgsz={cfg.train.imgsz} device={cfg.train.device}",
        "",
        "## 目录说明",
        "",
        "| 目录 | 内容 |",
        "| --- | --- |",
        "| `research/` | S1 数据集与论文调研 |",
        "| `models/` | S2 克隆的官方模型仓库 |",
        "| `datasets/` | S3/S4 原始数据与转换后的 YOLO 数据集 |",
        "| `configs/` | S4/S5 生成的 data.yaml 与模型 YAML |",
        "| `scripts/` | S5-S7 生成的训练/推理/可视化脚本 |",
        "| `runs/` | S5 训练输出与消融汇总 |",
        "| `reports/` | S7/S8 推理报告与项目文档 |",
        "| `app/` | 可视化系统运行时目录（上传与检测结果） |",
        "| `logs/` | 运行日志 |",
        "",
        "## 复现命令",
        "",
        "```bash",
        f"python -m vision_agent run --task {ws.task_file.name}",
        "```",
        "",
    ]
    ws.readme.write_text("\n".join(lines), encoding="utf-8")

    res.add(ws.task_file)
    res.add(ws.readme)
    res.data = {
        "workspace": str(ws.root),
        "classes": cfg.task.classes,
        "model_base": cfg.model.base,
    }
    res.message = f"工作区就绪: {ws.root}"
    return res
