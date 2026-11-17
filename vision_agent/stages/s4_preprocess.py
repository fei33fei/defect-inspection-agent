"""S4 · 标注转换与数据集构建（对应工作流第 5 步）。

输入可以是 VOC XML / COCO JSON / YOLO txt，输出统一为 YOLO 检测格式：
    processed/images/{train,val,test} + processed/labels/{train,val,test}

随后写 configs/data.yaml 并逐标签校验：
坐标是否在 [0,1]、类别索引是否越界、图签是否一一对应。
校验结果写入 datasets/preprocess_report.md。
"""

from __future__ import annotations

import json
from pathlib import Path

from .. import convert
from .base import StageContext, StageResult


def run(ctx: StageContext) -> StageResult:
    res = StageResult()
    ws = ctx.ws
    cfg = ctx.cfg

    # --- 1. 找到可用的原始数据目录 ---
    raw = ws.data_raw
    has_content = raw.exists() and any(raw.rglob("*"))
    if not has_content and cfg.data.local_dir:
        candidate = cfg.resolve(cfg.data.local_dir)
        if candidate.exists():
            raw = candidate
            ctx.logger.info(f"工作区未收拢到数据，改用原始目录: {raw}")
    if not raw.exists():
        res.ok = False
        res.message = f"找不到原始数据目录: {raw}"
        return res

    # --- 2. 构建数据集 ---
    build = convert.build_dataset(
        raw_dir=raw,
        out_dir=ws.data_processed,
        annotation_format=cfg.data.annotation_format,
        classes=cfg.task.classes or None,
        split=cfg.data.split,
        seed=cfg.data.seed,
        copy_mode=cfg.data.copy_mode,
    )
    if not build.ok:
        res.ok = False
        res.message = "数据集构建失败: " + "; ".join(build.errors[:5])
        (ws.dataset_stats).write_text(
            json.dumps(build.to_dict(), indent=2, ensure_ascii=False), encoding="utf-8"
        )
        res.add(ws.dataset_stats)
        return res

    # --- 3. data.yaml ---
    classes = build.classes or cfg.task.classes
    convert.write_data_yaml(ws.data_processed, ws.data_yaml, classes, name=cfg.task.name)

    # --- 4. 校验 ---
    validation = convert.validate_dataset(ws.data_processed, classes)
    stats = convert.collect_stats(ws.data_processed, classes)
    stats["validation"] = validation
    stats["source_dir"] = str(raw)
    stats["annotation_format"] = build.format
    (ws.dataset_stats).write_text(json.dumps(stats, indent=2, ensure_ascii=False), encoding="utf-8")

    # --- 5. 报告 ---
    lines = [
        "# 数据预处理报告",
        "",
        f"- **源目录**: {raw}",
        f"- **标注格式**: {build.format}",
        f"- **类别数**: {len(classes)} ({', '.join(classes)})",
        f"- **坐标格式**: YOLO 归一化 `cls cx cy w h`（cx,cy 为中心点，w,h 为宽高，范围 0-1）",
        "",
        "## 划分统计",
        "",
        "| 划分 | 图片数 | 标注框数 |",
        "| --- | ---: | ---: |",
    ]
    for split_name, info in stats["splits"].items():
        lines.append(f"| {split_name} | {info['images']} | {info['boxes']} |")
    lines.append(f"| **合计** | **{stats['total_images']}** | **{stats['total_boxes']}** |")
    lines.append("")
    empty = [name for name, info in stats["splits"].items() if info.get("images", 0) == 0]
    if empty:
        lines.append(
            f"> 注意：{'、'.join(empty)} 划分为空 —— 原始数据集自带 train/val 划分，"
            "Agent 默认沿用原划分以保证与历史实验可比；若需要独立测试集，"
            "请在任务配置里把 `data.split` 设为显式比例并清空原始划分目录。"
        )
        lines.append("")
    lines.append("## 类别分布")
    lines.append("")
    lines.append("| 类别 | 标注框数 |")
    lines.append("| --- | ---: |")
    for name in classes:
        lines.append(f"| {name} | {stats['per_class_counts'].get(name, 0)} |")
    lines.append("")
    lines.append("## 校验结果")
    lines.append("")
    lines.append(f"- 校验标签行数: {validation['checked_labels']}")
    lines.append(f"- 问题数: {validation['problem_count']}")
    lines.append(f"- 重复标注行: {validation.get('duplicate_labels', 0)}（ultralytics 训练时会自动去重）")
    if validation.get("duplicate_examples"):
        lines.append("")
        lines.append("重复标注示例：")
        lines.append("")
        for p in validation["duplicate_examples"][:10]:
            lines.append(f"- {p}")
    if validation["problems"]:
        lines.append("")
        lines.append("```")
        for p in validation["problems"][:40]:
            lines.append(p)
        lines.append("```")
    lines.append("")
    if build.warnings:
        lines.append("## 转换告警")
        lines.append("")
        for w in build.warnings[:40]:
            lines.append(f"- {w}")
        lines.append("")

    (ws.preprocess_report).write_text("\n".join(lines), encoding="utf-8")

    res.add(ws.dataset_stats)
    res.add(ws.preprocess_report)
    res.add(ws.data_yaml)
    res.data = {
        "total_images": stats["total_images"],
        "total_boxes": stats["total_boxes"],
        "classes": classes,
        "validation_ok": validation["ok"],
        "data_yaml": str(ws.data_yaml),
    }
    res.message = (
        f"数据集构建完成：{stats['total_images']} 张图 / {stats['total_boxes']} 个框，"
        f"校验问题 {validation['problem_count']} 个"
    )
    return res
