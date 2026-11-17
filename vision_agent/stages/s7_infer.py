"""S7 · 最优权重推理测试（对应工作流第 8 步，与 S7 可并行）。

做三件事：
    1. 用最优权重在指定划分（默认 test）上跑 val，拿整体 + 分类别指标；
    2. 对测试集图片逐张推理，保存标注图与预测 JSON；
    3. 汇总成 reports/inference_report.md。

执行方式是 subprocess：用「装了 torch 的那个解释器」跑独立脚本，
这样 Agent 自身所在环境不需要 GPU 依赖。
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Any, Dict

from ..codegen import generate_infer_script
from ..envcheck import resolve_training_python
from .base import StageContext, StageResult


def _best_model(ctx: StageContext) -> str:
    if ctx.ws.best_run.exists():
        try:
            data = json.loads(ctx.ws.best_run.read_text(encoding="utf-8"))
            if data.get("best_pt"):
                return str(data["best_pt"])
        except Exception:
            pass
    bests = sorted(ctx.ws.runs.rglob("weights/best.pt"), key=lambda p: p.stat().st_mtime, reverse=True)
    return str(bests[0]) if bests else ""


def _pick_split(ctx: StageContext, requested: str) -> str:
    """选择一个真实存在的划分。

    很多公开数据集只给了 train/val，没有 test；这时硬跑 test 会直接报错。
    这里按 test -> val -> train 的顺序回退，并在报告里如实标注用了哪个划分。
    """
    cfg = ctx.cfg
    root = cfg.resolve(cfg.data.raw_dir) if cfg.data.raw_dir else ctx.ws.data_raw
    processed = ctx.ws.data_processed
    candidates = [requested] + [s for s in ("test", "val", "train") if s != requested]
    for name in candidates:
        for base in (processed, root):
            if (base / "images" / name).exists() and any((base / "images" / name).iterdir()):
                if name != requested:
                    ctx.logger.warning(
                        f"划分 '{requested}' 不存在（原始数据集可能只提供 train/val），回退到 '{name}'"
                    )
                return name
    return requested


def run(ctx: StageContext) -> StageResult:
    res = StageResult()
    ws = ctx.ws
    cfg = ctx.cfg

    model_path = _best_model(ctx)
    if not model_path or not Path(model_path).exists():
        res.ok = False
        res.message = "找不到最优权重（best.pt），请先完成训练阶段"
        return res

    python_exe, _ = resolve_training_python()
    if python_exe is None:
        res.ok = False
        res.skipped = True
        res.message = "没有可用训练环境，无法执行推理测试"
        return res

    out_dir = ws.reports / "inference"
    out_dir.mkdir(parents=True, exist_ok=True)
    script = ws.scripts / f"infer_{cfg.task.name}.py"
    generate_infer_script(
        out_path=script,
        task_name=cfg.task.name,
        project_root=cfg.project_root,
        device=cfg.train.device,
        conf=cfg.app.conf,
        iou=cfg.app.iou,
        default_out=str(out_dir),
    )
    res.add(script)

    requested = str(ctx.opt("split", "test"))
    split = _pick_split(ctx, requested)
    metrics_path = out_dir / "metrics.json"

    if ctx.dry_run:
        ctx.logger.info("dry-run: 跳过推理测试")
        metrics: Dict[str, Any] = {}
    else:
        cmd = [
            str(python_exe),
            str(script),
            "--model",
            model_path,
            "--data",
            str(ws.data_yaml),
            "--split",
            split,
            "--out",
            str(out_dir),
            "--device",
            str(cfg.train.device),
        ]
        log_path = ws.logs / "infer.log"
        ctx.logger.info(f"推理测试: {' '.join(cmd[2:])}")
        with open(log_path, "w", encoding="utf-8", errors="replace") as log:
            log.write("# command: " + " ".join(cmd) + "\n\n")
            log.flush()
            proc = subprocess.run(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                encoding="utf-8",
                errors="replace",
                cwd=str(cfg.project_root if cfg.project_root.exists() else Path.cwd()),
            )
            log.write(proc.stdout or "")
        if proc.returncode != 0:
            ctx.logger.warning(f"推理脚本退出码 {proc.returncode}，详见 {log_path}")
        if metrics_path.exists():
            metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        else:
            metrics = {}

    # --- 报告 ---
    overall = metrics.get("overall", {}) or {}
    per_class = metrics.get("per_class", {}) or {}
    lines = [
        "# 推理测试报告",
        "",
        f"- **权重**: `{model_path}`",
        f"- **数据集划分**: {metrics.get('split', split)}",
        f"- **数据配置**: `{ws.data_yaml}`",
    ]
    if split != requested:
        lines.append(
            f"- **注意**: 原始数据集没有 `{requested}` 划分，本次改用 `{split}` 划分评估"
            "（若需要独立测试集，应在预处理阶段显式切分）"
        )
    lines.append("")
    lines += [
        "## 整体指标",
        "",
        "| 指标 | 数值 |",
        "| --- | ---: |",
        f"| Precision | {_fmt(overall.get('precision'))} |",
        f"| Recall | {_fmt(overall.get('recall'))} |",
        f"| mAP@50 | {_fmt(overall.get('map50'))} |",
        f"| mAP@50-95 | {_fmt(overall.get('map50_95'))} |",
        "",
    ]
    if per_class:
        lines.append("## 分类别指标")
        lines.append("")
        lines.append("| 类别 | mAP@50 | mAP@50-95 |")
        lines.append("| --- | ---: | ---: |")
        for name, m in per_class.items():
            lines.append(f"| {name} | {_fmt(m.get('map50'))} | {_fmt(m.get('map50_95'))} |")
        lines.append("")
    samples = metrics.get("samples") or []
    if samples:
        lines.append("## 样本级预测")
        lines.append("")
        lines.append(f"共保存 {len(samples)} 张标注图，目录: `{metrics.get('sample_dir', '')}`")
        lines.append("")
        lines.append("| 图片 | 目标数 | 类别 |")
        lines.append("| --- | ---: | --- |")
        for s in samples[:20]:
            boxes = s.get("boxes", [])
            names = ", ".join(f"{b.get('class_name')}({b.get('confidence')})" for b in boxes[:5])
            lines.append(f"| {Path(s.get('image', '')).name} | {len(boxes)} | {names or '-'} |")
        lines.append("")
    if metrics.get("val_error"):
        lines.append(f"> 评估出错: {metrics['val_error']}")
        lines.append("")
    if metrics.get("predict_error"):
        lines.append(f"> 推理出错: {metrics['predict_error']}")
        lines.append("")

    (ws.inference_report).write_text("\n".join(lines), encoding="utf-8")
    res.add(ws.inference_report)
    if metrics_path.exists():
        res.add(metrics_path)

    res.data = {"overall": overall, "per_class": per_class, "samples": len(samples)}
    res.message = "推理测试完成" if metrics else "推理测试未产出指标（详见 logs/infer.log）"
    return res


def _fmt(value) -> str:
    if value is None:
        return "-"
    try:
        return f"{float(value):.4f}"
    except (TypeError, ValueError):
        return str(value)
