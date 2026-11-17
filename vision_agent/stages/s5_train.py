"""S5 · 训练脚本生成 + 消融实验自动执行（对应工作流第 6 步）。

流程：
    1. 环境自检：找能跑 PyTorch 的解释器、确认 GPU；
       没有 Anaconda / torch 时按用户要求「提醒用户自己装一个 anaconda」，
       把安装步骤写进 logs/env_report.md 并中止本阶段；
    2. 生成训练脚本（scripts/train_<task>.py），优先复用本地 ultralytics 仓库，
       这样 SE/EMA/CBAM 等自定义模块可直接用；
    3. 组装消融矩阵：基线 + 单模块变体 + 组合变体，每个变体独立 run 目录；
    4. 逐个调用终端训练（subprocess），日志实时落盘到 logs/train_<variant>.log；
    5. 解析每个 run 的 results.csv，取 mAP50-95 最高的一轮作为最佳 epoch，
       汇总成 runs/ablation_summary.md / .csv，并选出全局最优权重。

为什么要「基线也跑一遍」：若基线用 COCO 预训练而变体随机初始化，
指标差异不能归因于注意力模块。Agent 会把这一点写进汇总表的读数须知。
"""

from __future__ import annotations

import csv
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from ..codegen import find_base_yaml, generate_train_script, variant_name
from ..envcheck import build_env_report, resolve_training_python, suggest_batch
from .base import StageContext, StageResult


# ---------------------------------------------------------------------------
# 消融矩阵
# ---------------------------------------------------------------------------


def _find_pretrained(cfg, project_root: Path) -> Optional[Path]:
    """找预训练权重 .pt。"""
    name = cfg.model.pretrained
    if not name:
        return None
    candidates = [
        project_root / name,
        project_root.parent / name,
        Path.cwd() / name,
        cfg.base_dir / name,
    ]
    for cand in candidates:
        if cand.exists():
            return cand
    return None


def _resolve_variant_model(name: str, ctx: StageContext) -> Optional[Path]:
    """定位变体的模型 YAML。"""
    cfg = ctx.cfg
    local = find_base_yaml(cfg.project_root, name)
    if local is not None:
        return local
    generated = ctx.ws.model_configs / f"{name}.yaml"
    if generated.exists():
        return generated
    # 在克隆下来的官方仓库里找
    for found in ctx.ws.models.rglob(f"{name}.yaml"):
        return found
    return None


def build_ablation_plan(ctx: StageContext) -> List[Dict[str, Any]]:
    """组装消融实验矩阵。"""
    cfg = ctx.cfg
    plan: List[Dict[str, Any]] = []

    pretrained = _find_pretrained(cfg, cfg.project_root)
    base_yaml = find_base_yaml(cfg.project_root, cfg.model.base)
    baseline_model = pretrained or base_yaml
    if baseline_model is not None:
        plan.append(
            {
                "key": "baseline",
                "name": "baseline",
                "model": str(baseline_model),
                "kind": "基线",
                "init": "COCO 预训练" if pretrained else "随机初始化",
                "modules": [],
            }
        )
    else:
        ctx.logger.warning(f"基线模型未找到（{cfg.model.base}），消融矩阵里不含基线")

    for variant in cfg.model.variants:
        model = _resolve_variant_model(variant, ctx)
        if model is None:
            ctx.logger.warning(f"变体 {variant} 的模型文件未找到，已跳过")
            continue
        plan.append(
            {
                "key": variant,
                "name": variant,
                "model": str(model),
                "kind": "单模块变体",
                "init": "随机初始化",
                "modules": [variant.split("-")[-1]],
            }
        )

    for combo in cfg.model.auto_variants:
        modules = [m for m in combo.replace(" ", "").split("+") if m]
        name = variant_name(cfg.model.base, modules)
        model = ctx.ws.model_configs / f"{name}.yaml"
        pending = False
        if not model.exists():
            model = _resolve_variant_model(name, ctx)
        if model is None:
            # 还没生成，但基座结构在本地 → S2 阶段会生成，这里先占位
            if find_base_yaml(cfg.project_root, cfg.model.base) is not None:
                pending = True
                model = ctx.ws.model_configs / f"{name}.yaml"
            else:
                ctx.logger.warning(f"组合变体 {name} 既未生成也找不到基座，跳过")
                continue
        plan.append(
            {
                "key": combo,
                "name": name,
                "model": str(model),
                "kind": "组合变体",
                "init": "随机初始化",
                "modules": modules,
                "pending": pending,
            }
        )

    # 用户在 train.ablations 里显式指定时做过滤
    wanted = [t.strip() for t in cfg.train.ablations if t and t.strip()]
    if wanted:
        def keep(entry: Dict[str, Any]) -> bool:
            for token in wanted:
                if token == entry["key"] or token == entry["name"]:
                    return True
                if token.lower() in [m.lower() for m in entry["modules"]]:
                    return True
            return False

        filtered = [e for e in plan if keep(e)]
        if filtered:
            plan = filtered

    return plan


# ---------------------------------------------------------------------------
# 训练执行
# ---------------------------------------------------------------------------


def _run_training(
    ctx: StageContext,
    entry: Dict[str, Any],
    python_exe: Path,
    script: Path,
    data_yaml: Path,
    project_dir: Path,
    batch: Optional[int] = None,
) -> Dict[str, Any]:
    """执行一次训练。返回该实验的结果摘要。

    带 OOM 自适应：训练日志里出现 "out of memory" 时，自动把 batch 减半重试
    （最多 2 次），并把调整过程记进摘要，避免长训练一崩就全白跑。
    """
    run_name = entry["name"]
    log_path = ctx.ws.logs / f"train_{run_name}.log"
    run_dir = project_dir / run_name

    # 复用已有 run：目录里已经有 results.csv 就不重训，只重新解析
    if ctx.opt("reuse_runs", False) and (run_dir / "results.csv").exists():
        ctx.logger.info(f"[{run_name}] 复用已有 run: {run_dir}")
        summary = _parse_run(run_dir)
        summary.update(
            {
                "name": run_name,
                "kind": entry["kind"],
                "model": entry["model"],
                "init": entry["init"],
                "modules": entry["modules"],
                "return_code": 0,
                "duration_sec": 0.0,
                "batch": ctx.cfg.train.batch,
                "log": str(log_path),
                "run_dir": str(run_dir),
                "reused": True,
            }
        )
        return summary

    current_batch = batch if batch else ctx.cfg.train.batch
    adjustments: List[Dict[str, Any]] = []
    summary: Dict[str, Any] = {}

    for attempt in range(3):
        cmd = [
            str(python_exe),
            str(script),
            "--model",
            entry["model"],
            "--data",
            str(data_yaml),
            "--epochs",
            str(ctx.cfg.train.epochs),
            "--batch",
            str(current_batch),
            "--imgsz",
            str(ctx.cfg.train.imgsz),
            "--device",
            str(ctx.cfg.train.device),
            "--workers",
            str(ctx.cfg.train.workers),
            "--patience",
            str(ctx.cfg.train.patience),
            "--optimizer",
            str(ctx.cfg.train.optimizer),
            "--lr0",
            str(ctx.cfg.train.lr0),
            "--seed",
            str(ctx.cfg.train.seed),
            "--project",
            str(project_dir),
            "--name",
            run_name,
        ]
        ctx.logger.info(f"[{run_name}] {python_exe.name} {' '.join(cmd[2:])}")

        started = time.time()
        with open(log_path, "w", encoding="utf-8", errors="replace") as log:
            log.write("# command: " + " ".join(cmd) + "\n\n")
            log.flush()
            try:
                proc = subprocess.Popen(
                    cmd,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    encoding="utf-8",
                    errors="replace",
                    bufsize=1,
                    cwd=str(ctx.cfg.project_root if ctx.cfg.project_root.exists() else Path.cwd()),
                )
                assert proc.stdout is not None
                for line in proc.stdout:
                    log.write(line)
                    log.flush()
                    if ctx.opt("echo_train", False):
                        print(line.rstrip(), flush=True)
                proc.wait()
                code = proc.returncode
            except Exception as exc:  # noqa: BLE001
                code = -1
                log.write(f"# 训练进程异常: {exc}\n")

        duration = time.time() - started
        run_dir = project_dir / run_name
        summary = _parse_run(run_dir)
        summary.update(
            {
                "name": run_name,
                "kind": entry["kind"],
                "model": entry["model"],
                "init": entry["init"],
                "modules": entry["modules"],
                "return_code": code,
                "duration_sec": round(duration, 1),
                "batch": current_batch,
                "log": str(log_path),
                "run_dir": str(run_dir),
                "batch_adjustments": adjustments,
            }
        )

        # OOM 自适应：显存不够就把 batch 减半重试
        if _looks_like_oom(log_path) and attempt < 2 and current_batch > 2:
            new_batch = max(2, current_batch // 2)
            adjustments.append(
                {"from": current_batch, "to": new_batch, "reason": "CUDA out of memory"}
            )
            ctx.logger.warning(
                f"[{run_name}] 显存不足（CUDA out of memory），batch {current_batch} -> {new_batch} 后重试"
            )
            current_batch = new_batch
            continue
        break

    return summary


def _looks_like_oom(log_path: Path) -> bool:
    """判断训练日志里是否出现了显存不足。"""
    try:
        text = Path(log_path).read_text(encoding="utf-8", errors="replace")
    except OSError:
        return False
    lowered = text.lower()
    return "out of memory" in lowered or "cuda error: out of memory" in lowered


def _parse_run(run_dir: Path) -> Dict[str, Any]:
    """解析 run 目录里的 results.csv，取 mAP50-95 最高的一轮。

    注意：ultralytics 写出的 CSV 表头是**带对齐空格**的（例如
    `"                 epoch,         metrics/mAP50-95(B),"`），
    所以这里不能用 csv.DictReader 直接按列名取，要先 strip 规范化表头。
    """
    out: Dict[str, Any] = {}
    results_csv = Path(run_dir) / "results.csv"
    if not results_csv.exists():
        out["error"] = "results.csv 不存在"
        return out

    rows: List[Dict[str, str]] = []
    try:
        with open(results_csv, "r", encoding="utf-8-sig", newline="") as fh:
            reader = csv.reader(fh)
            header = next(reader, None)
            if header is None:
                out["error"] = "results.csv 为空"
                return out
            header = [h.strip() for h in header]
            for raw in reader:
                if not raw or not raw[0].strip():
                    continue
                rows.append(
                    {header[i]: (raw[i].strip() if i < len(raw) else "") for i in range(len(header))}
                )
    except (OSError, csv.Error) as exc:
        out["error"] = f"results.csv 读取失败: {exc}"
        return out

    if not rows:
        out["error"] = "results.csv 无有效数据行"
        return out

    def pick(row: Dict[str, str], key: str) -> Optional[float]:
        val = row.get(key)
        if val is None or val == "":
            return None
        try:
            return float(val)
        except (TypeError, ValueError):
            return None

    best = None
    for row in rows:
        m95 = pick(row, "metrics/mAP50-95(B)")
        if m95 is None:
            continue
        if best is None or m95 > best["map50_95"]:
            try:
                epoch = int(float(row.get("epoch", 0) or 0))
            except (TypeError, ValueError):
                epoch = 0
            best = {
                "epoch": epoch,
                "precision": pick(row, "metrics/precision(B)"),
                "recall": pick(row, "metrics/recall(B)"),
                "map50": pick(row, "metrics/mAP50(B)"),
                "map50_95": m95,
            }
    out["best_epoch"] = best
    out["epochs_run"] = len(rows)
    best_pt = Path(run_dir) / "weights" / "best.pt"
    if best_pt.exists():
        out["best_pt"] = str(best_pt)
        out["best_pt_bytes"] = best_pt.stat().st_size
    if best is None:
        out["error"] = "results.csv 里没有找到 metrics/mAP50-95(B) 列"
    return out


# ---------------------------------------------------------------------------
# 汇总
# ---------------------------------------------------------------------------


def _write_ablation_summary(ctx: StageContext, results: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    """写消融汇总表，返回全局最优实验。"""
    valid = [r for r in results if r.get("best_epoch")]
    lines = [
        "# 消融实验汇总",
        "",
        f"> 数据来源：各 run 目录的 `results.csv`（取 mAP50-95 最高的一轮）。"
        f"由 vision_agent 生成于 {time.strftime('%Y-%m-%d %H:%M:%S')}。",
        "",
        "| 实验 | 类型 | 权重初始化 | batch | 实际轮数 | 最佳 epoch | Precision | Recall | mAP50 | mAP50-95 | 权重体积 |",
        "|:--|:--|:--|--:|--:|--:|--:|--:|--:|--:|--:|",
    ]
    csv_rows = []
    for r in results:
        be = r.get("best_epoch") or {}
        size_mb = ""
        if r.get("best_pt_bytes"):
            size_mb = f"{r['best_pt_bytes'] / 1024:.0f} KB"
        lines.append(
            "| {name} | {kind} | {init} | {batch} | {epochs} | {epoch} | {p} | {rc} | {m50} | {m95} | {size} |".format(
                name=r.get("name", "-"),
                kind=r.get("kind", "-"),
                init=r.get("init", "-"),
                batch=r.get("batch", "-"),
                epochs=r.get("epochs_run", "-"),
                epoch=be.get("epoch", "-"),
                p=f"{be['precision']:.4f}" if be.get("precision") is not None else "-",
                rc=f"{be['recall']:.4f}" if be.get("recall") is not None else "-",
                m50=f"{be['map50']:.4f}" if be.get("map50") is not None else "-",
                m95=f"{be['map50_95']:.4f}" if be.get("map50_95") is not None else "-",
                size=size_mb or "-",
            )
        )
        csv_rows.append(
            {
                "name": r.get("name"),
                "kind": r.get("kind"),
                "init": r.get("init"),
                "batch": r.get("batch"),
                "epochs_run": r.get("epochs_run"),
                "best_epoch": be.get("epoch"),
                "precision": be.get("precision"),
                "recall": be.get("recall"),
                "map50": be.get("map50"),
                "map50_95": be.get("map50_95"),
                "best_pt": r.get("best_pt"),
            }
        )

    lines.append("")
    lines.append("## 读数须知")
    lines.append("")
    lines.append(
        "1. 权重初始化必须一致才可归因：若基线用 COCO 预训练、变体随机初始化，"
        "指标差异同时包含「预训练收益」与「模块收益」。"
    )
    lines.append(
        "2. 早停 patience 与最大轮数应在同一组实验内保持一致，否则「实际轮数」不可比。"
    )
    lines.append("3. 参数量/GFLOPs/FPS 需另行用 `model.info()` 或实测获得。")
    oom_notes = [
        f"{r.get('name')}: batch {a.get('from')} → {a.get('to')}（{a.get('reason')}）"
        for r in results
        for a in (r.get("batch_adjustments") or [])
    ]
    if oom_notes:
        lines.append("4. 训练期间发生过显存不足自动降 batch：" + "；".join(oom_notes))
    lines.append("")

    ctx.ws.ablation_md.write_text("\n".join(lines), encoding="utf-8")
    with open(ctx.ws.ablation_csv, "w", encoding="utf-8-sig", newline="") as fh:
        if csv_rows:
            writer = csv.DictWriter(fh, fieldnames=list(csv_rows[0].keys()))
            writer.writeheader()
            writer.writerows(csv_rows)

    if not valid:
        return None
    best = max(valid, key=lambda r: r["best_epoch"].get("map50_95") or 0.0)
    payload = {
        "name": best["name"],
        "model": best["model"],
        "best_pt": best.get("best_pt"),
        "run_dir": best.get("run_dir"),
        "metrics": best["best_epoch"],
        "kind": best.get("kind"),
        "init": best.get("init"),
    }
    ctx.ws.best_run.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return payload


# ---------------------------------------------------------------------------
# 阶段入口
# ---------------------------------------------------------------------------


def run(ctx: StageContext) -> StageResult:
    res = StageResult()
    ws = ctx.ws
    cfg = ctx.cfg

    # --- 1. 环境自检 ---
    env_report = build_env_report()
    env_md = ws.logs / "env_report.md"
    env_md.write_text(env_report.to_markdown(), encoding="utf-8")
    res.add(env_md)
    ctx.logger.info(
        f"环境: python={env_report.python_exe} torch={env_report.torch_version} "
        f"cuda={env_report.cuda_available} gpu={env_report.gpu_name or '-'}"
    )

    python_exe, _info = resolve_training_python()
    if python_exe is None or not env_report.ready_for_training:
        res.ok = False
        res.skipped = True
        res.message = "训练环境未就绪，请按 logs/env_report.md 的步骤安装后重跑"
        res.data = {"env": env_report.to_dict()}
        return res

    # --- 2. 生成训练脚本 ---
    script = ws.scripts / f"train_{cfg.task.name}.py"
    generate_train_script(
        out_path=script,
        task_name=cfg.task.name,
        project_root=cfg.project_root,
        default_model=cfg.model.base,
        default_data=str(ws.data_yaml),
        epochs=cfg.train.epochs,
        batch=cfg.train.batch,
        imgsz=cfg.train.imgsz,
        device=cfg.train.device,
        workers=cfg.train.workers,
        patience=cfg.train.patience,
        optimizer=cfg.train.optimizer,
        lr0=cfg.train.lr0,
        seed=cfg.train.seed,
        project=str(ws.runs),
        name="baseline",
    )
    res.add(script)

    # --- 3. 消融矩阵 ---
    plan = build_ablation_plan(ctx)
    if not plan:
        res.ok = False
        res.message = "消融矩阵为空：既没找到基线模型，也没找到变体模型"
        return res
    ctx.logger.info(f"消融矩阵: {[e['name'] for e in plan]}")

    if not ws.data_yaml.exists():
        res.ok = False
        res.message = f"缺少 data.yaml（{ws.data_yaml}），请先执行预处理阶段"
        return res

    # --- 4. 执行训练 ---
    # batch <= 0 表示「按显存自动估算」
    batch = ctx.cfg.train.batch
    suggested = suggest_batch(env_report.gpu_vram_mb, ctx.cfg.train.imgsz, cfg.model.base)
    if batch and batch > 0:
        if suggested and batch > suggested:
            ctx.logger.warning(
                f"配置的 batch={batch} 高于按 {env_report.gpu_vram_mb}MB 显存估算的 {suggested}，"
                "可能会 OOM（Agent 已开启 OOM 自动降 batch 重试）"
            )
    else:
        batch = suggested or 16
        ctx.logger.info(f"按 {env_report.gpu_vram_mb}MB 显存自动选择 batch={batch}")

    results: List[Dict[str, Any]] = []
    if ctx.dry_run:
        ctx.logger.info("dry-run: 跳过实际训练，只打印命令")
        for entry in plan:
            ctx.logger.info(
                f"[dry-run][{entry['name']}] {python_exe} {script.name} --model {entry['model']} "
                f"--data {ws.data_yaml} --project {ws.runs} --batch {batch}"
            )
            results.append({**entry, "epochs_run": 0, "best_epoch": None, "batch": batch})
    elif not cfg.train.auto_train and not ctx.opt("force_train", False):
        ctx.logger.info("train.auto_train=false，跳过训练（用 --force-train 可强制执行）")
        results = [{**entry, "epochs_run": 0, "best_epoch": None, "batch": batch} for entry in plan]
    else:
        for entry in plan:
            ctx.logger.info(f"开始训练 [{entry['name']}] ...")
            try:
                summary = _run_training(ctx, entry, python_exe, script, ws.data_yaml, ws.runs, batch=batch)
            except Exception as exc:  # noqa: BLE001
                summary = {**entry, "error": str(exc), "best_epoch": None}
            results.append(summary)
            be = summary.get("best_epoch") or {}
            ctx.logger.info(
                f"完成 [{entry['name']}] 用时 {summary.get('duration_sec', 0)}s "
                f"mAP50-95={be.get('map50_95', 'N/A')}"
            )

    # --- 5. 汇总 ---
    best = _write_ablation_summary(ctx, results)
    res.add(ws.ablation_md)
    res.add(ws.ablation_csv)
    if best:
        res.add(ws.best_run)

    res.data = {
        "best": best,
        "experiments": [{"name": r.get("name"), "metrics": r.get("best_epoch")} for r in results],
        "env": env_report.to_dict(),
    }
    res.message = f"训练阶段完成，最优实验: {best['name'] if best else '无'}"
    return res
