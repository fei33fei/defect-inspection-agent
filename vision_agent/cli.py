"""命令行入口。

    python -m vision_agent init      --task task.yaml
    python -m vision_agent doctor
    python -m vision_agent plan      --task task.yaml
    python -m vision_agent run       --task task.yaml [--stages s4,s5] [--resume] [--dry-run]
    python -m vision_agent status    --task task.yaml
    python -m vision_agent search    --keywords "steel surface defect" [--papers]
    python -m vision_agent serve     --task task.yaml
    python -m vision_agent report    --task task.yaml
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import List, Optional

from .config import ConfigError, load_config, write_template_config
from .envcheck import build_env_report
from .logging_utils import bullet, get_logger, setup_logging, stage_banner
from .pipeline import STAGES, Pipeline, resolve_stage_keys
from .workspace import Workspace


def _add_common(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--task", default="task.yaml", help="任务配置文件路径")
    parser.add_argument("--verbose", action="store_true", help="输出调试日志")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="vision_agent",
        description="视觉检测 Agent：调研 → 取模型 → 取数据 → 预处理 → 训练 → 可视化 → 推理 → 文档",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p_init = sub.add_parser("init", help="生成任务配置模板")
    p_init.add_argument("--task", default="task.yaml")
    p_init.add_argument("--force", action="store_true", help="覆盖已有文件")

    sub.add_parser("doctor", help="检查 Anaconda / PyTorch / GPU 环境")

    p_plan = sub.add_parser("plan", help="只做计划推演，不执行任何阶段")
    _add_common(p_plan)

    p_run = sub.add_parser("run", help="执行流水线")
    _add_common(p_run)
    p_run.add_argument("--stages", nargs="*", default=None, help="阶段列表，如 s4 s5 或 s4-s7")
    p_run.add_argument("--resume", action="store_true", help="跳过已完成的阶段")
    p_run.add_argument("--dry-run", action="store_true", help="只打印将要执行的动作")
    p_run.add_argument("--no-train", action="store_true", help="跳过实际训练")
    p_run.add_argument("--force-train", action="store_true", help="即使 auto_train=false 也训练")
    p_run.add_argument(
        "--reuse-runs",
        action="store_true",
        help="若 run 目录已有 results.csv 就直接复用（只重新解析指标，不重训）",
    )
    p_run.add_argument("--echo-train", action="store_true", help="实时打印训练日志")
    p_run.add_argument("--split", default="test", help="推理测试使用的划分")
    p_run.add_argument("--keep-going", action="store_true", help="某阶段失败后继续执行后续阶段")

    p_status = sub.add_parser("status", help="查看工作区状态")
    _add_common(p_status)

    p_search = sub.add_parser("search", help="单独检索数据集或论文")
    p_search.add_argument("--keywords", nargs="+", required=True)
    p_search.add_argument("--papers", action="store_true", help="检索论文而不是数据集")
    p_search.add_argument("--limit", type=int, default=8)

    p_serve = sub.add_parser("serve", help="启动工作区里生成的可视化系统")
    _add_common(p_serve)
    p_serve.add_argument("--host", default=None)
    p_serve.add_argument("--port", type=int, default=None)
    p_serve.add_argument("--model", default=None)

    p_report = sub.add_parser("report", help="重新生成项目文档")
    _add_common(p_report)

    p_self = sub.add_parser("self-test", help="Agent 自检（不需要 pytest）")
    p_self.add_argument("--verbose", action="store_true", help="打印失败堆栈")

    return parser


# ---------------------------------------------------------------------------
# 各子命令
# ---------------------------------------------------------------------------


def cmd_init(args) -> int:
    out = Path(args.task)
    if out.exists() and not args.force:
        print(f"任务配置已存在: {out}（加 --force 覆盖）")
        return 1
    write_template_config(out)
    print(f"已生成任务配置模板: {out.resolve()}")
    print("下一步：编辑其中的 dataset_type / dataset_keywords / model / data，然后执行")
    print(f"  python -m vision_agent plan --task {out}")
    return 0


def cmd_doctor(args) -> int:
    logger = setup_logging(None, verbose=getattr(args, "verbose", False))
    report = build_env_report()
    print()
    print(report.to_markdown())
    logger.info(f"训练解释器: {report.python_exe or '未找到'}")
    return 0 if report.ready_for_training else 2


def cmd_plan(args) -> int:
    setup_logging(None, verbose=args.verbose)
    cfg = load_config(args.task)
    ws = Workspace(cfg.workspace)
    logger = get_logger("plan")

    print()
    stage_banner("任务计划推演", "只推演，不执行")
    print(f"  任务名称    : {cfg.task.name}")
    print(f"  数据集类型  : {cfg.task.dataset_type}")
    print(f"  目标数据集  : {cfg.task.dataset_name or '-'}")
    print(f"  类别        : {', '.join(cfg.task.classes) or '-'}")
    print(f"  工作目录    : {ws.root}")
    print(f"  项目仓库    : {cfg.project_root}")
    print(f"  基线模型    : {cfg.model.base}")
    print(f"  消融变体    : {', '.join(cfg.model.variants) or '-'}")
    print(f"  组合变体    : {', '.join(cfg.model.auto_variants) or '-'}")
    print()

    print("阶段清单：")
    for spec in STAGES:
        print(f"  {spec.key.upper()}  {spec.title:<20} {spec.subtitle}")
    print()

    # 环境
    env = build_env_report()
    print("环境检查：")
    print(f"  Anaconda   : {env.conda_path or '未检测到'}")
    print(f"  训练解释器 : {env.python_exe or '未找到'}")
    print(f"  PyTorch    : {env.torch_version or '未安装'}")
    print(f"  CUDA       : {'可用' if env.cuda_available else '不可用'} {env.gpu_name}")
    print(f"  可训练     : {'是' if env.ready_for_training else '否'}")
    if env.instructions and not env.ready_for_training:
        print("  需要处理：")
        bullet(env.instructions, indent="    ")
    print()

    # 数据
    print("数据处理：")
    print(f"  来源        : {cfg.data.source}")
    if cfg.data.source == "local" and cfg.data.local_dir:
        local = cfg.resolve(cfg.data.local_dir)
        from .convert import detect_annotation_format

        if local.exists():
            fmt, evidence = detect_annotation_format(local)
            print(f"  本地目录    : {local}  ({evidence})")
            print(f"  探测格式    : {fmt}")
        else:
            print(f"  本地目录    : {local}  (不存在!)")
    elif cfg.data.source == "huggingface":
        print(f"  HF 仓库     : {cfg.data.hf_repo}")
    elif cfg.data.source == "url":
        for u in cfg.data.urls:
            print(f"  下载链接    : {u}")
    elif cfg.data.source == "kaggle":
        print(f"  Kaggle 数据集: {cfg.data.kaggle_dataset} (需手动下载)")
    print(f"  切分比例    : {cfg.data.split}")
    print()

    # 消融矩阵
    from .stages.s5_train import build_ablation_plan
    from .stages.base import StageContext
    from .llm import LLMClient
    from .state import PipelineState

    ctx = StageContext(
        cfg=cfg,
        ws=ws,
        state=PipelineState(ws.state_file),
        llm=LLMClient(cfg.llm),
        logger=logger,
    )
    try:
        plan = build_ablation_plan(ctx)
        print("消融矩阵：")
        for entry in plan:
            mark = "  (待 S2 生成)" if entry.get("pending") else ""
            print(f"  - {entry['name']:<20} {entry['kind']:<10} {entry['model']}{mark}")
        if not plan:
            print("  (空 — 可能是基线/变体模型文件没找到)")
    except Exception as exc:  # noqa: BLE001
        print(f"消融矩阵推演失败: {exc}")
    print()

    print("下一步：")
    print(f"  python -m vision_agent run --task {cfg.config_path}")
    print(f"  python -m vision_agent run --task {cfg.config_path} --dry-run   # 只打印将执行的动作")
    print()
    return 0


def cmd_run(args) -> int:
    cfg = load_config(args.task)
    ws = Workspace(cfg.workspace)
    logger = setup_logging(ws.logs, verbose=args.verbose)

    options = {
        "dry_run": args.dry_run,
        "force_train": args.force_train,
        "echo_train": args.echo_train,
        "split": args.split,
        "reuse_runs": getattr(args, "reuse_runs", False),
    }
    if args.no_train:
        options["force_train"] = False
        cfg.train.auto_train = False

    pipeline = Pipeline(cfg, options=options)
    summary = pipeline.run(
        stage_keys=args.stages,
        resume=args.resume,
        stop_on_error=not args.keep_going,
    )
    logger.info("=" * 78)
    for row in pipeline.status_table():
        logger.info(f"  {row['stage']:<4} {row['status']:<9} {row['note'][:70]}")
    logger.info(f"工作区: {ws.root}")
    return 0 if summary["ok"] else 1


def cmd_status(args) -> int:
    cfg = load_config(args.task)
    ws = Workspace(cfg.workspace)
    if not ws.state_file.exists():
        print(f"尚未运行过，工作区: {ws.root}")
        return 0
    pipeline = Pipeline(cfg)
    rows = pipeline.status_table()
    print()
    print(f"{'阶段':<6}{'状态':<10}{'开始':<22}{'结束':<22}说明")
    for r in rows:
        print(f"{r['stage']:<6}{r['status']:<10}{r['started_at']:<22}{r['ended_at']:<22}{r['note'][:50]}")
    print()
    return 0


def cmd_search(args) -> int:
    setup_logging(None, verbose=args.verbose)
    from .search import datasets as ds
    from .search import papers as ps

    if args.papers:
        hits = ps.search_all(args.keywords, limit=args.limit)
        for p in hits:
            print(f"[{p.source}] {p.title} ({p.year or '?'})")
            print(f"    {p.url}")
    else:
        hits = ds.search_all(args.keywords, limit=args.limit)
        for d in hits:
            print(f"[{d.source}] {d.name}")
            print(f"    {d.url}  license={d.license or '-'}")
    return 0


def cmd_serve(args) -> int:
    import subprocess

    cfg = load_config(args.task)
    ws = Workspace(cfg.workspace)
    app_script = ws.scripts / f"app_{cfg.task.name}.py"
    if not app_script.exists():
        print(f"还没生成可视化系统脚本: {app_script}")
        print("先执行: python -m vision_agent run --task <task.yaml> --stages s5 s6")
        return 1

    model = args.model
    if not model and ws.best_run.exists():
        try:
            model = json.loads(ws.best_run.read_text(encoding="utf-8")).get("best_pt")
        except Exception:
            model = None

    cmd = [sys.executable, str(app_script)]
    if model:
        cmd += ["--model", str(model)]
    cmd += ["--host", args.host or cfg.app.host, "--port", str(args.port or cfg.app.port)]
    print("启动可视化系统:", " ".join(cmd))
    try:
        return subprocess.call(cmd, cwd=str(ws.root))
    except KeyboardInterrupt:
        return 0


def cmd_report(args) -> int:
    cfg = load_config(args.task)
    ws = Workspace(cfg.workspace)
    logger = setup_logging(ws.logs, verbose=args.verbose)
    pipeline = Pipeline(cfg)
    summary = pipeline.run(stage_keys=["s8"], resume=False)
    logger.info(f"报告位置: {ws.project_report}")
    return 0 if summary["ok"] else 1


def cmd_self_test(args) -> int:
    from .selftest import run_selftest

    ok, checks = run_selftest(verbose=getattr(args, "verbose", False))
    print()
    for c in checks:
        print(f"  [{'PASS' if c.ok else 'FAIL'}] {c.name}")
        print(f"         {c.message}")
    print()
    passed = sum(1 for c in checks if c.ok)
    print(f"自检结果：{passed}/{len(checks)} 通过")
    return 0 if ok else 1


def main(argv: Optional[List[str]] = None) -> int:
    # Windows 控制台默认 GBK，中文/符号可能编码失败；统一转 UTF-8 并容错
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, ValueError):
            pass

    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        if args.command == "init":
            return cmd_init(args)
        if args.command == "doctor":
            return cmd_doctor(args)
        if args.command == "plan":
            return cmd_plan(args)
        if args.command == "run":
            return cmd_run(args)
        if args.command == "status":
            return cmd_status(args)
        if args.command == "search":
            return cmd_search(args)
        if args.command == "serve":
            return cmd_serve(args)
        if args.command == "report":
            return cmd_report(args)
        if args.command == "self-test":
            return cmd_self_test(args)
    except ConfigError as exc:
        print(f"配置错误: {exc}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print("\n已中断", file=sys.stderr)
        return 130
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
