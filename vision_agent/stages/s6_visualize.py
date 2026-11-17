"""S6 · 可视化检测系统（对应工作流第 7 步）。

产出两个东西：
    1. scripts/app_<task>.py —— 自包含的 Flask 应用
       （上传图片 → 检测 → 预览 + 下载；摄像头/外接摄像头 MJPEG 视频流）
    2. reports/app_usage.md —— 使用说明

设计取舍：Agent **不**默认把服务跑起来。
视频流要占摄像头、占端口，属于「需要用户在场」的交互步骤，
所以默认只生成代码与说明，由用户手动启动（也可以加 --start-app 让 Agent 代跑）。
"""

from __future__ import annotations

import json
from pathlib import Path

from ..codegen import generate_app_script
from .base import StageContext, StageResult


def _best_model(ctx: StageContext) -> str:
    """取最优权重路径。"""
    if ctx.ws.best_run.exists():
        try:
            data = json.loads(ctx.ws.best_run.read_text(encoding="utf-8"))
            if data.get("best_pt"):
                return str(data["best_pt"])
        except Exception:
            pass
    # 退而求其次：在 runs 里找任意 best.pt
    bests = sorted(ctx.ws.runs.rglob("weights/best.pt"), key=lambda p: p.stat().st_mtime, reverse=True)
    return str(bests[0]) if bests else ""


def run(ctx: StageContext) -> StageResult:
    res = StageResult()
    ws = ctx.ws
    cfg = ctx.cfg

    model_path = _best_model(ctx)
    app_script = ws.scripts / f"app_{cfg.task.name}.py"
    generate_app_script(
        out_path=app_script,
        task_name=cfg.task.name,
        model_path=model_path,
        conf=cfg.app.conf,
        iou=cfg.app.iou,
        camera_default=cfg.app.camera_default,
        host=cfg.app.host,
        port=cfg.app.port,
        title=cfg.app.title,
        upload_dir=str(ws.app_uploads),
        output_dir=str(ws.app_outputs),
    )
    res.add(app_script)

    # --- 使用说明 ---
    lines = [
        f"# 可视化检测系统 · 使用说明",
        "",
        f"- **脚本**: `{app_script.name}`",
        f"- **权重**: `{model_path or '（尚未训练出 best.pt）'}`",
        f"- **地址**: http://{cfg.app.host}:{cfg.app.port}/",
        "",
        "## 启动",
        "",
        "```bash",
        f'python "{app_script}" --model "{model_path}" --host {cfg.app.host} --port {cfg.app.port}',
        "```",
        "",
        "## 功能",
        "",
        "| 页面 | 路径 | 说明 |",
        "| --- | --- | --- |",
        "| 图片检测 | `/` | 上传图片 → 标注结果预览 → 下载标注图 |",
        "| 摄像头检测 | `/camera` | 选择摄像头序号或视频文件，MJPEG 实时流 |",
        "| 检测接口 | `/api/predict` | POST 字段 `image`，返回 JSON（框、置信度、耗时） |",
        "| 摄像头列表 | `/api/cameras` | 探测可用的摄像头序号 |",
        "| 状态检查 | `/health` | 模型是否加载成功 |",
        "",
        "## 摄像头说明",
        "",
        "- 内置摄像头通常是 `0`，外接 USB 摄像头通常是 `1`、`2`……；",
        "- 页面上的下拉框会先探测一遍可用设备；",
        "- 也可以直接选择视频文件做流式检测；",
        "- 需要 `opencv-python`（`pip install opencv-python`）；",
        "- 浏览器若提示摄像头权限，请允许访问。",
        "",
        "## 依赖",
        "",
        "```bash",
        "pip install flask opencv-python",
        "```",
        "",
        "## 常见问题",
        "",
        "| 现象 | 原因与处理 |",
        "| --- | --- |",
        "| 页面报「未安装 ultralytics」 | 用装了 torch+ultralytics 的解释器启动本脚本 |",
        "| 摄像头打不开 | 换序号；确认没被其他程序占用；确认 opencv 已装 |",
        "| 检测很慢 | 调低 --conf 无关，主要是输入尺寸与设备；可用 CUDA 加速 |",
        "",
    ]
    (ws.app_usage).write_text("\n".join(lines), encoding="utf-8")
    res.add(ws.app_usage)

    res.data = {"app_script": str(app_script), "model": model_path}
    res.message = f"可视化系统已生成: {app_script.name}（需用户手动启动）"
    return res
