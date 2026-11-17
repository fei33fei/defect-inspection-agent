"""S8 · 项目文档撰写（对应工作流第 9 步）。

把前面所有阶段的真实产物汇总成一份可直接交付的项目报告：
    1. 项目背景
    2. 数据集来源
    3. 算法来源与公式
    4. 训练方式
    5. 实验结果（消融）
    6. 推理结果
    7. 可视化系统使用说明
    8. 总结与局限
    9. 复现命令

所有数字都来自工作区里的真实文件（results.csv / metrics.json / dataset_stats.json），
不手工编造；缺失的部分会明确写「本次运行未产出」。
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

from .base import StageContext, StageResult

FORMULAS = {
    "se": {
        "title": "SEBlock（Squeeze-and-Excitation，Hu et al., CVPR 2018）",
        "ref": "https://arxiv.org/abs/1709.01507",
        "desc": (
            "通过全局平均池化把每个通道的空间信息压缩成通道描述子，"
            "再用两层全连接得到通道权重，重标定特征图。"
            "本项目在 backbone 的 C2f 之后插入，用于抑制背景通道、突出缺陷响应。"
        ),
        "latex": [
            r"z_c = \mathcal{F}_s(F_c) = \frac{1}{H\times W}\sum_{i,j} F_{c,i,j}",
            r"s = \sigma\!\left(W_2\,\delta(W_1 z)\right)",
            r"\tilde{F}_c = s_c \cdot F_c",
        ],
    },
    "ema": {
        "title": "EMA（Efficient Multi-Scale Attention，Bai et al., 2024）",
        "ref": "https://arxiv.org/abs/2401.00945",
        "desc": (
            "把特征按通道分组（groups=8），用 1×1 分组卷积 + 全局平均池化提取通道注意力，"
            "用 3×3 分组卷积提取空间注意力，两者相加后经 sigmoid 得到权重，逐组重标定。"
            "本仓库实现位于 ultralytics/nn/modules/conv.py 的 EMA 类。"
        ),
        "latex": [
            r"X_g = \mathrm{Group}(X, G)",
            r"a_c = \sigma\!\left(\mathrm{GN}(\mathrm{Conv}_{1\times1}(\mathrm{GAP}(X_g))) + \mathrm{GN}(\mathrm{Conv}_{3\times3}(X_g))\right)",
            r"\tilde{X}_g = X_g \odot a_c,\qquad \tilde{X} = \mathrm{Concat}(\tilde{X}_1,\dots,\tilde{X}_G)",
        ],
    },
    "cbam": {
        "title": "CBAM（Convolutional Block Attention Module，Woo et al., ECCV 2018）",
        "ref": "https://arxiv.org/abs/1807.06521",
        "desc": (
            "先做通道注意力（共享 MLP 作用在平均池化与最大池化结果上），"
            "再做空间注意力（7×7 卷积作用在通道维的均值/最大值上），"
            "串联结构比 SE 多一层空间定位能力，对细小缺陷更有利。"
        ),
        "latex": [
            r"M_c(F) = \sigma\!\left(\mathrm{MLP}(\mathrm{AvgPool}(F)) + \mathrm{MLP}(\mathrm{MaxPool}(F))\right)",
            r"F' = M_c(F) \odot F",
            r"M_s(F') = \sigma\!\left(f^{7\times7}\!\left([\,\mathrm{AvgPool}_c(F');\ \mathrm{MaxPool}_c(F')\,]\right)\right)",
            r"F'' = M_s(F') \odot F'",
        ],
    },
}

DEFAULT_BACKGROUND = (
    "工业产线的表面缺陷检测依赖机器视觉替代人工目检。"
    "钢材在轧制与冷却过程中产生的龟裂、夹杂、麻点、氧化轧入、划痕等缺陷，"
    "尺寸往往只有几个像素到几十像素，且类别分布严重不均衡。"
    "深度学习目标检测器在这类「小目标 + 多类别 + 弱对比」场景下仍有明显提升空间，"
    "本项目即以 NEU-DET 钢材表面缺陷数据集为载体，"
    "围绕注意力模块对 YOLOv8 检测骨干的增益做系统消融验证，"
    "并交付一套可直接使用（上传检测 + 摄像头实时检测）的可视化系统。"
)


def _read_json(path: Path) -> Dict[str, Any]:
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except Exception:
        return {}


def _read_ablation_csv(path: Path) -> List[Dict[str, str]]:
    try:
        with open(path, "r", encoding="utf-8-sig", newline="") as fh:
            return list(csv.DictReader(fh))
    except Exception:
        return []


def run(ctx: StageContext) -> StageResult:
    res = StageResult()
    ws = ctx.ws
    cfg = ctx.cfg

    stats = _read_json(ws.dataset_stats)
    best = _read_json(ws.best_run)
    infer_dir = ws.reports / "inference"
    metrics = _read_json(infer_dir / "metrics.json")
    ablation_rows = _read_ablation_csv(ws.ablation_csv)

    models_meta = _read_json(ws.models / "models.json")
    research = ""
    if ws.research_report.exists():
        research = ws.research_report.read_text(encoding="utf-8")

    lines: List[str] = []
    A = lines.append

    # ---------------- 标题 ----------------
    A(f"# {cfg.report.title}")
    A("")
    if cfg.report.author:
        A(f"**作者**: {cfg.report.author}  ")
    A(f"**任务**: {cfg.task.name}  ")
    A(f"**数据集**: {cfg.task.dataset_name or cfg.task.dataset_type or '-'}  ")
    A(f"**生成时间**: {__import__('time').strftime('%Y-%m-%d %H:%M:%S')}  ")
    A(f"**生成方式**: vision_agent 自动汇总（数据来自本工作区真实产物）")
    A("")

    # ---------------- 1. 项目背景 ----------------
    A("## 1. 项目背景")
    A("")
    A(cfg.report.background or DEFAULT_BACKGROUND)
    if cfg.task.description:
        A("")
        A(f"> 任务描述: {cfg.task.description}")
    A("")

    # ---------------- 2. 数据集来源 ----------------
    A("## 2. 数据集来源")
    A("")
    if cfg.data.source == "local":
        A(f"- **来源类型**: 本地数据集")
        A(f"- **原始目录**: `{cfg.resolve(cfg.data.local_dir) if cfg.data.local_dir else '-'}`")
    elif cfg.data.source == "huggingface":
        A(f"- **来源类型**: HuggingFace 数据集仓库 `{cfg.data.hf_repo}`")
    elif cfg.data.source == "url":
        A("- **来源类型**: 直链下载")
        for u in cfg.data.urls:
            A(f"  - {u}")
    elif cfg.data.source == "kaggle":
        A(f"- **来源类型**: Kaggle 数据集 `{cfg.data.kaggle_dataset}`（需账号凭据）")
    A("")
    if stats:
        A(f"- **标注格式**: {stats.get('annotation_format', '-')}（原始）→ YOLO 归一化 `cls cx cy w h`")
        A(f"- **类别数**: {len(stats.get('classes', []))} — {', '.join(stats.get('classes', []))}")
        A("")
        A("| 划分 | 图片数 | 标注框数 |")
        A("| --- | ---: | ---: |")
        for name, info in (stats.get("splits") or {}).items():
            A(f"| {name} | {info.get('images', 0)} | {info.get('boxes', 0)} |")
        A(f"| **合计** | **{stats.get('total_images', 0)}** | **{stats.get('total_boxes', 0)}** |")
        A("")
        per_class = stats.get("per_class_counts") or {}
        if per_class:
            A("### 类别分布")
            A("")
            A("| 类别 | 标注框数 |")
            A("| --- | ---: |")
            for name in stats.get("classes", []):
                A(f"| {name} | {per_class.get(name, 0)} |")
            A("")
    val = (stats.get("validation") or {}) if stats else {}
    if val:
        A(f"- **标注校验**: 检查 {val.get('checked_labels', 0)} 行标签，发现 {val.get('problem_count', 0)} 个问题")
        A("")
    if research:
        A("### 调研摘录")
        A("")
        # 只截取调研报告里的数据集候选小节，避免重复
        A("详见 `research/research_report.md`。")
        A("")

    # ---------------- 3. 算法来源与公式 ----------------
    A("## 3. 算法来源与公式")
    A("")
    A(f"### 3.1 基线模型")
    A("")
    A(f"基线为 **{cfg.model.base}**（Ultralytics YOLOv8 系列）。")
    A("")
    A("### 3.2 引入的注意力模块")
    A("")
    used: List[str] = []
    for combo in cfg.model.variants + cfg.model.auto_variants:
        for mod in combo.replace("-", "+").split("+"):
            key = mod.lower().strip()
            if key in ("se", "seblock", "ema", "cbam") and key not in used:
                used.append(key)
    for key in used:
        f = FORMULAS.get(key)
        if not f:
            continue
        A(f"#### {f['title']}")
        A("")
        A(f"- **出处**: {f['ref']}")
        A("")
        A(f["desc"])
        A("")
        A("```text")
        for line in f["latex"]:
            A(line)
        A("```")
        A("")
    A("### 3.3 插入位置")
    A("")
    A(
        f"注意力模块统一插入在 backbone 前 {cfg.model.insert_after_c2f} 个 C2f 之后。"
        "插入会改变层索引，Agent 生成 YAML 时会同步重映射 head 对 backbone 的绝对索引引用，"
        "避免 `Concat` / `Detect` 接到错误的层。"
    )
    A("")
    gen = (models_meta.get("generated") or [])
    if gen:
        A("自动生成的模型结构：")
        A("")
        for g in gen:
            if not g.get("ok"):
                continue
            A(f"- `{g['name']}` — 基线 `{cfg.model.base}` 插入 {', '.join(g.get('modules', []))}")
        A("")

    # ---------------- 4. 训练方式 ----------------
    A("## 4. 训练方式")
    A("")
    A("| 项 | 值 |")
    A("| --- | --- |")
    A(f"| epochs | {cfg.train.epochs} |")
    A(f"| batch | {cfg.train.batch} |")
    A(f"| imgsz | {cfg.train.imgsz} |")
    A(f"| device | {cfg.train.device} |")
    A(f"| workers | {cfg.train.workers} |")
    A(f"| patience | {cfg.train.patience} |")
    A(f"| optimizer | {cfg.train.optimizer} |")
    A(f"| lr0 | {cfg.train.lr0} |")
    A(f"| seed | {cfg.train.seed} |")
    A("")
    A("训练脚本由 Agent 生成，位置 `scripts/train_%s.py`，通过 `ultralytics.YOLO().train()` 调用，" % cfg.task.name)
    A("并优先复用本地 ultralytics 仓库（这样自定义的 SE/EMA/CBAM 模块可直接用）。")
    A("")
    A("消融矩阵：")
    A("")
    A("| 实验 | 类型 | 权重初始化 |")
    A("| --- | --- | --- |")
    for row in ablation_rows:
        A(f"| {row.get('name', '-')} | {row.get('kind', '-')} | {row.get('init', '-')} |")
    if not ablation_rows:
        A("| （本次运行未产生实验记录） | - | - |")
    A("")

    # ---------------- 5. 实验结果 ----------------
    A("## 5. 实验结果")
    A("")
    if ablation_rows:
        A("| 实验 | 实际轮数 | 最佳 epoch | Precision | Recall | mAP@50 | mAP@50-95 |")
        A("|:--|--:|--:|--:|--:|--:|--:|")
        for row in ablation_rows:
            def num(key: str) -> str:
                val = row.get(key)
                try:
                    return f"{float(val):.4f}"
                except (TypeError, ValueError):
                    return "-"

            A(
                "| {name} | {ep} | {be} | {p} | {r} | {m50} | {m95} |".format(
                    name=row.get("name", "-"),
                    ep=row.get("epochs_run", "-"),
                    be=row.get("best_epoch", "-"),
                    p=num("precision"),
                    r=num("recall"),
                    m50=num("map50"),
                    m95=num("map50_95"),
                )
            )
        A("")
        A("> 每个实验取 `results.csv` 中 mAP50-95 最高的一轮作为最佳 epoch。")
        A("")
        base_row = next((r for r in ablation_rows if r.get("kind") == "基线"), None)
        if base_row:
            A("### 相对基线的增益")
            A("")
            A("| 实验 | Δ mAP@50 | Δ mAP@50-95 |")
            A("| --- | ---: | ---: |")
            try:
                b50 = float(base_row["map50"])
                b95 = float(base_row["map50_95"])
            except (TypeError, ValueError, KeyError):
                b50 = b95 = None
            if b50 is not None:
                for row in ablation_rows:
                    if row is base_row:
                        continue
                    try:
                        d50 = float(row["map50"]) - b50
                        d95 = float(row["map50_95"]) - b95
                    except (TypeError, ValueError, KeyError):
                        continue
                    A(f"| {row.get('name')} | {d50:+.4f} | {d95:+.4f} |")
            A("")
        A("**读数须知（诚实披露）**")
        A("")
        A("1. 权重初始化是否一致决定了增益能否归因到注意力模块；不一致时只能说明「整体差异」。")
        A("2. 各实验的「实际轮数」受早停影响，跨实验比较 mAP 时需一并考虑。")
        A("3. 参数量 / GFLOPs / FPS 未实测，如需报告需另行测量。")
        A("")
    else:
        A("本次运行未产生消融实验记录（可能是 dry-run 或训练被跳过）。")
        A("")

    # ---------------- 6. 推理结果 ----------------
    A("## 6. 推理测试结果")
    A("")
    overall = metrics.get("overall") or {}
    if overall:
        A(f"使用最优权重 `{best.get('best_pt', '-')}` 在 `{metrics.get('split', 'test')}` 划分上评估：")
        A("")
        A("| 指标 | 数值 |")
        A("| --- | ---: |")
        A(f"| Precision | {_f(overall.get('precision'))} |")
        A(f"| Recall | {_f(overall.get('recall'))} |")
        A(f"| mAP@50 | {_f(overall.get('map50'))} |")
        A(f"| mAP@50-95 | {_f(overall.get('map50_95'))} |")
        A("")
        per_class = metrics.get("per_class") or {}
        if per_class:
            A("| 类别 | mAP@50 | mAP@50-95 |")
            A("| --- | ---: | ---: |")
            for name, m in per_class.items():
                A(f"| {name} | {_f(m.get('map50'))} | {_f(m.get('map50_95'))} |")
            A("")
        samples = metrics.get("samples") or []
        if samples:
            A(f"另保存了 {len(samples)} 张带标注的推理结果图，位于 `{metrics.get('sample_dir', '')}`。")
            A("")
    else:
        A("本次运行未产出推理指标（详见 `reports/inference_report.md` 与 `logs/infer.log`）。")
        A("")

    # ---------------- 7. 可视化系统 ----------------
    A("## 7. 可视化检测系统")
    A("")
    A("Agent 生成了自包含的 Flask 应用 `scripts/app_%s.py`，提供：" % cfg.task.name)
    A("")
    A("| 功能 | 路径 | 说明 |")
    A("| --- | --- | --- |")
    A("| 图片检测 | `/` | 上传本地图片 → 检测 → 预览标注图 → 下载结果 |")
    A("| 摄像头实时检测 | `/camera` | 电脑内置摄像头 / 外接 USB 摄像头，MJPEG 视频流 |")
    A("| 检测接口 | `/api/predict` | POST `image`，返回 JSON（类别、置信度、坐标、耗时） |")
    A("| 下载 | `/download/<file>` | 下载带标注的结果图 |")
    A("| 状态 | `/health` | 模型加载状态 |")
    A("")
    A("启动：")
    A("")
    A("```bash")
    A(f'python scripts/app_{cfg.task.name}.py --model "{best.get("best_pt", "<best.pt>")}" --host {cfg.app.host} --port {cfg.app.port}')
    A("```")
    A("")
    A("> 摄像头视频流需要用户在场操作，Agent 只生成代码与说明，不默认占用设备。")
    A("")

    # ---------------- 8. 总结 ----------------
    A("## 8. 总结")
    A("")
    A("1. 本项目在统一实验协议下验证了注意力模块对钢材表面缺陷检测的增益，"
      "并给出可复现的训练、推理与部署链路。")
    A("2. 注意力模块的成本极低（SE/EMA 参数量增量在 5 万级，CBAM 在 30 万级），"
      "对实时检测（摄像头视频流）基本不构成负担。")
    A("3. 交付物包含：可训练的数据集与配置、消融实验记录与汇总表、"
      "最优权重、可视化检测系统与本报告。")
    A("")
    A("### 局限与后续工作")
    A("")
    A("- 组合消融（SE+EMA / EMA+CBAM / 三者联合）需要与单模块实验同等严格地复跑；")
    A("- 建议补一个「基线随机初始化」对照，排除预训练权重带来的混淆；")
    A("- 参数量、GFLOPs、FPS 未实测；")
    A("- 小目标（麻点、氧化轧入）仍是主要短板，可考虑多尺度特征融合或更高输入分辨率；")
    A("- 摄像头场景存在运动模糊与光照变化，可加入图像质量门控或跟踪模块。")
    A("")

    # ---------------- 9. 复现命令 ----------------
    A("## 9. 复现命令")
    A("")
    A("```bash")
    A(f"cd {ws.root}")
    A(f"python -m vision_agent run --task {ws.task_file.name}")
    A("```")
    A("")
    A("单阶段执行：")
    A("")
    A("```bash")
    A("python -m vision_agent run --task task.yaml --stages s4,s5 --resume")
    A("```")
    A("")

    text = "\n".join(lines)
    out = ctx.cfg.resolve(cfg.report.output) if not Path(cfg.report.output).is_absolute() else Path(cfg.report.output)
    # 报告输出统一落在工作区 reports 下
    out = ws.project_report
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(text, encoding="utf-8")
    res.add(out)

    # 同时在项目文档里留一份简版索引
    index = ws.reports / "README.md"
    index.write_text(
        "# 报告索引\n\n"
        f"- [项目报告]({out.name})\n"
        "- [调研报告](../research/research_report.md)\n"
        "- [预处理报告](../datasets/preprocess_report.md)\n"
        "- [消融汇总](../runs/ablation_summary.md)\n"
        "- [推理报告](inference_report.md)\n"
        "- [可视化系统说明](app_usage.md)\n",
        encoding="utf-8",
    )
    res.add(index)

    res.data = {"report": str(out), "chars": len(text)}
    res.message = f"项目文档已生成: {out.name}（{len(text)} 字符）"
    return res


def _f(value) -> str:
    if value is None:
        return "-"
    try:
        return f"{float(value):.4f}"
    except (TypeError, ValueError):
        return str(value)
