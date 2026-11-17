"""S1 · 数据集与论文调研（对应工作流第 2 步）。

流程：
    1. 用关键词在权威数据集站点（HuggingFace / Zenodo / Kaggle / OpenDataLab）检索；
    2. 在学术站点（arXiv / Semantic Scholar / Crossref / OpenAlex）检索相关论文；
    3. 把检索结果喂给大模型，产出三份产物：
       - research_report.md   数据集候选 + 论文 + 改进方向总览
       - datasets.json        结构化数据集候选（可直接被 S3 消费）
       - papers.json          结构化论文
       - improvements.md      模型改进 / 训练方案（含公式要点）

没有大模型时走确定性模板：报告结构一样，内容直接来自真实检索结果，
只是没有经过模型的归纳润色（文件里会明确标注）。
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Dict, List

from ..search import datasets as ds_search
from ..search import papers as paper_search
from .base import StageContext, StageResult

DATASET_SOURCES = {"huggingface", "zenodo", "kaggle", "opendatalab"}
PAPER_SOURCES = {"arxiv", "semanticscholar", "crossref", "openalex"}


def _keywords(ctx: StageContext) -> List[str]:
    cfg = ctx.cfg
    kws: List[str] = []
    if cfg.task.dataset_name:
        kws.append(cfg.task.dataset_name)
    kws.extend(cfg.task.dataset_keywords)
    if cfg.task.dataset_type:
        kws.append(cfg.task.dataset_type)
    # 去重且保持顺序
    seen, out = set(), []
    for kw in kws:
        k = (kw or "").strip()
        if k and k.lower() not in seen:
            seen.add(k.lower())
            out.append(k)
    return out


def _extract_key_points(abstract: str, keywords: List[str], max_points: int = 2) -> str:
    """从摘要里挑出与本任务最相关的句子。

    模板模式下没有大模型做归纳，就用关键词命中的句子做「要点」，
    比整段贴摘要更有信息量。
    """
    text = (abstract or "").strip()
    if not text:
        return ""
    # 简单断句（英文摘要用句号，中文用句号/分号）
    raw_sentences = re.split(r"(?<=[.!?;。；])\s+", text)
    sentences = [s.strip() for s in raw_sentences if len(s.strip()) > 20]
    if not sentences:
        return text[:180]
    terms = [k.lower() for k in keywords if k and len(k) > 2]
    terms += ["attention", "yolo", "detect", "defect", "surface", "steel", "attention module"]
    scored = []
    for idx, sent in enumerate(sentences):
        low = sent.lower()
        score = sum(low.count(t) for t in terms)
        if idx == 0:
            score += 2  # 摘要首句通常是研究动机
        scored.append((score, -idx, sent))
    scored.sort(reverse=True)
    picked = [s for _, _, s in scored[:max_points]]
    # 保持原文顺序
    picked.sort(key=lambda s: sentences.index(s))
    return " ".join(picked)[:320]


def _extract_dataset_facts(text: str) -> Dict[str, str]:
    """从数据集描述里抠出规模 / 类别数 / 格式等硬信息。"""
    low = (text or "").lower()
    facts: Dict[str, str] = {}
    m = re.search(r"([\d,]{3,})\s*(?:images|photos|pictures|frames)", low)
    if m:
        facts["图片数"] = m.group(1)
    m = re.search(r"(\d+)\s*(?:categories|classes|types|defect types)", low)
    if m:
        facts["类别数"] = m.group(1)
    if "yolo" in low:
        facts["格式"] = "YOLO"
    elif "coco" in low:
        facts["格式"] = "COCO"
    elif "voc" in low or "pascal" in low:
        facts["格式"] = "VOC XML"
    m = re.search(r"(?:resolution|size|pixels?)[^.;]{0,30}?(\d{3,4})\s*[x×]\s*(\d{3,4})", low)
    if m:
        facts["分辨率"] = f"{m.group(1)}x{m.group(2)}"
    return facts


def scan_local_attention_modules(project_root) -> List[Dict[str, str]]:
    """扫描本地 ultralytics 仓库里可用的注意力/增强模块。

    模板模式下没有大模型给「改进方向」，但本地代码里到底有哪些模块是可以确定的——
    扫出来的模块就是能立刻做消融的候选，比空泛建议有用得多。
    """
    import re as _re
    from pathlib import Path as _Path

    root = _Path(project_root)
    modules_dir = root / "ultralytics" / "nn" / "modules"
    if not modules_dir.exists():
        return []
    pattern = _re.compile(r"^class\s+(\w+)\s*\(", _re.M)
    hints = (
        "SE",
        "CBAM",
        "EMA",
        "Attention",
        "ECA",
        "SimAM",
        "CA",
        "SA",
        "GC",
        "SK",
        "Bottleneck",
        "Rep",
        "Ghost",
        "Mobile",
    )
    found: List[Dict[str, str]] = []
    for py in sorted(modules_dir.glob("*.py")):
        text = py.read_text(encoding="utf-8", errors="replace")
        for name in pattern.findall(text):
            if not any(h.lower() in name.lower() for h in hints):
                continue
            # 取出类附近的 docstring 作为说明
            doc = ""
            m = _re.search(rf"class\s+{_re.escape(name)}\b[^\n]*:\s*\n\s*\"\"\"(.*?)\"\"\"", text, _re.S)
            if m:
                doc = m.group(1).strip().splitlines()[0][:120]
            found.append({"name": name, "file": py.name, "doc": doc})
    # 去重
    seen, uniq = set(), []
    for item in found:
        if item["name"] in seen:
            continue
        seen.add(item["name"])
        uniq.append(item)
    return uniq


def _rank_datasets(candidates, keywords: List[str]):
    """按与任务的相关度给数据集候选排序，并剔除明显不相关的命中。

    检索站点会因为关键词子串命中而返回噪音（例如搜 "NEU-DET" 会命中
    名字里带 "neu" 的无关数据集），模板模式下没有模型帮你判断，
    这里用「领域词命中 + 完整关键词命中」做一个确定性的相关性打分。
    """
    domain_terms = {
        "defect",
        "surface",
        "steel",
        "yolo",
        "inspection",
        "crack",
        "工业",
        "缺陷",
        "钢材",
        "表面",
    }
    kws = [k.lower() for k in keywords if k and len(k) > 3]

    def _norm(text: str) -> str:
        return re.sub(r"[^a-z0-9\u4e00-\u9fff]+", "", (text or "").lower())

    norm_kws = [_norm(k) for k in kws]
    norm_kws = [k for k in norm_kws if len(k) >= 4]

    scored = []
    for c in candidates:
        text = f"{c.name} {c.description} {c.source}".lower()
        norm_text = _norm(text)
        score = sum(text.count(k) for k in kws)
        # 归一化后再匹配一次，覆盖 neu_det / neu-det / NEUDET 这类写法差异
        score += 3 * sum(1 for k in norm_kws if k in norm_text)
        score += 3 * sum(1 for d in domain_terms if d in text)
        scored.append((score, c))
    scored.sort(key=lambda x: -x[0])
    kept = [(s, c) for s, c in scored if s > 0]
    if not kept:  # 全都判定为不相关时，宁可原样保留，不给用户空手
        kept = scored
    return kept


def run(ctx: StageContext) -> StageResult:
    res = StageResult()
    ws = ctx.ws
    ws.research_raw.mkdir(parents=True, exist_ok=True)

    keywords = _keywords(ctx)
    limit = int(ctx.cfg.search.max_results)
    timeout = float(ctx.cfg.search.timeout)

    # --- 1. 检索数据集 ---
    dataset_hits: List[ds_search.DatasetCandidate] = []
    if ctx.cfg.search.enabled and not ctx.cfg.search.offline:
        srcs = [s for s in ctx.cfg.search.sources if s in DATASET_SOURCES] or ["huggingface", "zenodo"]
        ctx.logger.info(f"检索数据集站点: {srcs} · 关键词={keywords}")
        dataset_hits = ds_search.search_all(keywords, limit=limit, sources=srcs, timeout=timeout)
        ranked = _rank_datasets(dataset_hits, keywords)
        dropped = len(dataset_hits) - len(ranked)
        dataset_hits = [c for _, c in ranked]
        if dropped:
            ctx.logger.info(f"按相关度过滤掉 {dropped} 条疑似无关命中")
        (ws.research_raw / "datasets_raw.json").write_text(
            json.dumps(
                [{"score": s, **c.to_dict()} for s, c in ranked], indent=2, ensure_ascii=False
            ),
            encoding="utf-8",
        )
    ctx.logger.info(f"数据集候选: {len(dataset_hits)} 条")

    # --- 2. 检索论文 ---
    paper_hits: List[paper_search.PaperCandidate] = []
    if ctx.cfg.search.enabled and not ctx.cfg.search.offline:
        psrcs = [s for s in ctx.cfg.search.sources if s in PAPER_SOURCES] or ["arxiv", "semanticscholar"]
        ctx.logger.info(f"检索论文站点: {psrcs}")
        paper_hits = paper_search.search_all(keywords, limit=limit, sources=psrcs, timeout=timeout)
        (ws.research_raw / "papers_raw.json").write_text(
            json.dumps([p.to_dict() for p in paper_hits], indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
    ctx.logger.info(f"论文候选: {len(paper_hits)} 条")

    # --- 3. 归纳 ---
    local_modules = scan_local_attention_modules(ctx.cfg.project_root)
    if local_modules:
        ctx.logger.info(f"本地仓库可用模块: {[m['name'] for m in local_modules][:10]}")
    digest = _build_digest(ctx, dataset_hits, paper_hits)
    digest["local_modules"] = local_modules
    report, source, err = _synthesize(ctx, digest)
    (ws.research_report).write_text(report, encoding="utf-8")

    datasets_json = [d.to_dict() for d in dataset_hits]
    for item in datasets_json:
        item["annotation_format_guess"] = ds_search.guess_annotation_format(
            ds_search.DatasetCandidate(**{k: v for k, v in item.items() if k in ds_search.DatasetCandidate.__dataclass_fields__})
        )
    (ws.research_datasets).write_text(
        json.dumps(datasets_json, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    (ws.research_papers).write_text(
        json.dumps([p.to_dict() for p in paper_hits], indent=2, ensure_ascii=False), encoding="utf-8"
    )
    (ws.research_improvements).write_text(_improvements_md(ctx, digest, source), encoding="utf-8")
    (ws.research / "local_modules.json").write_text(
        json.dumps(local_modules, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    res.add(ws.research_report)
    res.add(ws.research_datasets)
    res.add(ws.research_papers)
    res.add(ws.research_improvements)
    res.data = {
        "dataset_hits": len(dataset_hits),
        "paper_hits": len(paper_hits),
        "llm_source": source,
        "llm_error": err,
    }
    res.message = f"调研完成：数据集 {len(dataset_hits)} 条 / 论文 {len(paper_hits)} 条（归纳来源: {source}）"
    return res


# ---------------------------------------------------------------------------
# 摘要
# ---------------------------------------------------------------------------


def _build_digest(ctx: StageContext, ds_hits, paper_hits) -> Dict[str, Any]:
    cfg = ctx.cfg
    return {
        "task": {
            "name": cfg.task.name,
            "dataset_type": cfg.task.dataset_type,
            "dataset_name": cfg.task.dataset_name,
            "keywords": _keywords(ctx),
            "classes": cfg.task.classes,
            "base_model": cfg.model.base,
            "variants": cfg.model.variants + cfg.model.auto_variants,
        },
        "datasets": [d.to_dict() for d in ds_hits],
        "papers": [p.to_dict() for p in paper_hits],
    }


def _synthesize(ctx: StageContext, digest: Dict[str, Any]):
    """调用大模型生成调研报告；失败则用模板。"""
    template = _template_report(digest, "template")
    system = (
        "你是一名资深计算机视觉研究员。你的任务是基于给定的检索结果，"
        "为工业表面缺陷检测项目写一份可落地的调研报告。"
        "只使用给定材料中的事实，不要编造不存在的论文、数据集或链接。"
        "如果材料不足，就明确写「材料不足」。输出 Markdown。"
    )
    user = (
        "以下是检索到的原始材料（JSON）。请输出一份 Markdown 调研报告，包含以下小节：\n"
        "## 1. 数据集候选\n表格列出：名称、来源站点、规模/图片数、类别、标注格式、许可、下载链接、推荐度与理由。\n"
        "## 2. 相关论文与算法来源\n表格列出：标题、年份、出处、链接、与本项目的关系（用于哪个改进点）。\n"
        "## 3. 模型改进方向\n"
        "结合上面论文与当前基线（{base}），给出 3-5 个可执行的改进方向，每个写清楚："
        "改什么模块、为什么有效、预期收益、实现位置（backbone/head/loss）、消融实验怎么设计。\n"
        "## 4. 训练方案建议\n数据增强、损失函数、优化器、学习率策略、训练轮数、早停、评价指标。\n"
        "## 5. 数据预处理注意事项\n针对该数据集的标注格式，说明转换成 YOLO 归一化格式时要注意什么。\n\n"
        "原始材料：\n```json\n{json}\n```"
    ).format(base=digest["task"]["base_model"], json=json.dumps(digest, ensure_ascii=False, indent=1)[:12000])
    result = ctx.llm.complete_or_template(system, user, template, label="research")
    if result.ok and result.source == "llm":
        return result.text, "llm", ""
    return result.text or template, result.source or "template", result.error


def _template_report(digest: Dict[str, Any], source: str) -> str:
    """确定性模板（无大模型时的兜底），内容直接来自检索结果。"""
    task = digest["task"]
    lines: List[str] = []
    lines.append(f"# {task['dataset_type'] or task['name']} · 数据集与算法调研报告")
    lines.append("")
    if source == "template":
        lines.append("> ⚠️ 本报告由确定性模板生成（未调用大模型润色）。数据与论文条目均来自真实检索结果。")
        lines.append("")
    lines.append(f"- **目标数据集**: {task['dataset_name'] or '未指定'}")
    lines.append(f"- **检索关键词**: {', '.join(task['keywords'])}")
    lines.append(f"- **基线模型**: {task['base_model']}")
    lines.append(f"- **计划变体**: {', '.join(task['variants']) or '无'}")
    lines.append("")

    lines.append("## 1. 数据集候选")
    lines.append("")
    if digest["datasets"]:
        lines.append("| 名称 | 来源 | 规模 / 格式 | 许可 | 链接 |")
        lines.append("| --- | --- | --- | --- | --- |")
        for d in digest["datasets"][:20]:
            facts = _extract_dataset_facts(d.get("description", ""))
            size = "、".join(f"{k}={v}" for k, v in facts.items()) or "-"
            fmt = d.get("annotation_format_guess") or ""
            if fmt and fmt != "未知（需人工确认）":
                size = f"{size}、{fmt}" if size != "-" else fmt
            lines.append(
                f"| {d.get('name', '-')[:60]} | {d.get('source', '-')} | {size} | {d.get('license') or '-'} | {d.get('url', '')} |"
            )
    else:
        lines.append("未检索到候选数据集（可能是离线或关键词过窄）。")
    lines.append("")

    lines.append("## 2. 相关论文与算法来源")
    lines.append("")
    if digest["papers"]:
        lines.append("| 标题 | 年份 | 出处 | 要点 | 链接 |")
        lines.append("| --- | --- | --- | --- | --- |")
        for p in digest["papers"][:20]:
            key_points = _extract_key_points(p.get("abstract", ""), digest["task"]["keywords"])
            lines.append(
                f"| {(p.get('title') or '-')[:60]} | {p.get('year') or '-'} | {(p.get('venue') or '-')[:24]} "
                f"| {key_points or '-'} | {p.get('url', '')} |"
            )
    else:
        lines.append("未检索到论文。")
    lines.append("")

    lines.append("## 3. 模型改进方向")
    lines.append("")
    lines.append(_improvements_body(digest))
    lines.append("")
    lines.append("## 4. 训练方案建议")
    lines.append("")
    lines.append("- 图像尺寸 640，batch 16~32，AdamW 或 SGD(momentum=0.937, nesterov) 起手；")
    lines.append("- 早停 patience 50，保存 mAP50-95 最佳轮次；")
    lines.append("- 增强：Mosaic/MixUp（ultralytics 默认）+ 随机亮度对比度扰动；")
    lines.append("- 指标：mAP50、mAP50-95、Precision、Recall，并给出分类别表格。")
    lines.append("")
    lines.append("## 5. 数据预处理注意事项")
    lines.append("")
    lines.append(
        "- VOC XML 坐标是 1-based 且可能越界，需裁剪到 [0, W/H] 后再归一化；\n"
        "- 归一化后需校验 cx±w/2 ∈ [0,1]，否则检测头会学到错误目标；\n"
        "- 图片与标签必须同名同扩展名，否则 ultralytics 会静默丢样本。")
    lines.append("")
    return "\n".join(lines)


def _improvements_body(digest: Dict[str, Any]) -> str:
    task = digest["task"]
    variants = task.get("variants") or []
    local_modules = digest.get("local_modules") or []
    lines = []
    if variants:
        lines.append(f"本项目已规划的消融变体：{', '.join(variants)}。")
        lines.append("")
    lines.append("| 改进点 | 说明 | 落地位置 | 消融设计 |")
    lines.append("| --- | --- | --- | --- |")
    lines.append(
        "| SEBlock | 通道注意力，抑制背景响应、突出缺陷通道 | backbone C2f 之后 | 基线 vs +SE |"
    )
    lines.append(
        "| CBAM | 通道+空间双重注意力，定位细小缺陷 | backbone C2f 之后 | 基线 vs +CBAM |"
    )
    lines.append(
        "| EMA(多尺度) | 多分支分组卷积聚合多尺度上下文 | backbone C2f 之后 | 基线 vs +EMA |"
    )
    lines.append(
        "| 组合 | SE+EMA+CBAM 同时插入 | backbone C2f 之后 | 单模块 vs 组合 |"
    )
    lines.append("")
    if local_modules:
        lines.append(
            "### 本地仓库里可直接做消融的模块"
        )
        lines.append("")
        lines.append(
            "以下模块已在你的 ultralytics 仓库里实现，Agent 可以直接为它们生成模型 YAML 并纳入消融矩阵："
        )
        lines.append("")
        lines.append("| 模块 | 所在文件 | 说明 |")
        lines.append("| --- | --- | --- |")
        for m in local_modules[:20]:
            lines.append(f"| {m['name']} | `{m['file']}` | {m.get('doc') or '-'} |")
        lines.append("")
    lines.append(
        "> 组合消融必须补一个「基线随机初始化」对照："
        "若基线用 COCO 预训练而变体从随机初始化训练，指标差异不能归因于注意力模块。"
    )
    return "\n".join(lines)


def _improvements_md(ctx: StageContext, digest: Dict[str, Any], source: str) -> str:
    header = "# 模型改进与训练方案\n\n"
    if source == "template":
        header += "> ⚠️ 由确定性模板生成（未调用大模型）。\n\n"
    header += _improvements_body(digest)
    header += "\n## 训练方案\n\n"
    cfg = ctx.cfg
    header += (
        f"- epochs={cfg.train.epochs}, batch={cfg.train.batch}, imgsz={cfg.train.imgsz}, "
        f"device={cfg.train.device}, patience={cfg.train.patience}\n"
        f"- 优化器={cfg.train.optimizer}, lr0={cfg.train.lr0}, seed={cfg.train.seed}\n"
        "- 每个变体独立 run 目录，互不覆盖；统一以 mAP50-95 最高轮次作为最佳 epoch。\n"
    )
    return header
