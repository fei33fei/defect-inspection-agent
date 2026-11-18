"""代码生成：模型 YAML / 训练脚本 / 推理脚本 / 可视化 App。

为什么要有这一层：
    Agent 不应该让用户手写消融实验的模型结构。
    「SE + EMA + CBAM」这种组合，本质是在 backbone 的 C2f 之后插入注意力模块，
    而插入会改变层索引，head 里对 backbone 的引用必须同步重映射。
    这里把这件事做对，并保证生成的 YAML 能被 ultralytics 的 parse_model 正确解析。

模块别名表（module alias -> ultralytics 类名）：
    se / seblock / senet      -> SEBlock
    ema                      -> EMA
    cbam                     -> CBAM
    ca / channelattention     -> ChannelAttention
    sa / spatialattention     -> SpatialAttention
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import yaml

MODULE_ALIASES: Dict[str, str] = {
    "se": "SEBlock",
    "seblock": "SEBlock",
    "senet": "SEBlock",
    "ema": "EMA",
    "cbam": "CBAM",
    "ca": "ChannelAttention",
    "channelattention": "ChannelAttention",
    "sa": "SpatialAttention",
    "spatialattention": "SpatialAttention",
}


def resolve_module(name: str) -> str:
    key = (name or "").strip().lower().replace("-", "").replace("_", "")
    if key not in MODULE_ALIASES:
        raise ValueError(f"未知模块 '{name}'，可选: {sorted(set(MODULE_ALIASES))}")
    return MODULE_ALIASES[key]


def variant_name(base: str, modules: List[str]) -> str:
    """生成变体名，例如 yolov8n + [se, ema, cbam] -> yolov8n-se-ema-cbam。"""
    base_stem = Path(base).stem
    if not modules:
        return base_stem
    return base_stem + "-" + "-".join(m.lower() for m in modules)


# ---------------------------------------------------------------------------
# 模型 YAML 生成
# ---------------------------------------------------------------------------


def _remap_from(
    frm: Any,
    backbone_map: Dict[int, int],
    old_backbone_len: int,
    new_backbone_len: int,
) -> Any:
    """把 from 字段里的绝对索引映射到插入模块后的新索引。"""
    if isinstance(frm, bool):
        return frm
    if isinstance(frm, int):
        if frm >= 0:
            if frm < old_backbone_len:
                return backbone_map.get(frm, frm)
            return new_backbone_len + (frm - old_backbone_len)
        return frm
    if isinstance(frm, list):
        return [_remap_from(x, backbone_map, old_backbone_len, new_backbone_len) for x in frm]
    return frm


def _flow_scalar(value: Any) -> str:
    """把一个标量转成 YAML 行内表示。"""
    if value is None:
        return "None"
    if isinstance(value, bool):
        return "True" if value else "False"
    if isinstance(value, (int, float)):
        return str(value)
    text = str(value)
    if text == "nc":  # ultralytics 里 nc 是占位符，不能加引号
        return "nc"
    return "'" + text.replace("'", "''") + "'"


def _flow_list(values: List[Any]) -> str:
    return "[" + ", ".join(_flow_scalar(v) for v in values) + "]"


def dump_model_yaml(data: Dict[str, Any], header: str = "") -> str:
    """按 ultralytics 的风格输出模型 YAML。

    ultralytics 自带的 YAML 长这样（人读起来很顺）：
        backbone:
          # [from, repeats, module, args]
          - [-1, 1, Conv, [64, 3, 2]]  # 0-P1/2
    用 yaml.safe_dump 会输出成 `- - -1` 的样式，可读性差，所以这里自己序列化。
    """
    lines: List[str] = []
    if header:
        lines.append(header.rstrip("\n"))

    for key, value in data.items():
        if key in {"backbone", "head"}:
            lines.append(f"{key}:")
            lines.append("  # [from, repeats, module, args]")
            for idx, row in enumerate(value or []):
                row = list(row)
                frm = row[0]
                frm_s = _flow_list(frm) if isinstance(frm, list) else _flow_scalar(frm)
                repeats = _flow_scalar(row[1]) if len(row) > 1 else "1"
                module = str(row[2]) if len(row) > 2 else ""
                args = row[3] if len(row) > 3 else []
                if isinstance(args, list):
                    args_s = _flow_list(args)
                else:
                    args_s = _flow_scalar(args)
                lines.append(f"  - [{frm_s}, {repeats}, {module}, {args_s}]  # {idx}")
        elif key == "scales":
            lines.append("scales:")
            for scale, params in (value or {}).items():
                lines.append(f"  {scale}: {_flow_list(list(params))}")
        elif isinstance(value, (list, tuple)):
            lines.append(f"{key}: {_flow_list(list(value))}")
        else:
            lines.append(f"{key}: {_flow_scalar(value)}")
    return "\n".join(lines) + "\n"


def insert_modules_into_yaml(
    base_yaml_text: str,
    modules: List[str],
    insert_after_c2f: int = 3,
    nc: Optional[int] = None,
    base_name: str = "base.yaml",
) -> Dict[str, Any]:
    """在 backbone 的前 insert_after_c2f 个 C2f 之后插入注意力模块。

    返回 {"text": 生成的 YAML 文本, "inserted": 插入位置列表, "yaml": 结构化数据}
    """
    data = yaml.safe_load(base_yaml_text) or {}
    backbone = list(data.get("backbone") or [])
    head = list(data.get("head") or [])
    if not backbone:
        raise ValueError("基础 YAML 里没有 backbone 定义")

    old_backbone_len = len(backbone)

    # 找 C2f 位置
    c2f_positions = [i for i, row in enumerate(backbone) if len(row) > 2 and row[2] == "C2f"]
    if not c2f_positions:
        raise ValueError("基础 YAML 的 backbone 里没有 C2f 模块")
    targets = set(c2f_positions[: max(1, insert_after_c2f)])

    new_backbone: List[List[Any]] = []
    backbone_map: Dict[int, int] = {}
    inserted: List[Dict[str, Any]] = []

    for old_idx, row in enumerate(backbone):
        new_row = list(row)
        new_row[0] = _remap_from(row[0], backbone_map, old_backbone_len, old_backbone_len)
        backbone_map[old_idx] = len(new_backbone)
        new_backbone.append(new_row)

        if old_idx in targets:
            # 该 C2f 的输出通道数（作为模块的 c1 占位；ultralytics 解析时会用真实通道覆盖）
            c1 = _c2f_out_channels(row)
            for mod in modules:
                cls = resolve_module(mod)
                new_backbone.append([-1, 1, cls, [c1]])
                inserted.append(
                    {
                        "after_backbone_index": old_idx,
                        "module": cls,
                        "channels": c1,
                        "new_index": len(new_backbone) - 1,
                    }
                )

    new_backbone_len = len(new_backbone)
    new_head: List[List[Any]] = []
    for row in head:
        new_row = list(row)
        new_row[0] = _remap_from(row[0], backbone_map, old_backbone_len, new_backbone_len)
        new_head.append(new_row)

    out: Dict[str, Any] = {}
    for key, value in data.items():
        if key == "backbone":
            out[key] = new_backbone
        elif key == "head":
            out[key] = new_head
        else:
            out[key] = value
    if nc is not None:
        out["nc"] = nc

    header = (
        "# 由 vision_agent 自动生成\n"
        f"# 基础结构: {base_name}\n"
        f"# 插入模块: {', '.join(resolve_module(m) for m in modules)}\n"
        f"# 插入位置: backbone 中前 {len(targets)} 个 C2f 之后\n"
        "# 注意: ultralytics 解析时会按真实通道数覆盖模块的 c1/c2 参数\n"
    )
    text = dump_model_yaml(out, header)
    return {"text": text, "inserted": inserted, "yaml": out}


def _c2f_out_channels(row: List[Any]) -> int:
    """取 C2f 行的输出通道声明（作为占位参数）。"""
    args = row[3] if len(row) > 3 and isinstance(row[3], list) else []
    for val in args:
        if isinstance(val, int) and val > 8:
            return val
    return 128


def _scale_stripped(base: str) -> str:
    """去掉模型名末尾的规模字母：yolov8n -> yolov8。

    ultralytics 的 scales 模板（yolov8.yaml）通过文件名里的 n/s/m/l/x 决定宽度，
    所以「yolov8n.yaml」这种具体规模文件不存在时，回退到模板是合理的。
    """
    m = re.match(r"^(.*?)([nslmx])$", base or "")
    return m.group(1) if m else base


def find_base_yaml(project_root: Path, base: str) -> Optional[Path]:
    """在本地 ultralytics 仓库里找到基础模型 YAML。

    优先精确匹配 `yolov8n.yaml`；找不到时回退到 scales 模板 `yolov8.yaml`
    （ultralytics 会在加载时按文件名推断规模）。
    """
    project_root = Path(project_root)
    names = [base]
    stripped = _scale_stripped(base)
    if stripped and stripped != base:
        names.append(stripped)

    for name in names:
        candidates = [
            project_root / "ultralytics" / "cfg" / "models" / "v8" / f"{name}.yaml",
            project_root / "ultralytics" / "cfg" / "models" / "v5" / f"{name}.yaml",
            project_root / "ultralytics" / "cfg" / "models" / f"{name}.yaml",
            project_root / f"{name}.yaml",
        ]
        for cand in candidates:
            if cand.exists():
                return cand

    # 兜底：递归找
    for name in names:
        for found in project_root.rglob(f"{name}.yaml"):
            if "cfg" in str(found) or "models" in str(found):
                return found
    return None


def generate_model_yaml(
    project_root: Path,
    base: str,
    modules: List[str],
    out_dir: Path,
    insert_after_c2f: int = 3,
    nc: Optional[int] = None,
) -> Dict[str, Any]:
    """生成一个消融变体的模型 YAML，返回元信息。"""
    project_root = Path(project_root)
    base_yaml = find_base_yaml(project_root, base)
    if base_yaml is None:
        return {
            "ok": False,
            "error": f"在 {project_root} 下找不到基础模型 YAML: {base}.yaml",
            "name": variant_name(base, modules),
        }
    text = base_yaml.read_text(encoding="utf-8")
    result = insert_modules_into_yaml(
        text, modules, insert_after_c2f=insert_after_c2f, nc=nc, base_name=base_yaml.name
    )
    name = variant_name(base, modules)
    out_path = Path(out_dir) / f"{name}.yaml"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(result["text"], encoding="utf-8")
    return {
        "ok": True,
        "name": name,
        "path": str(out_path),
        "base_yaml": str(base_yaml),
        "modules": [resolve_module(m) for m in modules],
        "inserted": result["inserted"],
    }


def _render(template: str, mapping: Dict[str, Any]) -> str:
    """把模板里的 @KEY@ 替换成实际值。

    不用 str.format 是因为生成的 Python 代码里含有大量 {} （dict 字面量、f-string），
    用 format 需要逐个转义，漏一个就会在运行期炸掉。
    """
    out = template
    for key, value in mapping.items():
        out = out.replace(f"@{key}@", str(value))
    return out

# ---------------------------------------------------------------------------
# 训练脚本
# ---------------------------------------------------------------------------

TRAIN_SCRIPT_TEMPLATE = '''#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""由 vision_agent 生成的训练脚本（@TASK_NAME@）。

用法:
    python @SCRIPT_NAME@ --model @DEFAULT_MODEL@ --data @DEFAULT_DATA@ --epochs @EPOCHS@ --device @DEVICE@

产物:
    <project>/<name>/weights/best.pt   最佳权重
    <project>/<name>/results.csv       逐 epoch 指标
    <project>/<name>/train_summary.json 训练摘要
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

# --- 可选：使用本地 ultralytics 仓库（含已魔改的 SE/EMA/CBAM 模块）---
# 默认走官方 ultralytics + vision_agent.attention 补丁，无需任何本地仓库。
# 若你有一份魔改过的 ultralytics，可设环境变量 VA_PROJECT_ROOT 指向它：
#     set VA_PROJECT_ROOT=D:/path/to/ultralytics
# 或在 task.yaml 里配 project_root（生成时会写入下面的默认值）。
_SCRIPT_BASE = Path(__file__).resolve().parent.parent
PROJECT_ROOT = Path(os.environ.get("VA_PROJECT_ROOT", r"@PROJECT_ROOT@"))
if not PROJECT_ROOT.is_absolute():
    PROJECT_ROOT = (_SCRIPT_BASE / PROJECT_ROOT).resolve()
if PROJECT_ROOT.exists():
    sys.path.insert(0, str(PROJECT_ROOT))

# --- 让官方 ultralytics 也支持 SEBlock / EMA / CBAM ---
# 若上面的本地仓库已自带这些模块，install() 会自动跳过重复注入。
try:
    from vision_agent.attention import install as _install_attention

    _install_attention()
except Exception:  # noqa: BLE001 - 补丁失败不应阻断训练，降级为原样报错
    pass

import torch  # noqa: E402
from ultralytics import YOLO  # noqa: E402


def parse_args():
    p = argparse.ArgumentParser(description="vision_agent 训练脚本")
    p.add_argument("--model", default="@DEFAULT_MODEL@", help="模型 YAML 或 .pt 权重")
    p.add_argument("--data", default="@DEFAULT_DATA@", help="data.yaml 路径")
    p.add_argument("--epochs", type=int, default=@EPOCHS@)
    p.add_argument("--batch", type=int, default=@BATCH@)
    p.add_argument("--imgsz", type=int, default=@IMGSZ@)
    p.add_argument("--device", default="@DEVICE@")
    p.add_argument("--workers", type=int, default=@WORKERS@)
    p.add_argument("--patience", type=int, default=@PATIENCE@)
    p.add_argument("--optimizer", default="@OPTIMIZER@")
    p.add_argument("--lr0", type=float, default=@LR0@)
    p.add_argument("--seed", type=int, default=@SEED@)
    p.add_argument("--project", default="@PROJECT@")
    p.add_argument("--name", default="@NAME@")
    p.add_argument("--exist-ok", action="store_true", default=True)
    return p.parse_args()


def main():
    args = parse_args()
    device = args.device
    if device not in ("", "cpu") and torch.cuda.is_available() and device == "0":
        device = "0"

    # 相对路径以脚本所在目录的上一级为基准（与生成时的路径约定一致）
    def _resolve(value):
        p = Path(value)
        return str(p if p.is_absolute() else (_SCRIPT_BASE / p).resolve())

    model_arg = _resolve(args.model)
    data_arg = _resolve(args.data)
    print(f"[train] model={model_arg} data={data_arg} device={device} epochs={args.epochs}", flush=True)

    model = YOLO(model_arg)
    project_arg = _resolve(args.project)
    model.train(
        data=data_arg,
        epochs=args.epochs,
        batch=args.batch,
        imgsz=args.imgsz,
        device=device,
        workers=args.workers,
        patience=args.patience,
        optimizer=args.optimizer,
        lr0=args.lr0,
        seed=args.seed,
        project=project_arg,
        name=args.name,
        exist_ok=args.exist_ok,
        verbose=True,
    )

    run_dir = Path(project_arg) / args.name
    best = run_dir / "weights" / "best.pt"
    last = run_dir / "weights" / "last.pt"
    summary = {
        "model": args.model,
        "data": args.data,
        "epochs": args.epochs,
        "batch": args.batch,
        "imgsz": args.imgsz,
        "device": device,
        "run_dir": str(run_dir),
        "best": str(best) if best.exists() else None,
        "last": str(last) if last.exists() else None,
    }

    if best.exists():
        try:
            metrics = YOLO(str(best)).val(data=args.data, split="val", device=device, verbose=False)
            box = getattr(metrics, "box", None)
            if box is not None:
                summary["val"] = {
                    "precision": float(box.mp),
                    "recall": float(box.mr),
                    "map50": float(box.map50),
                    "map50_95": float(box.map),
                }
            # box.ap50 是 numpy 数组，不能用 `or []`（会触发数组真值歧义错误）
            ap50_raw = getattr(box, "ap50", None)
            summary["per_class_map50"] = [float(v) for v in (list(ap50_raw) if ap50_raw is not None else [])]
            names_raw = getattr(metrics, "names", None)
            if isinstance(names_raw, dict):
                summary["class_names"] = [names_raw[k] for k in sorted(names_raw)]
            else:
                summary["class_names"] = list(names_raw) if names_raw else []
        except Exception as exc:  # noqa: BLE001
            summary["val_error"] = str(exc)

    out = run_dir / "train_summary.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[train] 完成，摘要写入 {out}", flush=True)
    print(json.dumps(summary, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
'''


def generate_train_script(
    out_path: Path,
    task_name: str,
    project_root: Path,
    default_model: str,
    default_data: str,
    epochs: int,
    batch: int,
    imgsz: int,
    device: str,
    workers: int,
    patience: int,
    optimizer: str,
    lr0: float,
    seed: int,
    project: str,
    name: str,
) -> Path:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    text = _render(
        TRAIN_SCRIPT_TEMPLATE,
        {
            "TASK_NAME": task_name,
            "SCRIPT_NAME": out_path.name,
            "PROJECT_ROOT": _project_root_default(project_root, out_path),
            "DEFAULT_MODEL": default_model,
            "DEFAULT_DATA": _rel_to_script(default_data, out_path),
            "EPOCHS": epochs,
            "BATCH": batch,
            "IMGSZ": imgsz,
            "DEVICE": device,
            "WORKERS": workers,
            "PATIENCE": patience,
            "OPTIMIZER": optimizer,
            "LR0": lr0,
            "SEED": seed,
            "PROJECT": _rel_to_script(project, out_path),
            "NAME": name,
        },
    )
    out_path.write_text(text, encoding="utf-8")
    return out_path


# ---------------------------------------------------------------------------
# 推理测试脚本
# ---------------------------------------------------------------------------

INFER_SCRIPT_TEMPLATE = '''#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""由 vision_agent 生成的推理测试脚本（@TASK_NAME@）。

对最优权重做一次完整评估:
    1. 在指定划分上跑 val, 拿到整体与分类别指标;
    2. 对测试集图片逐张推理, 保存标注图与 JSON 预测;
    3. 汇总成 metrics.json, 供项目文档引用。

用法:
    python @SCRIPT_NAME@ --model <best.pt> --data <data.yaml> --split test --out <输出目录>
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

_SCRIPT_BASE = Path(__file__).resolve().parent.parent
PROJECT_ROOT = Path(os.environ.get("VA_PROJECT_ROOT", r"@PROJECT_ROOT@"))
if not PROJECT_ROOT.is_absolute():
    PROJECT_ROOT = (_SCRIPT_BASE / PROJECT_ROOT).resolve()
if PROJECT_ROOT.exists():
    import sys

    sys.path.insert(0, str(PROJECT_ROOT))

# --- 让官方 ultralytics 也支持 SEBlock / EMA / CBAM ---
try:
    from vision_agent.attention import install as _install_attention

    _install_attention()
except Exception:  # noqa: BLE001
    pass

from ultralytics import YOLO  # noqa: E402


def _save_annotated(result, path):
    """保存标注图。

    ultralytics 8.0.x 的 Results 没有 save() 方法（8.1 才有），
    所以先试 save()，失败就退回 result.plot() + cv2.imwrite。
    """
    path = str(path)
    try:
        result.save(filename=path)
        return path
    except AttributeError:
        pass
    import cv2

    cv2.imwrite(path, result.plot())
    return path


def parse_args():
    p = argparse.ArgumentParser(description="vision_agent 推理测试")
    p.add_argument("--model", required=True, help="权重路径 (best.pt)")
    p.add_argument("--data", required=True, help="data.yaml 路径")
    p.add_argument("--split", default="test", choices=["train", "val", "test"])
    p.add_argument("--out", default="@DEFAULT_OUT@")
    p.add_argument("--device", default="@DEVICE@")
    p.add_argument("--conf", type=float, default=@CONF@)
    p.add_argument("--iou", type=float, default=@IOU@)
    p.add_argument("--max-samples", type=int, default=40, help="最多保存多少张标注图")
    return p.parse_args()


def main():
    args = parse_args()

    # 相对路径以脚本所在目录的上一级为基准（与生成时的路径约定一致）
    def _resolve(value):
        p = Path(value)
        return str(p if p.is_absolute() else (_SCRIPT_BASE / p).resolve())

    out_dir = Path(_resolve(args.out))
    out_dir.mkdir(parents=True, exist_ok=True)

    model = YOLO(_resolve(args.model))
    data_arg = _resolve(args.data)

    # ---- 1. 整体指标 ----
    metrics = {}
    try:
        res = model.val(data=data_arg, split=args.split, device=args.device, verbose=False)
        box = getattr(res, "box", None)
        # names 可能是 dict（索引->类别名）或 list，取值而不是键
        names_raw = getattr(res, "names", None)
        if isinstance(names_raw, dict):
            names = [names_raw[k] for k in sorted(names_raw)]
        else:
            names = list(names_raw) if names_raw else []
        if box is not None:
            metrics["overall"] = {
                "precision": float(box.mp),
                "recall": float(box.mr),
                "map50": float(box.map50),
                "map50_95": float(box.map),
            }
            per = {}
            # 注意: box.ap50 / box.ap 是 numpy 数组，不能写成 `or []`（会触发
            # "truth value of an array is ambiguous"），必须显式判空。
            ap50_raw = getattr(box, "ap50", None)
            ap_raw = getattr(box, "ap", None)
            ap50 = list(ap50_raw) if ap50_raw is not None else []
            ap = list(ap_raw) if ap_raw is not None else []
            for i, name in enumerate(names):
                per[name] = {
                    "map50": float(ap50[i]) if i < len(ap50) else None,
                    "map50_95": float(ap[i]) if i < len(ap) else None,
                }
            metrics["per_class"] = per
        metrics["split"] = args.split
    except Exception as exc:  # noqa: BLE001
        metrics["val_error"] = str(exc)

    # ---- 2. 逐张推理 ----
    try:
        import yaml

        data_cfg = yaml.safe_load(Path(data_arg).read_text(encoding="utf-8"))
        root = Path(data_cfg.get("path") or ".")
        img_dir = Path(str(root / "images" / args.split))
        images = sorted([p for p in img_dir.glob("*") if p.suffix.lower() in (".jpg", ".jpeg", ".png", ".bmp")])
        sample_dir = out_dir / "samples"
        sample_dir.mkdir(parents=True, exist_ok=True)
        preds = []
        for img_path in images[: args.max_samples]:
            result = model.predict(str(img_path), conf=args.conf, iou=args.iou, device=args.device, verbose=False)[0]
            _save_annotated(result, sample_dir / f"pred_{img_path.stem}.jpg")
            boxes = []
            if result.boxes is not None and len(result.boxes):
                for b in result.boxes:
                    cls_id = int(b.cls[0])
                    xyxy = [float(v) for v in b.xyxy[0]]
                    boxes.append({
                        "class_id": cls_id,
                        "class_name": result.names.get(cls_id, str(cls_id)),
                        "confidence": round(float(b.conf[0]), 4),
                        "xyxy": [round(v, 2) for v in xyxy],
                    })
            preds.append({"image": str(img_path), "boxes": boxes})
        metrics["samples"] = preds
        metrics["sample_dir"] = str(sample_dir)
    except Exception as exc:  # noqa: BLE001
        metrics["predict_error"] = str(exc)

    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(metrics.get("overall", metrics), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
'''


def _rel_to_script(value: str, out_path: Path) -> str:
    """把绝对路径转成「相对生成脚本所在目录」的形式。

    生成的可视化脚本会把它当作默认值写进环境变量。若直接写绝对路径，
    换个机器就失效；写成相对路径后，脚本与它的产物目录一起搬走仍然可用。

    基准是脚本所在目录的上一级（通常是 ``<workspace>/scripts`` 的父级
    即 workspace 根），这样 ``runs/``、``app/`` 这类工作区内路径都能变成
    短相对路径。

    不在同一个根下时（例如权重来自仓库外部），退回绝对路径——
    这种情况保持可用比保持可移植更重要。
    """
    if not value:
        return value
    try:
        base = Path(out_path).resolve().parent.parent
        return str(Path(value).resolve().relative_to(base))
    except (ValueError, OSError):
        return value


def _project_root_default(project_root: Path, out_path: Path | None = None) -> str:
    """生成脚本里 ``PROJECT_ROOT`` 的默认值。

    只保留**工作区内**的相对路径；指向仓库外部的本地 ultralytics 时留空，
    避免把某台机器的绝对路径烤进生成物。留空时脚本走
    「官方 ultralytics + 注意力补丁」这条路；若确实要用本地魔改版，
    设环境变量 ``VA_PROJECT_ROOT`` 即可。
    """
    if not project_root:
        return ""
    try:
        root = Path(project_root).resolve()
    except (TypeError, ValueError, OSError):
        return ""
    if not root.exists():
        return ""
    if out_path is not None:
        base = Path(out_path).resolve().parent.parent
        try:
            return str(root.relative_to(base))
        except ValueError:
            return ""
    return str(root)


def generate_app_script(
    out_path: Path,
    task_name: str,
    model_path: str,
    conf: float,
    iou: float,
    camera_default: int,
    host: str,
    port: int,
    title: str,
    upload_dir: str,
    output_dir: str,
) -> Path:
    """生成可视化系统脚本（基于 vision_agent/webapp.py 的自包含副本）。"""
    webapp_src = Path(__file__).with_name("webapp.py")
    body = webapp_src.read_text(encoding="utf-8")

    # 环境变量默认值必须放在 `from __future__` 之后，否则生成的脚本语法非法
    future_line = "from __future__ import annotations\n"
    # 路径默认值相对脚本自身解析，避免把某台机器的绝对路径烤进生成物。
    # 相对路径统一以 "." 表示「脚本所在目录的上级」，由 _rel_to_script 计算。
    env_block = (
        "\nimport os as _os\n"
        "from pathlib import Path as _Path\n"
        "_BASE = _Path(__file__).resolve().parent.parent\n"
        "\n"
        "def _abs(value):\n"
        "    \"\"\"把默认值解析成绝对路径：相对路径以上一级目录为基准。\"\"\"\n"
        "    if not value:\n"
        "        return value\n"
        "    p = _Path(value)\n"
        "    return str(p if p.is_absolute() else (_BASE / p).resolve())\n"
        "\n"
        # Agent 包所在目录（vision_agent/ 的上一级）。生成的脚本要 import
        # vision_agent.attention 才能装上 SE/EMA/CBAM 补丁；工作区跟仓库不在
        # 同一棵目录树下时，光靠向上回溯找不到，所以把路径写进脚本。
        # 这是「用户机器上」的位置，不是构建机的路径；路径失效时会自动回退到
        # 向上回溯 / VA_AGENT_DIR 环境变量。
        f"_os.environ.setdefault('VA_AGENT_DIR', r'{Path(__file__).resolve().parent.parent}')\n"
        f"_os.environ.setdefault('VA_MODEL', _abs(r'{_rel_to_script(model_path, out_path)}'))\n"
        f"_os.environ.setdefault('VA_CONF', '{conf}')\n"
        f"_os.environ.setdefault('VA_IOU', '{iou}')\n"
        f"_os.environ.setdefault('VA_CAMERA', '{camera_default}')\n"
        f"_os.environ.setdefault('VA_TITLE', '{title}')\n"
        f"_os.environ.setdefault('VA_HOST', '{host}')\n"
        f"_os.environ.setdefault('VA_PORT', '{port}')\n"
        f"_os.environ.setdefault('VA_UPLOAD_DIR', _abs(r'{_rel_to_script(upload_dir, out_path)}'))\n"
        f"_os.environ.setdefault('VA_OUTPUT_DIR', _abs(r'{_rel_to_script(output_dir, out_path)}'))\n\n"
    )
    if future_line in body:
        body = body.replace(future_line, future_line + env_block, 1)
    else:
        body = env_block + body

    header = (
        f"# -*- coding: utf-8 -*-\n"
        f"# 由 vision_agent 生成的可视化检测系统（{task_name}）\n"
        f"# 源模板: vision_agent/webapp.py\n"
        f"#\n"
        f"# 启动:\n"
        f"#   python {Path(out_path).name} --host {host} --port {port}\n"
        f"#   然后浏览器打开 http://{host}:{port}/\n"
        f"#\n"
        f"# 默认配置已写入下方环境变量，可直接 python {Path(out_path).name} 启动。\n"
    )
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(header + body, encoding="utf-8")
    return out_path


def generate_infer_script(
    out_path: Path,
    task_name: str,
    project_root: Path,
    device: str,
    conf: float,
    iou: float,
    default_out: str,
) -> Path:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    text = _render(
        INFER_SCRIPT_TEMPLATE,
        {
            "TASK_NAME": task_name,
            "SCRIPT_NAME": out_path.name,
            "PROJECT_ROOT": _project_root_default(project_root, out_path),
            "DEVICE": device,
            "CONF": conf,
            "IOU": iou,
            "DEFAULT_OUT": _rel_to_script(default_out, out_path),
        },
    )
    out_path.write_text(text, encoding="utf-8")
    return out_path
