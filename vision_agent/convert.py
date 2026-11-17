"""标注格式转换与数据集构建（工作流第 5 步）。

支持的输入格式：
    voc_xml    Pascal VOC XML 目录（NEU-DET 原始格式就是这种）
    coco_json  COCO instances_*.json
    yolo_txt   已经是 YOLO 归一化 txt

输出统一为 YOLO 检测格式::

    processed/
      images/{train,val,test}/*.jpg
      labels/{train,val,test}/*.txt   每行 "cls cx cy w h"，坐标已按图像宽高归一化
      classes.txt
      class_mapping.json
      dataset_stats.json

关键约束（YOLO 检测头的硬性要求）：
    - cx, cy, w, h 必须在 [0, 1]；
    - cx ± w/2 必须落在 [0, 1] 内（否则框跑到图外）；
    - 类别索引必须是连续整数；
    - 图片与标签必须同名同扩展名（ Ultralytics 按 stem 配对）。

另外：如果原始数据集已经带 train/val 目录划分，默认**沿用原有划分**，
不重新随机切分——重新切分会让此前的实验结果失去可比性。
"""

from __future__ import annotations

import json
import os
import random
import shutil
import xml.etree.ElementTree as ET
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}


@dataclass
class BuildResult:
    ok: bool = True
    format: str = ""
    out_dir: str = ""
    total_images: int = 0
    total_boxes: int = 0
    splits: Dict[str, Dict[str, int]] = field(default_factory=dict)
    classes: List[str] = field(default_factory=list)
    class_mapping: Dict[str, int] = field(default_factory=dict)
    per_class_counts: Dict[str, int] = field(default_factory=dict)
    warnings: List[str] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)
    data_yaml: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# ---------------------------------------------------------------------------
# 格式探测
# ---------------------------------------------------------------------------


def detect_annotation_format(root: Path) -> Tuple[str, str]:
    """探测数据集标注格式，返回 (format, 证据描述)。

    判定顺序：VOC XML → COCO JSON → YOLO txt。
    YOLO 标签的判定不依赖文件名，而是看路径里有没有 `labels` 这一层目录
    （工业数据集常见 `images/train` + `labels/train` 的组织方式）。
    """
    root = Path(root)
    if not root.exists():
        return "unknown", "目录不存在"

    xml_files = list(root.rglob("*.xml"))
    json_files = [
        p for p in root.rglob("*.json") if p.name.startswith("instances") or "coco" in p.name.lower()
    ]
    txt_files = []
    for p in root.rglob("*.txt"):
        parts = [part.lower() for part in p.parts]
        if "labels" in parts or "label" in parts or "annotations" in parts:
            txt_files.append(p)
    if not txt_files:
        # 兜底：任何 .txt（排除明显的配置/说明文件）
        txt_files = [p for p in root.rglob("*.txt") if p.name.lower() not in {"classes.txt", "readme.txt"}]

    if xml_files:
        return "voc_xml", f"发现 {len(xml_files)} 个 XML 标注文件"
    if json_files:
        return "coco_json", f"发现 {len(json_files)} 个 COCO JSON"
    if txt_files:
        return "yolo_txt", f"发现 {len(txt_files)} 个 YOLO txt 标签"
    return "unknown", "未发现可识别的标注文件"


def discover_images(root: Path) -> List[Path]:
    return sorted(p for p in Path(root).rglob("*") if p.suffix.lower() in IMAGE_EXTS)


# ---------------------------------------------------------------------------
# VOC XML -> YOLO
# ---------------------------------------------------------------------------


def parse_voc(xml_path: Path) -> List[Dict[str, Any]]:
    """解析一个 VOC XML，返回 [{name, bbox(xmin,ymin,xmax,ymax), w, h, dw, dh}]。"""
    try:
        tree = ET.parse(xml_path)
    except ET.ParseError:
        return []
    root = tree.getroot()

    size = root.find("size")
    img_w = int(float(size.find("width").text)) if size is not None and size.find("width") is not None else 0
    img_h = int(float(size.find("height").text)) if size is not None and size.find("height") is not None else 0

    objects: List[Dict[str, Any]] = []
    for obj in root.findall("object"):
        name_el = obj.find("name")
        name = (name_el.text or "").strip() if name_el is not None else ""
        if not name:
            continue
        box = obj.find("bndbox")
        if box is None:
            continue
        try:
            xmin = float(box.find("xmin").text)
            ymin = float(box.find("ymin").text)
            xmax = float(box.find("xmax").text)
            ymax = float(box.find("ymax").text)
        except (AttributeError, TypeError, ValueError):
            continue
        objects.append({"name": name, "bbox": [xmin, ymin, xmax, ymax], "w": img_w, "h": img_h})
    return objects


def voc_to_yolo_label(objects: List[Dict[str, Any]], class_to_idx: Dict[str, int]) -> Tuple[str, List[str]]:
    """把 VOC 对象转成 YOLO 标注文本，返回 (文本, 警告列表)。"""
    lines: List[str] = []
    warns: List[str] = []
    for obj in objects:
        name = obj["name"]
        if name not in class_to_idx:
            warns.append(f"未知类别 '{name}'（已跳过）")
            continue
        xmin, ymin, xmax, ymax = obj["bbox"]
        img_w, img_h = obj["w"], obj["h"]
        if not img_w or not img_h:
            warns.append(f"{name}: 图像尺寸缺失，无法归一化")
            continue
        # VOC 坐标是 1-based 的，且可能有 0 边界，先裁剪再归一化
        xmin = max(0.0, min(float(xmin), img_w))
        xmax = max(0.0, min(float(xmax), img_w))
        ymin = max(0.0, min(float(ymin), img_h))
        ymax = max(0.0, min(float(ymax), img_h))
        if xmax <= xmin or ymax <= ymin:
            warns.append(f"{name}: 非法框 {obj['bbox']}")
            continue
        cx = ((xmin + xmax) / 2.0) / img_w
        cy = ((ymin + ymax) / 2.0) / img_h
        nw = (xmax - xmin) / img_w
        nh = (ymax - ymin) / img_h
        if not (0 <= cx <= 1 and 0 <= cy <= 1 and 0 < nw <= 1 and 0 < nh <= 1):
            warns.append(f"{name}: 归一化后越界 {cx:.4f},{cy:.4f},{nw:.4f},{nh:.4f}")
            continue
        lines.append(f"{class_to_idx[name]} {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}")
    return "\n".join(lines), warns


# ---------------------------------------------------------------------------
# COCO -> YOLO
# ---------------------------------------------------------------------------


def coco_to_yolo(
    coco_json: Path,
    out_labels_dir: Path,
    class_to_idx: Dict[str, int],
    img_id_to_name: Optional[Dict[int, str]] = None,
) -> Tuple[int, List[str]]:
    """COCO JSON -> YOLO 标签目录。返回 (写出的标签数, 警告)。"""
    warns: List[str] = []
    data = json.loads(Path(coco_json).read_text(encoding="utf-8"))
    images = data.get("images", []) or []
    annotations = data.get("annotations", []) or []
    categories = data.get("categories", []) or []

    cat_name = {c["id"]: c.get("name", "") for c in categories}
    id_map: Dict[int, int] = {}
    for cid, cname in cat_name.items():
        if cname in class_to_idx:
            id_map[cid] = class_to_idx[cname]
        else:
            warns.append(f"COCO 类别 '{cname}' 不在类别表中（已跳过）")

    out_labels_dir = Path(out_labels_dir)
    out_labels_dir.mkdir(parents=True, exist_ok=True)

    per_image: Dict[int, List[str]] = defaultdict(list)
    for ann in annotations:
        img_id = ann.get("image_id")
        bbox = ann.get("bbox") or []
        if img_id is None or len(bbox) < 4:
            continue
        cid = ann.get("category_id")
        if cid not in id_map:
            continue
        x, y, w, h = bbox[:4]
        img = next((im for im in images if im.get("id") == img_id), None)
        if img is None:
            continue
        iw = float(img.get("width") or 0)
        ih = float(img.get("height") or 0)
        if not iw or not ih:
            continue
        cx = (x + w / 2.0) / iw
        cy = (y + h / 2.0) / ih
        nw, nh = w / iw, h / ih
        if not (0 <= cx <= 1 and 0 <= cy <= 1 and 0 < nw <= 1 and 0 < nh <= 1):
            warns.append(f"image_id={img_id} 归一化后越界")
            continue
        per_image[img_id].append(f"{id_map[cid]} {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}")

    written = 0
    for img in images:
        img_id = img.get("id")
        file_name = img.get("file_name") or (img_id_to_name or {}).get(img_id, "")
        if not file_name:
            continue
        stem = Path(file_name).stem
        target = out_labels_dir / f"{stem}.txt"
        lines = per_image.get(img_id, [])
        target.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
        written += 1
    return written, warns


# ---------------------------------------------------------------------------
# YOLO txt 校验
# ---------------------------------------------------------------------------


def validate_yolo_label(text: str, num_classes: int) -> Tuple[bool, List[str]]:
    """校验一行 YOLO 标签，���回 (是否合法, 问题列表)。"""
    problems: List[str] = []
    parts = text.split()
    if len(parts) != 5:
        return False, [f"字段数应为 5，实际 {len(parts)}"]
    try:
        cls = int(float(parts[0]))
        vals = [float(p) for p in parts[1:]]
    except ValueError:
        return False, ["存在非数值字段"]
    if cls < 0 or cls >= num_classes:
        problems.append(f"类别索引 {cls} 越界 (0-{num_classes - 1})")
    cx, cy, w, h = vals
    for name, v in (("cx", cx), ("cy", cy), ("w", w), ("h", h)):
        if not (0.0 <= v <= 1.0):
            problems.append(f"{name}={v:.4f} 超出 [0,1]")
    if w <= 0 or h <= 0:
        problems.append("宽高必须为正")
    if cx - w / 2 < -1e-6 or cx + w / 2 > 1 + 1e-6:
        problems.append("框横向超出图像边界")
    if cy - h / 2 < -1e-6 or cy + h / 2 > 1 + 1e-6:
        problems.append("框纵向超出图像边界")
    return (not problems), problems


# ---------------------------------------------------------------------------
# 数据集构建
# ---------------------------------------------------------------------------


def _link_or_copy(src: Path, dst: Path, mode: str) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        return
    if mode == "symlink":
        try:
            os.symlink(src.resolve(), dst)
            return
        except (OSError, NotImplementedError):
            pass
    if mode in {"link", "copy"}:
        try:
            os.link(src, dst)
            return
        except (OSError, NotImplementedError):
            pass
    shutil.copy2(src, dst)


def _existing_split_dirs(root: Path) -> Dict[str, Path]:
    """识别已有的 train/val/test 划分目录。"""
    found: Dict[str, Path] = {}
    for name in ("train", "val", "test", "valid", "validation"):
        for candidate in (root / "images" / name, root / name, root / "Images" / name):
            if candidate.exists() and any(candidate.iterdir()):
                key = {"valid": "val", "validation": "val"}.get(name, name)
                found.setdefault(key, candidate)
                break
    return found


def build_dataset(
    raw_dir: Path,
    out_dir: Path,
    annotation_format: str = "auto",
    classes: Optional[List[str]] = None,
    split: Optional[Dict[str, float]] = None,
    seed: int = 0,
    copy_mode: str = "link",
    respect_existing_split: bool = True,
) -> BuildResult:
    """把原始数据集转换成标准 YOLO 检测数据集。"""
    raw_dir = Path(raw_dir)
    out_dir = Path(out_dir)
    result = BuildResult(out_dir=str(out_dir))

    if not raw_dir.exists():
        result.ok = False
        result.errors.append(f"原始数据目录不存在: {raw_dir}")
        return result

    fmt, evidence = detect_annotation_format(raw_dir)
    if annotation_format != "auto":
        fmt = annotation_format
    result.format = fmt
    if fmt == "unknown":
        result.ok = False
        result.errors.append(f"无法识别标注格式（{evidence}）")
        return result

    # --- 1. 收集类别 ---
    class_list = list(classes or [])
    if fmt == "voc_xml":
        xml_files = sorted(raw_dir.rglob("*.xml"))
        found_names: List[str] = []
        for xf in xml_files[:500]:
            for obj in parse_voc(xf):
                if obj["name"] not in found_names:
                    found_names.append(obj["name"])
        if not class_list:
            class_list = sorted(found_names)
    elif fmt == "coco_json":
        json_files = [p for p in raw_dir.rglob("*.json") if p.name.startswith("instances") or "coco" in p.name.lower()]
        if json_files:
            data = json.loads(json_files[0].read_text(encoding="utf-8"))
            cats = sorted((c.get("name", "") for c in data.get("categories", [])), key=str.lower)
            if not class_list:
                class_list = [c for c in cats if c]
    else:
        # YOLO txt: 从 data.yaml / classes.txt 里找，找不到就先按索引占位
        for cand in (raw_dir / "classes.txt", raw_dir.parent / "classes.txt"):
            if cand.exists():
                lines = [l.strip() for l in cand.read_text(encoding="utf-8").splitlines() if l.strip()]
                if lines and not class_list:
                    class_list = lines
                break
        if not class_list:
            max_idx = 0
            for lbl in list(raw_dir.rglob("*.txt"))[:500]:
                for line in lbl.read_text(encoding="utf-8", errors="replace").splitlines():
                    if line.strip():
                        try:
                            max_idx = max(max_idx, int(float(line.split()[0])))
                        except (ValueError, IndexError):
                            pass
            class_list = [f"class_{i}" for i in range(max_idx + 1)]

    if not class_list:
        result.ok = False
        result.errors.append("无法确定类别列表，请在 task.classes 中显式指定")
        return result

    class_to_idx = {name: i for i, name in enumerate(class_list)}
    result.classes = class_list
    result.class_mapping = class_to_idx

    # --- 2. 准备输出目录 ---
    for split_name in ("train", "val", "test"):
        (out_dir / "images" / split_name).mkdir(parents=True, exist_ok=True)
        (out_dir / "labels" / split_name).mkdir(parents=True, exist_ok=True)

    # --- 3. 分支处理 ---
    split_ratios = split or {"train": 0.8, "val": 0.1, "test": 0.1}
    existing = _existing_split_dirs(raw_dir) if respect_existing_split else {}

    if fmt == "yolo_txt" and existing:
        result.warnings.append(f"沿用原始划分: {sorted(existing)}（未重新随机切分，保证与历史实验可比）")
        _copy_existing_split(raw_dir, existing, out_dir, copy_mode, result, num_classes=len(class_list))
    elif fmt == "voc_xml" and existing:
        result.warnings.append(f"沿用原始划分: {sorted(existing)}")
        _convert_voc_with_split(raw_dir, existing, out_dir, copy_mode, result, class_to_idx, len(class_list))
    else:
        _convert_flat_with_split(raw_dir, out_dir, fmt, split_ratios, seed, copy_mode, result, class_to_idx, len(class_list))

    # --- 4. 统计与落盘 ---
    stats = collect_stats(out_dir, class_list)
    result.total_images = stats["total_images"]
    result.total_boxes = stats["total_boxes"]
    result.splits = stats["splits"]
    result.per_class_counts = stats["per_class_counts"]

    (out_dir / "classes.txt").write_text("\n".join(class_list) + "\n", encoding="utf-8")
    (out_dir / "class_mapping.json").write_text(
        json.dumps(class_to_idx, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    (out_dir / "dataset_stats.json").write_text(
        json.dumps(stats, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return result


def _copy_existing_split(
    raw_dir: Path,
    existing: Dict[str, Path],
    out_dir: Path,
    copy_mode: str,
    result: BuildResult,
    num_classes: int,
) -> None:
    """YOLO txt 已有划分：直接搬运并校验。"""
    for split_name, src_dir in existing.items():
        for img in sorted(p for p in src_dir.rglob("*") if p.suffix.lower() in IMAGE_EXTS):
            stem = img.stem
            dst_img = out_dir / "images" / split_name / img.name
            _link_or_copy(img, dst_img, copy_mode)
            # 标签可能在 images 同级 labels/ 下
            for cand in (
                raw_dir / "labels" / split_name / f"{stem}.txt",
                src_dir.parent.parent / "labels" / split_name / f"{stem}.txt",
                src_dir / f"{stem}.txt",
            ):
                if cand.exists():
                    _link_or_copy(cand, out_dir / "labels" / split_name / f"{stem}.txt", copy_mode)
                    break
            else:
                (out_dir / "labels" / split_name / f"{stem}.txt").write_text("", encoding="utf-8")
                result.warnings.append(f"{split_name}/{img.name}: 找不到标签文件，已生成空标签")


def _convert_voc_with_split(
    raw_dir: Path,
    existing: Dict[str, Path],
    out_dir: Path,
    copy_mode: str,
    result: BuildResult,
    class_to_idx: Dict[str, int],
    num_classes: int,
) -> None:
    """VOC XML 已有划分：按原划分转换。"""
    for split_name, src_dir in existing.items():
        for img in sorted(p for p in src_dir.rglob("*") if p.suffix.lower() in IMAGE_EXTS):
            xml_candidates = [
                src_dir / f"{img.stem}.xml",
                raw_dir / "Annotations" / split_name / f"{img.stem}.xml",
                raw_dir / "annotations" / split_name / f"{img.stem}.xml",
            ]
            xml_path = next((c for c in xml_candidates if c.exists()), None)
            _link_or_copy(img, out_dir / "images" / split_name / img.name, copy_mode)
            if xml_path is None:
                (out_dir / "labels" / split_name / f"{img.stem}.txt").write_text("", encoding="utf-8")
                result.warnings.append(f"{split_name}/{img.name}: 缺少 XML 标注")
                continue
            objects = parse_voc(xml_path)
            text, warns = voc_to_yolo_label(objects, class_to_idx)
            (out_dir / "labels" / split_name / f"{img.stem}.txt").write_text(
                text + ("\n" if text else ""), encoding="utf-8"
            )
            result.warnings.extend(f"{split_name}/{img.name}: {w}" for w in warns)


def _convert_flat_with_split(
    raw_dir: Path,
    out_dir: Path,
    fmt: str,
    split_ratios: Dict[str, float],
    seed: int,
    copy_mode: str,
    result: BuildResult,
    class_to_idx: Dict[str, int],
    num_classes: int,
) -> None:
    """无预划分：随机切分后转换。"""
    if fmt == "voc_xml":
        pairs: List[Tuple[Path, Optional[Path]]] = []
        for xml_path in sorted(raw_dir.rglob("*.xml")):
            stem = xml_path.stem
            img = next(
                (
                    p
                    for p in raw_dir.rglob(f"{stem}.*")
                    if p.suffix.lower() in IMAGE_EXTS
                ),
                None,
            )
            pairs.append((img, xml_path) if img else (None, xml_path))
        pairs = [(img, xml) for img, xml in pairs if img]
    elif fmt == "coco_json":
        json_files = [
            p for p in raw_dir.rglob("*.json") if p.name.startswith("instances") or "coco" in p.name.lower()
        ]
        if not json_files:
            result.errors.append("未找到 COCO JSON")
            result.ok = False
            return
        coco_path = json_files[0]
        data = json.loads(coco_path.read_text(encoding="utf-8"))
        images_dir = raw_dir
        pairs = []
        for im in data.get("images", []) or []:
            fn = im.get("file_name", "")
            img = images_dir / fn
            if not img.exists():
                cand = next(raw_dir.rglob(Path(fn).name), None)
                img = cand if cand else img
            pairs.append((img, coco_path))
    else:
        pairs = []
        for lbl in sorted(raw_dir.rglob("*.txt")):
            stem = lbl.stem
            img = next((p for p in raw_dir.rglob(f"{stem}.*") if p.suffix.lower() in IMAGE_EXTS), None)
            if img:
                pairs.append((img, lbl))

    if not pairs:
        result.ok = False
        result.errors.append("没有找到可用的图像样本")
        return

    rng = random.Random(seed)
    pairs.sort(key=lambda p: str(p[0]))
    rng.shuffle(pairs)

    names = [n for n in ("train", "val", "test") if split_ratios.get(n, 0) > 0]
    total = len(pairs)
    start = 0
    assign: Dict[str, List[Tuple[Path, Optional[Path]]]] = {}
    for i, name in enumerate(names):
        ratio = split_ratios.get(name, 0)
        if i == len(names) - 1:
            end = total
        else:
            end = start + int(round(total * ratio))
        assign[name] = pairs[start:end]
        start = end

    for split_name, items in assign.items():
        img_dir = out_dir / "images" / split_name
        lbl_dir = out_dir / "labels" / split_name
        img_dir.mkdir(parents=True, exist_ok=True)
        lbl_dir.mkdir(parents=True, exist_ok=True)
        for img, ann in items:
            dst_img = img_dir / img.name
            _link_or_copy(img, dst_img, copy_mode)
            stem = img.stem
            if fmt == "voc_xml" and ann is not None:
                objects = parse_voc(ann)
                text, warns = voc_to_yolo_label(objects, class_to_idx)
                lbl_dir.joinpath(f"{stem}.txt").write_text(text + ("\n" if text else ""), encoding="utf-8")
                result.warnings.extend(f"{split_name}/{img.name}: {w}" for w in warns)
            elif fmt == "coco_json" and ann is not None:
                pass  # COCO 统一在下面批量处理
            else:
                src_lbl = ann
                if src_lbl is not None and Path(src_lbl).exists():
                    _link_or_copy(Path(src_lbl), lbl_dir / f"{stem}.txt", copy_mode)
                else:
                    lbl_dir.joinpath(f"{stem}.txt").write_text("", encoding="utf-8")

        if fmt == "coco_json":
            coco_path = next((a for _, a in items if a is not None), None)
            if coco_path:
                _write_coco_labels(coco_path, lbl_dir, class_to_idx, result)


def _write_coco_labels(coco_path: Path, lbl_dir: Path, class_to_idx: Dict[str, int], result: BuildResult) -> None:
    """把 COCO 标注按 image stem 写入标签目录。"""
    data = json.loads(Path(coco_path).read_text(encoding="utf-8"))
    cat_name = {c["id"]: c.get("name", "") for c in data.get("categories", []) or []}
    id_map = {cid: class_to_idx[name] for cid, name in cat_name.items() if name in class_to_idx}
    img_meta = {im.get("id"): im for im in data.get("images", []) or []}
    per_image: Dict[str, List[str]] = defaultdict(list)
    for ann in data.get("annotations", []) or []:
        img_id = ann.get("image_id")
        bbox = ann.get("bbox") or []
        meta = img_meta.get(img_id)
        if meta is None or len(bbox) < 4 or ann.get("category_id") not in id_map:
            continue
        iw = float(meta.get("width") or 0)
        ih = float(meta.get("height") or 0)
        if not iw or not ih:
            continue
        x, y, w, h = bbox[:4]
        cx, cy = (x + w / 2) / iw, (y + h / 2) / ih
        nw, nh = w / iw, h / ih
        if not (0 <= cx <= 1 and 0 <= cy <= 1 and 0 < nw <= 1 and 0 < nh <= 1):
            continue
        stem = Path(meta.get("file_name", "")).stem
        per_image[stem].append(f"{id_map[ann['category_id']]} {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}")
    for stem, lines in per_image.items():
        lbl_dir.joinpath(f"{stem}.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")


# ---------------------------------------------------------------------------
# 统计与校验
# ---------------------------------------------------------------------------


def collect_stats(out_dir: Path, classes: List[str]) -> Dict[str, Any]:
    """统计数据集：各划分图片数、框数、类别分布。"""
    out_dir = Path(out_dir)
    splits: Dict[str, Dict[str, int]] = {}
    class_counter: Counter = Counter()
    total_images = 0
    total_boxes = 0
    for split_name in ("train", "val", "test"):
        img_dir = out_dir / "images" / split_name
        lbl_dir = out_dir / "labels" / split_name
        images = list(img_dir.glob("*")) if img_dir.exists() else []
        images = [p for p in images if p.suffix.lower() in IMAGE_EXTS]
        boxes = 0
        for lbl in (lbl_dir.glob("*.txt") if lbl_dir.exists() else []):
            for line in lbl.read_text(encoding="utf-8", errors="replace").splitlines():
                if not line.strip():
                    continue
                boxes += 1
                try:
                    idx = int(float(line.split()[0]))
                    if 0 <= idx < len(classes):
                        class_counter[classes[idx]] += 1
                except (ValueError, IndexError):
                    continue
        splits[split_name] = {"images": len(images), "boxes": boxes}
        total_images += len(images)
        total_boxes += boxes
    return {
        "total_images": total_images,
        "total_boxes": total_boxes,
        "splits": splits,
        "per_class_counts": dict(class_counter),
        "classes": classes,
    }


def validate_dataset(out_dir: Path, classes: List[str]) -> Dict[str, Any]:
    """逐标签校验：坐标范围、类别越界、图签配对。"""
    out_dir = Path(out_dir)
    problems: List[str] = []
    duplicate_examples: List[str] = []
    duplicates = 0
    checked = 0
    for split_name in ("train", "val", "test"):
        img_dir = out_dir / "images" / split_name
        lbl_dir = out_dir / "labels" / split_name
        if not img_dir.exists():
            continue
        imgs = {p.stem for p in img_dir.glob("*") if p.suffix.lower() in IMAGE_EXTS}
        lbls = {p.stem for p in lbl_dir.glob("*.txt")} if lbl_dir.exists() else set()
        for stem in sorted(imgs - lbls):
            problems.append(f"{split_name}/{stem}: 图片缺少标签文件")
        for stem in sorted(lbls - imgs):
            problems.append(f"{split_name}/{stem}: 标签没有对应图片")
        for lbl in sorted(lbl_dir.glob("*.txt")) if lbl_dir.exists() else []:
            seen_rows: set = set()
            for lineno, line in enumerate(lbl.read_text(encoding="utf-8", errors="replace").splitlines(), 1):
                if not line.strip():
                    continue
                checked += 1
                ok, msgs = validate_yolo_label(line, len(classes))
                if not ok:
                    problems.append(f"{split_name}/{lbl.name}:{lineno}: {'; '.join(msgs)}")
                # 完全重复的标注行（ultralytics 训练时会自动去重，但值得提前告知）
                key = " ".join(line.split())
                if key in seen_rows:
                    duplicates += 1
                    if len(duplicate_examples) < 20:
                        duplicate_examples.append(f"{split_name}/{lbl.name}:{lineno} 与前面的标注完全重复")
                seen_rows.add(key)
    return {
        "checked_labels": checked,
        "problem_count": len(problems),
        "problems": problems[:200],
        "duplicate_labels": duplicates,
        "duplicate_examples": duplicate_examples,
        "ok": not problems,
    }


def write_data_yaml(out_dir: Path, yaml_path: Path, classes: List[str], name: str = "dataset") -> Path:
    """写 ultralytics 用的 data.yaml。path 用绝对路径，避免训练时找不到。"""
    import yaml as _yaml

    out_dir = Path(out_dir).resolve()
    yaml_path = Path(yaml_path)
    yaml_path.parent.mkdir(parents=True, exist_ok=True)
    data = {
        "path": str(out_dir),
        "train": "images/train",
        "val": "images/val",
        "test": "images/test",
        "names": {i: c for i, c in enumerate(classes)},
    }
    yaml_path.write_text(
        "# 由 vision_agent 自动生成\n" + _yaml.safe_dump(data, allow_unicode=True, sort_keys=False),
        encoding="utf-8",
    )
    return yaml_path
