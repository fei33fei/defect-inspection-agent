"""Agent 自检用例（不需要 GPU / torch 即可运行）。

    cd vision_agent
    python -m pytest tests -q
    # 或者没有 pytest 时：
    python tests/test_smoke.py
"""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from vision_agent import codegen, convert  # noqa: E402
from vision_agent.config import load_config  # noqa: E402
from vision_agent.state import PipelineState, STATUS_DONE  # noqa: E402
from vision_agent.workspace import Workspace  # noqa: E402

EXAMPLE = ROOT / "config" / "task.example.yaml"
PROJECT_ROOT = ROOT.parent / "ultralytics-main"


# ---------------------------------------------------------------------------
# 配置
# ---------------------------------------------------------------------------


def test_example_config_loads():
    cfg = load_config(EXAMPLE)
    assert cfg.task.name == "steel-defect-detection"
    assert cfg.task.classes and len(cfg.task.classes) == 6
    assert cfg.model.base == "yolov8n"
    assert cfg.model.auto_variants == ["se+ema+cbam"]
    # 路径相对 task.yaml 所在目录解析
    assert cfg.project_root.exists()
    assert cfg.workspace.is_absolute()


def test_config_rejects_bad_split():
    import tempfile as _tf

    with _tf.TemporaryDirectory() as tmp:
        p = Path(tmp) / "task.yaml"
        p.write_text(
            yaml.safe_dump(
                {
                    "task": {"name": "x", "dataset_type": "y"},
                    "data": {"source": "local", "local_dir": ".", "split": {"train": 0.5, "val": 0.2, "test": 0.1}},
                },
                allow_unicode=True,
            ),
            encoding="utf-8",
        )
        try:
            load_config(p)
            raise AssertionError("应当抛出 ConfigError")
        except Exception as exc:
            assert "split" in str(exc)


# ---------------------------------------------------------------------------
# 标注转换
# ---------------------------------------------------------------------------

VOC_XML = """<?xml version="1.0"?>
<annotation>
  <size><width>200</width><height>100</height><depth>3</depth></size>
  <object>
    <name>crazing</name>
    <bndbox><xmin>10</xmin><ymin>20</ymin><xmax>60</xmax><ymax>70</ymax></bndbox>
  </object>
</annotation>
"""


def test_voc_to_yolo_normalization():
    objs = convert.parse_voc.__wrapped__ if hasattr(convert.parse_voc, "__wrapped__") else None  # noqa: F841
    with tempfile.TemporaryDirectory() as tmp:
        xml = Path(tmp) / "a.xml"
        xml.write_text(VOC_XML, encoding="utf-8")
        objects = convert.parse_voc(xml)
        assert len(objects) == 1
        text, warns = convert.voc_to_yolo_label(objects, {"crazing": 0})
        assert not warns
        parts = text.split()
        assert parts[0] == "0"
        cx, cy, w, h = (float(v) for v in parts[1:])
        assert abs(cx - 0.175) < 1e-6  # (10+60)/2/200
        assert abs(cy - 0.45) < 1e-6  # (20+70)/2/100
        assert abs(w - 0.25) < 1e-6
        assert abs(h - 0.5) < 1e-6


def test_coco_to_yolo():
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        coco = {
            "images": [{"id": 1, "file_name": "x.jpg", "width": 100, "height": 100}],
            "annotations": [{"id": 1, "image_id": 1, "category_id": 1, "bbox": [10, 20, 30, 40]}],
            "categories": [{"id": 1, "name": "scratch"}],
        }
        (tmp / "instances.json").write_text(json.dumps(coco), encoding="utf-8")
        written, warns = convert.coco_to_yolo(tmp / "instances.json", tmp / "labels", {"scratch": 3})
        assert written == 1
        assert not warns
        text = (tmp / "labels" / "x.txt").read_text(encoding="utf-8").strip()
        parts = text.split()
        assert parts[0] == "3"
        assert abs(float(parts[1]) - 0.25) < 1e-6  # (10+15)/100
        assert abs(float(parts[2]) - 0.40) < 1e-6  # (20+20)/100


def test_validate_yolo_label_catches_out_of_range():
    ok, _ = convert.validate_yolo_label("0 0.5 0.5 0.2 0.2", 6)
    assert ok
    ok, problems = convert.validate_yolo_label("9 0.5 0.5 0.2 0.2", 6)
    assert not ok and any("越界" in p for p in problems)
    ok, problems = convert.validate_yolo_label("0 0.99 0.5 0.4 0.2", 6)
    assert not ok and any("超出图像边界" in p for p in problems)


def test_detect_annotation_format_on_labels_dir():
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        (root / "images" / "train").mkdir(parents=True)
        (root / "labels" / "train").mkdir(parents=True)
        (root / "labels" / "train" / "a.txt").write_text("0 0.5 0.5 0.2 0.2\n", encoding="utf-8")
        fmt, evidence = convert.detect_annotation_format(root)
        assert fmt == "yolo_txt"
        assert "YOLO" in evidence


# ---------------------------------------------------------------------------
# 模型 YAML 生成
# ---------------------------------------------------------------------------


def test_insert_modules_remaps_head_indices():
    base_yaml = PROJECT_ROOT / "ultralytics" / "cfg" / "models" / "v8" / "yolov8.yaml"
    if not base_yaml.exists():
        return  # 没有本地仓库时跳过
    result = codegen.insert_modules_into_yaml(
        base_yaml.read_text(encoding="utf-8"), ["se", "ema", "cbam"], insert_after_c2f=3
    )
    data = yaml.safe_load(result["text"])
    backbone = data["backbone"]
    head = data["head"]
    base_len = len(yaml.safe_load(base_yaml.read_text(encoding="utf-8"))["backbone"])

    # 每个插入点插了 3 个模块，共 9 个
    assert sum(1 for row in backbone if row[2] in {"SEBlock", "EMA", "CBAM"}) == 9
    # backbone 变长
    assert len(backbone) == base_len + 9

    # head 里的绝对引用有两种语义：
    #   ref < backbone_len        -> 指向 backbone 某一层
    #   ref >= backbone_len       -> 指向 head 内部的层（全局下标）
    # 插入模块后两者都要正确平移，且不能越界。
    total = len(backbone) + len(head)
    head_refs = []
    for row in head:
        frm = row[0]
        refs = frm if isinstance(frm, list) else [frm]
        for ref in refs:
            if isinstance(ref, int) and ref >= 0:
                assert ref < total, f"head 引用 {ref} 越界（总层数 {total}）"
                head_refs.append(ref)

    # 反向引用（-1 / -2 / -3）保持不变
    for row in head:
        frm = row[0]
        refs = frm if isinstance(frm, list) else [frm]
        for ref in refs:
            if isinstance(ref, int) and ref < 0:
                assert ref in (-1, -2, -3)

    # 指向 backbone 的引用必须落在「插入后」的 backbone 范围内，
    # 并且指向的层仍然是原来那一层（P3 / P4 / SPPF）。
    base_data = yaml.safe_load(base_yaml.read_text(encoding="utf-8"))
    base_backbone = base_data["backbone"]
    old_p3 = next(i for i, r in enumerate(base_backbone) if r[2] == "C2f" and r[3][0] == 256)
    old_sppf = next(i for i, r in enumerate(base_backbone) if r[2] == "SPPF")
    # 重新算一遍映射：old -> new
    mapping, new_idx = {}, 0
    c2f_seen = 0
    for i, r in enumerate(base_backbone):
        mapping[i] = new_idx
        new_idx += 1
        if r[2] == "C2f":
            c2f_seen += 1
            if c2f_seen <= 3:
                new_idx += 3
    assert mapping[old_p3] in head_refs, "P3 层引用没有正确重映射"
    assert mapping[old_sppf] in head_refs, "SPPF 层引用没有正确重映射"

    # Detect 的输入层：基础 YAML 里用的是全局绝对索引，插入后必须整体平移
    base_head = base_data["head"]
    base_detect = base_head[-1]
    base_refs = base_detect[0] if isinstance(base_detect[0], list) else [base_detect[0]]
    old_bl = len(base_backbone)
    expected = []
    for ref in base_refs:
        if isinstance(ref, int) and ref >= old_bl:
            expected.append(len(backbone) + (ref - old_bl))
        else:
            expected.append(ref)
    detect = head[-1]
    assert detect[2] == "Detect"
    refs = detect[0] if isinstance(detect[0], list) else [detect[0]]
    assert len(refs) == 3
    assert refs == expected, f"Detect 输入层平移错误: {refs} != {expected}"


def test_variant_naming():
    assert codegen.variant_name("yolov8n", ["se", "ema", "cbam"]) == "yolov8n-se-ema-cbam"
    assert codegen.resolve_module("SEBlock") == "SEBlock"
    assert codegen.resolve_module("se") == "SEBlock"


# ---------------------------------------------------------------------------
# 工作区与状态
# ---------------------------------------------------------------------------


def test_workspace_creates_all_dirs():
    with tempfile.TemporaryDirectory() as tmp:
        ws = Workspace(Path(tmp) / "ws")
        ws.ensure()
        for d in ws.all_dirs():
            assert d.exists(), d
        assert ws.data_yaml.parent.exists()
        assert ws.rel(ws.runs) == "runs"


def test_state_roundtrip():
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "state.json"
        st = PipelineState(path)
        st.start("s1", "测试")
        st.finish("s1", STATUS_DONE, artifacts=[Path("a.md")], note="完成", data={"n": 3})
        st.save()
        st2 = PipelineState(path)
        assert st2.is_done("s1")
        assert st2.get("s1", "n") == 3
        assert st2.status_of("s9") == "pending"


def test_generated_scripts_are_valid_python():
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        train = codegen.generate_train_script(
            out_path=tmp / "train_x.py",
            task_name="x",
            project_root=PROJECT_ROOT,
            default_model="yolov8n.yaml",
            default_data="data.yaml",
            epochs=1,
            batch=2,
            imgsz=320,
            device="cpu",
            workers=0,
            patience=10,
            optimizer="auto",
            lr0=0.01,
            seed=0,
            project="runs",
            name="x",
        )
        compile(train.read_text(encoding="utf-8"), str(train), "exec")

        infer = codegen.generate_infer_script(
            out_path=tmp / "infer_x.py",
            task_name="x",
            project_root=PROJECT_ROOT,
            device="cpu",
            conf=0.25,
            iou=0.45,
            default_out="out",
        )
        compile(infer.read_text(encoding="utf-8"), str(infer), "exec")

        app = codegen.generate_app_script(
            out_path=tmp / "app_x.py",
            task_name="x",
            model_path="best.pt",
            conf=0.25,
            iou=0.45,
            camera_default=0,
            host="127.0.0.1",
            port=8000,
            title="测试",
            upload_dir="uploads",
            output_dir="outputs",
        )
        compile(app.read_text(encoding="utf-8"), str(app), "exec")


def test_selftest_suite_all_green():
    """把内置自检套件也纳入 pytest（单一事实来源）。"""
    from vision_agent.selftest import run_selftest

    ok, checks = run_selftest()
    failed = [f"{c.name}: {c.message}" for c in checks if not c.ok]
    assert ok, "\n".join(failed)
    assert len(checks) >= 10, f"自检项太少: {len(checks)}"


if __name__ == "__main__":
    failures = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn()
                print(f"  PASS  {name}")
            except Exception as exc:  # noqa: BLE001
                failures += 1
                print(f"  FAIL  {name}: {exc}")
    print(f"\n失败 {failures} 个")
    sys.exit(1 if failures else 0)
