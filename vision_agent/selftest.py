"""Agent 自检：不依赖 pytest，直接跑一遍核心能力的冒烟检查。

    python -m vision_agent self-test

覆盖：
    配置解析、标注转换（VOC/COCO/YOLO）、坐标校验、模型 YAML 索引重映射、
    生成脚本语法、模板模式研究质量、工作区与状态机、可视化接口（假模型）。
"""

from __future__ import annotations

import json
import tempfile
import traceback
from pathlib import Path
from typing import Any, Callable, Dict, List, Tuple

from . import codegen, convert
from .config import load_config
from .state import PipelineState, STATUS_DONE
from .workspace import Workspace

ROOT = Path(__file__).resolve().parent.parent
EXAMPLE = ROOT / "config" / "task.example.yaml"
PROJECT_ROOT = ROOT.parent / "ultralytics-main"


class Check:
    """一条自检结果。"""

    def __init__(self, name: str):
        self.name = name
        self.ok = False
        self.message = ""

    def __repr__(self) -> str:  # pragma: no cover
        return f"{'PASS' if self.ok else 'FAIL'} {self.name}: {self.message}"


def _run_all() -> List[Check]:
    checks: List[Check] = []

    def check(name: str):
        def deco(fn: Callable[[], str]):
            c = Check(name)
            try:
                c.message = fn() or "OK"
                c.ok = True
            except Exception as exc:  # noqa: BLE001
                c.message = f"{type(exc).__name__}: {exc}"
                if _VERBOSE:
                    traceback.print_exc()
            checks.append(c)
            return fn

        return deco

    # ---------------- 配置 ----------------
    @check("配置解析：示例 task.yaml 可加载且校验通过")
    def _cfg():
        cfg = load_config(EXAMPLE)
        assert cfg.task.name, "任务名为空"
        assert cfg.task.classes, "类别列表为空"
        assert cfg.workspace.is_absolute(), "工作区路径未解析为绝对路径"
        return f"{cfg.task.name} / 类别 {len(cfg.task.classes)} / 工作区 {cfg.workspace.name}"

    @check("配置校验：非法 split 比例会被拒绝")
    def _cfg_bad():
        import yaml as _yaml

        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "t.yaml"
            p.write_text(
                _yaml.safe_dump(
                    {
                        "task": {"name": "x", "dataset_type": "y"},
                        "data": {
                            "source": "local",
                            "local_dir": ".",
                            "split": {"train": 0.5, "val": 0.2, "test": 0.1},
                        },
                    },
                    allow_unicode=True,
                ),
                encoding="utf-8",
            )
            try:
                load_config(p)
            except Exception as exc:
                assert "split" in str(exc)
                return "已按预期拒绝"
            raise AssertionError("非法 split 竟然通过了校验")

    # ---------------- 标注转换 ----------------
    @check("标注转换：VOC XML -> YOLO 归一化坐标")
    def _voc():
        xml = """<?xml version="1.0"?>
<annotation><size><width>200</width><height>100</height></size>
<object><name>crazing</name>
<bndbox><xmin>10</xmin><ymin>20</ymin><xmax>60</xmax><ymax>70</ymax></bndbox>
</object></annotation>"""
        with tempfile.TemporaryDirectory() as tmp:
            f = Path(tmp) / "a.xml"
            f.write_text(xml, encoding="utf-8")
            objects = convert.parse_voc(f)
            assert len(objects) == 1, f"解析出 {len(objects)} 个目标"
            text, warns = convert.voc_to_yolo_label(objects, {"crazing": 0})
            assert not warns, f"出现告警: {warns}"
            p = text.split()
            assert abs(float(p[1]) - 0.175) < 1e-6, f"cx 错误 {p[1]}"
            assert abs(float(p[3]) - 0.25) < 1e-6, f"w 错误 {p[3]}"
            return "cx/cy/w/h 与手算一致"

    @check("标注转换：COCO JSON -> YOLO")
    def _coco():
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            (tmp / "instances.json").write_text(
                json.dumps(
                    {
                        "images": [{"id": 1, "file_name": "x.jpg", "width": 100, "height": 100}],
                        "annotations": [
                            {"id": 1, "image_id": 1, "category_id": 1, "bbox": [10, 20, 30, 40]}
                        ],
                        "categories": [{"id": 1, "name": "scratch"}],
                    }
                ),
                encoding="utf-8",
            )
            written, warns = convert.coco_to_yolo(tmp / "instances.json", tmp / "labels", {"scratch": 3})
            assert written == 1 and not warns, f"written={written} warns={warns}"
            parts = (tmp / "labels" / "x.txt").read_text(encoding="utf-8").split()
            assert parts[0] == "3" and abs(float(parts[1]) - 0.25) < 1e-6
            return "类别索引与中心点均正确"

    @check("坐标校验：越界框与非法类别会被抓出来")
    def _validate():
        ok, _ = convert.validate_yolo_label("0 0.5 0.5 0.2 0.2", 6)
        assert ok
        ok, p1 = convert.validate_yolo_label("9 0.5 0.5 0.2 0.2", 6)
        assert not ok and any("越界" in x for x in p1)
        ok, p2 = convert.validate_yolo_label("0 0.99 0.5 0.4 0.2", 6)
        assert not ok and any("超出图像边界" in x for x in p2)
        return "类别越界 / 框越界 均能识别"

    @check("数据格式探测：labels/ 目录结构能被识别为 YOLO")
    def _detect():
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "images" / "train").mkdir(parents=True)
            (root / "labels" / "train").mkdir(parents=True)
            (root / "labels" / "train" / "a.txt").write_text("0 0.5 0.5 0.2 0.2\n", encoding="utf-8")
            fmt, _ = convert.detect_annotation_format(root)
            assert fmt == "yolo_txt", f"识别为 {fmt}"
            return "yolo_txt"

    # ---------------- 模型生成 ----------------
    @check("模型生成：插入注意力模块后 head 索引正确重映射")
    def _model_yaml():
        base = PROJECT_ROOT / "ultralytics" / "cfg" / "models" / "v8" / "yolov8.yaml"
        if not base.exists():
            return "跳过（本地没有 ultralytics 仓库）"
        import yaml as _yaml

        result = codegen.insert_modules_into_yaml(
            base.read_text(encoding="utf-8"), ["se", "ema", "cbam"], insert_after_c2f=3
        )
        data = _yaml.safe_load(result["text"])
        backbone, head = data["backbone"], data["head"]
        base_len = len(_yaml.safe_load(base.read_text(encoding="utf-8"))["backbone"])
        n_att = sum(1 for r in backbone if r[2] in {"SEBlock", "EMA", "CBAM"})
        assert n_att == 9, f"注意力层数 {n_att}"
        assert len(backbone) == base_len + 9
        total = len(backbone) + len(head)
        for row in head:
            frm = row[0]
            refs = frm if isinstance(frm, list) else [frm]
            for ref in refs:
                if isinstance(ref, int) and ref >= 0:
                    assert ref < total, f"引用 {ref} 越界"
        detect = head[-1]
        assert detect[2] == "Detect"
        return f"backbone {base_len}->{len(backbone)}，Detect 输入 {detect[0]}"

    @check("代码生成：训练/推理/App 脚本语法合法")
    def _scripts():
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
                title="t",
                upload_dir="u",
                output_dir="o",
            )
            compile(app.read_text(encoding="utf-8"), str(app), "exec")
            return "三个脚本均可编译"

    # ---------------- 模板模式研究质量 ----------------
    @check("模板模式：论文要点抽取可用")
    def _key_points():
        from .stages.s1_research import _extract_key_points

        abstract = (
            "Surface defect detection is critical for industrial quality control. "
            "We propose a lightweight attention module that improves YOLO accuracy on steel defects. "
            "Experiments show gains in mAP on the NEU-DET benchmark."
        )
        points = _extract_key_points(abstract, ["steel defect", "attention"])
        assert points and len(points) > 20, "要点为空"
        assert "attention" in points.lower() or "defect" in points.lower()
        return f"抽取 {len(points)} 字"

    @check("模板模式：本地可用模块扫描")
    def _local_modules():
        from .stages.s1_research import scan_local_attention_modules

        mods = scan_local_attention_modules(PROJECT_ROOT)
        if not (PROJECT_ROOT / "ultralytics" / "nn" / "modules").exists():
            return "跳过（本地没有 ultralytics 仓库）"
        names = [m["name"] for m in mods]
        assert names, "没扫到任何模块"
        return f"扫到 {len(names)} 个：{', '.join(names[:8])}"

    @check("模板模式：数据集事实抽取（规模/类别/格式）")
    def _facts():
        from .stages.s1_research import _extract_dataset_facts

        facts = _extract_dataset_facts(
            "The dataset contains 1,800 images of steel surface with 6 defect types in YOLO format, resolution 200x200"
        )
        assert facts.get("图片数") == "1,800", facts
        assert facts.get("类别数") == "6", facts
        assert facts.get("格式") == "YOLO", facts
        return str(facts)

    # ---------------- 工作区与状态 ----------------
    @check("工作区：目录结构创建完整")
    def _workspace():
        with tempfile.TemporaryDirectory() as tmp:
            ws = Workspace(Path(tmp) / "ws")
            ws.ensure()
            for d in ws.all_dirs():
                assert d.exists(), f"缺少 {d}"
            return f"{len(ws.all_dirs())} 个目录"

    @check("状态机：阶段状态可持久化")
    def _state():
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "state.json"
            st = PipelineState(path)
            st.start("s1", "t")
            st.finish("s1", STATUS_DONE, artifacts=[Path("a.md")], note="ok", data={"n": 3})
            st2 = PipelineState(path)
            assert st2.is_done("s1") and st2.get("s1", "n") == 3
            return "读写一致"

    # ---------------- 可视化接口（假模型） ----------------
    @check("可视化接口：假模型下 /health 与 /api/predict 正常")
    def _webapp():
        try:
            import flask  # noqa: F401
        except ImportError:
            return "跳过（未安装 flask）"
        from . import webapp

        class FakeBoxes:
            def __len__(self):
                return 0

        class FakeResult:
            boxes = FakeBoxes()
            names = {0: "crazing"}

            def plot(self):
                import numpy as np

                return np.zeros((40, 40, 3), dtype="uint8")

        class FakeModel:
            def predict(self, *a, **kw):
                return [FakeResult()]

        webapp._model_cache["model"] = FakeModel()
        webapp.MODEL_PATH = "fake.pt"
        with tempfile.TemporaryDirectory() as tmp:
            webapp.UPLOAD_DIR = Path(tmp) / "up"
            webapp.OUTPUT_DIR = Path(tmp) / "out"
            app = webapp.create_app()
            assert app is not None
            client = app.test_client()
            resp = client.get("/health")
            assert resp.status_code == 200, resp.status_code
            data = resp.get_json()
            assert data["model_loaded"] is True, data

            # 上传一张 1x1 的 png
            import base64
            import io as _io

            png = base64.b64decode(
                "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8BQDwAEhQGAhKmMIQAAAABJRU5ErkJggg=="
            )
            resp = client.post(
                "/api/predict",
                data={"image": (_io.BytesIO(png), "a.png")},
                content_type="multipart/form-data",
            )
            assert resp.status_code == 200, resp.status_code
            assert "boxes" in resp.get_json()
            return "/health 与 /api/predict 均正常"

    @check("可视化接口：批量检测与对比图")
    def _webapp_batch():
        try:
            import flask  # noqa: F401
        except ImportError:
            return "跳过（未安装 flask）"
        from . import webapp

        class FakeBoxes:
            def __len__(self):
                return 0

        class FakeResult:
            boxes = FakeBoxes()
            names = {0: "crazing"}

            def plot(self):
                import numpy as np

                return np.zeros((40, 40, 3), dtype="uint8")

        class FakeModel:
            def predict(self, *a, **kw):
                return [FakeResult()]

        webapp._model_cache["model"] = FakeModel()
        webapp.MODEL_PATH = "fake.pt"
        with tempfile.TemporaryDirectory() as tmp:
            webapp.UPLOAD_DIR = Path(tmp) / "up"
            webapp.OUTPUT_DIR = Path(tmp) / "out"
            app = webapp.create_app()
            client = app.test_client()
            import base64
            import io as _io

            png = base64.b64decode(
                "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8BQDwAEhQGAhKmMIQAAAABJRU5ErkJggg=="
            )
            resp = client.post(
                "/api/predict_batch",
                data={
                    "images": [
                        (_io.BytesIO(png), "a.png"),
                        (_io.BytesIO(png), "b.png"),
                    ]
                },
                content_type="multipart/form-data",
            )
            assert resp.status_code == 200, resp.status_code
            data = resp.get_json()
            assert data["count"] == 2, data
            assert len(data["results"]) == 2, data
            assert data["results"][0]["image_url"], "缺少标注图链接"
            return f"批量 {data['count']} 张，标注图链接与对比图 {'已生成' if data.get('comparison_url') else '未生成（无 cv2 时属正常）'}"

    return checks


_VERBOSE = False


def run_selftest(verbose: bool = False) -> Tuple[bool, List[Check]]:
    """执行自检，返回 (是否全部通过, 检查列表)。"""
    global _VERBOSE
    _VERBOSE = verbose
    checks = _run_all()
    return all(c.ok for c in checks), checks
