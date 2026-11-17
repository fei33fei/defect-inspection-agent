"""任务配置：把用户的一句话需求（数据集类型 + 模型 + 工作目录）落成可执行的 task.yaml。

配置分层（各层都可以被用户在 task.yaml 里覆盖）：
    task     任务元信息：名字、数据集类型/名称/类别、关键词
    model    模型选择：基线、消融变体、权重来源、官方仓库来源
    data     数据来源：本地 / URL / HuggingFace / Kaggle，标注格式、切分比例
    train    训练超参与消融矩阵
    llm      大模型接入方式
    search   联网检索开关
    app      可视化系统配置
    report   项目文档配置

解析策略：
    - 配置文件缺失时给出最小可用配置（default_config）；
    - 未知键不报错但会记 warning，避免用户写错字段名后静默失效；
    - 所有相对路径都相对「task.yaml 所在目录」解析，Agent 在任何 cwd 下行为一致。
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

try:
    import yaml
except ImportError as exc:  # pragma: no cover - 环境缺失时给出明确指引
    raise SystemExit(
        "缺少 pyyaml，请在训练/运行环境中执行: pip install pyyaml\n"
        f"原始错误: {exc}"
    ) from exc


class ConfigError(ValueError):
    """任务配置非法。"""


# ---------------------------------------------------------------------------
# 各配置分片
# ---------------------------------------------------------------------------


@dataclass
class TaskInfo:
    name: str = "vision-task"
    dataset_type: str = ""
    dataset_name: str = ""
    dataset_keywords: List[str] = field(default_factory=list)
    classes: List[str] = field(default_factory=list)
    description: str = ""
    notes: str = ""


@dataclass
class ModelConfig:
    base: str = "yolov8n"
    variants: List[str] = field(default_factory=list)
    auto_variants: List[str] = field(default_factory=list)
    pretrained: str = "yolov8n.pt"
    repo: str = ""
    source: str = "auto"  # auto | local | github
    modules: List[str] = field(default_factory=list)
    insert_after_c2f: int = 3  # 在 backbone 前 N 个 C2f 之后插入注意力模块


@dataclass
class DataConfig:
    source: str = "local"  # local | url | huggingface | kaggle
    local_dir: str = ""
    raw_dir: str = "datasets/raw"
    urls: List[str] = field(default_factory=list)
    hf_repo: str = ""
    kaggle_dataset: str = ""
    annotation_format: str = "auto"  # auto | voc_xml | coco_json | yolo_txt
    image_exts: List[str] = field(
        default_factory=lambda: [".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"]
    )
    split: Dict[str, float] = field(default_factory=lambda: {"train": 0.8, "val": 0.1, "test": 0.1})
    seed: int = 0
    copy_mode: str = "link"  # link | copy | symlink（link=优先硬链接，失败回退复制）


@dataclass
class TrainConfig:
    epochs: int = 300
    batch: int = 16
    imgsz: int = 640
    device: str = "0"
    workers: int = 8
    patience: int = 50
    optimizer: str = "auto"
    lr0: float = 0.01
    seed: int = 0
    project: str = "runs"
    auto_train: bool = True
    ablations: List[str] = field(default_factory=list)
    timeout_minutes: int = 0  # 0 = 不限制


@dataclass
class LLMConfig:
    provider: str = "auto"  # auto | anthropic | chat-completions | none
    base_url: str = ""
    api_key: str = ""
    api_key_env: str = "VISION_AGENT_LLM_API_KEY"
    model: str = ""
    temperature: float = 0.3
    max_tokens: int = 4096
    timeout: float = 180.0
    fallback: str = "template"  # template | strict
    config_path: str = ""  # 默认读取 ~/.minimax/config.yaml


@dataclass
class SearchConfig:
    enabled: bool = True
    max_results: int = 8
    timeout: int = 30
    offline: bool = False
    sources: List[str] = field(
        default_factory=lambda: ["huggingface", "zenodo", "arxiv", "semanticscholar", "github"]
    )


@dataclass
class AppConfig:
    host: str = "127.0.0.1"
    port: int = 8000
    conf: float = 0.25
    iou: float = 0.45
    title: str = "视觉检测系统"
    camera_default: int = 0
    max_upload_mb: int = 20


@dataclass
class ReportConfig:
    title: str = "目标检测项目报告"
    author: str = ""
    background: str = ""
    output: str = "reports/project_report.md"


@dataclass
class AgentConfig:
    task: TaskInfo = field(default_factory=TaskInfo)
    model: ModelConfig = field(default_factory=ModelConfig)
    data: DataConfig = field(default_factory=DataConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    llm: LLMConfig = field(default_factory=LLMConfig)
    search: SearchConfig = field(default_factory=SearchConfig)
    app: AppConfig = field(default_factory=AppConfig)
    report: ReportConfig = field(default_factory=ReportConfig)

    # 运行期字段，不从 yaml 读取
    config_path: Path = field(default_factory=lambda: Path("task.yaml"))
    base_dir: Path = field(default_factory=lambda: Path("."))
    workspace: Path = field(default_factory=lambda: Path("./workspace"))
    project_root: Path = field(default_factory=lambda: Path("."))
    _unknown_keys: List[str] = field(default_factory=list, repr=False)

    # -- 路径 ---------------------------------------------------------------
    def resolve(self, value: str) -> Path:
        """把配置里的相对路径解析成绝对路径（相对 task.yaml 所在目录）。"""
        p = Path(value).expanduser()
        return p if p.is_absolute() else (self.base_dir / p).resolve()

    def validate(self) -> None:
        errors: List[str] = []
        if not self.task.name:
            errors.append("task.name 不能为空")
        if not self.task.dataset_type and not self.task.dataset_keywords:
            errors.append("需要至少提供 task.dataset_type 或 task.dataset_keywords")
        if self.data.source not in {"local", "url", "huggingface", "kaggle"}:
            errors.append(f"data.source 非法: {self.data.source}")
        if self.data.source == "url" and not self.data.urls:
            errors.append("data.source=url 时必须提供 data.urls")
        if self.data.source == "huggingface" and not self.data.hf_repo:
            errors.append("data.source=huggingface 时必须提供 data.hf_repo")
        if self.data.source == "kaggle" and not self.data.kaggle_dataset:
            errors.append("data.source=kaggle 时必须提供 data.kaggle_dataset")
        if self.data.source == "local" and not self.data.local_dir:
            errors.append("data.source=local 时必须提供 data.local_dir")
        if self.data.annotation_format not in {"auto", "voc_xml", "coco_json", "yolo_txt"}:
            errors.append(f"data.annotation_format 非法: {self.data.annotation_format}")
        total = sum(self.data.split.values())
        if total <= 0:
            errors.append("data.split 比例之和必须大于 0")
        elif abs(total - 1.0) > 0.02:
            errors.append(f"data.split 比例之和应为 1.0，当前为 {total:.3f}")
        if errors:
            raise ConfigError("任务配置校验失败:\n  - " + "\n  - ".join(errors))


# ---------------------------------------------------------------------------
# 解析
# ---------------------------------------------------------------------------

_SECTIONS = {
    "task": TaskInfo,
    "model": ModelConfig,
    "data": DataConfig,
    "train": TrainConfig,
    "llm": LLMConfig,
    "search": SearchConfig,
    "app": AppConfig,
    "report": ReportConfig,
}


def _build(cls, data: Dict[str, Any], unknown: List[str]):
    """把 dict 递归映射到 dataclass，未知键记入 unknown。"""
    if not isinstance(data, dict):
        raise ConfigError(f"{cls.__name__} 段必须是字典，实际是 {type(data).__name__}")
    kwargs: Dict[str, Any] = {}
    fields = {f.name: f for f in dataclasses.fields(cls)}
    for key, value in data.items():
        if key not in fields:
            unknown.append(f"{cls.__name__}.{key}")
            continue
        ftype = fields[key].type
        # split 用的是 Dict[str, float]，其余字段按声明类型做基础转换
        if key == "split" and isinstance(value, dict):
            kwargs[key] = {str(k): float(v) for k, v in value.items()}
        elif isinstance(value, list) and ftype is not str:
            kwargs[key] = list(value)
        else:
            kwargs[key] = value
    return cls(**kwargs)


def load_config(path: str | Path) -> AgentConfig:
    """读取 task.yaml，构造并校验 AgentConfig。"""
    config_path = Path(path).expanduser().resolve()
    if not config_path.exists():
        raise ConfigError(f"任务配置文件不存在: {config_path}")
    raw = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    if not isinstance(raw, dict):
        raise ConfigError(f"任务配置文件格式错误（应为字典）: {config_path}")

    unknown: List[str] = []
    kwargs: Dict[str, Any] = {}
    for section, cls in _SECTIONS.items():
        kwargs[section] = _build(cls, raw.get(section, {}) or {}, unknown)

    cfg = AgentConfig(
        config_path=config_path,
        base_dir=config_path.parent,
        _unknown_keys=unknown,
        **kwargs,
    )

    # 顶层可选字段
    workspace = raw.get("workspace")
    project_root = raw.get("project_root")
    cfg.workspace = cfg.resolve(workspace) if workspace else (cfg.base_dir / "workspace").resolve()
    cfg.project_root = cfg.resolve(project_root) if project_root else cfg.base_dir

    # 若提供了 pretrained 且 base 未指定，用权重名推导 base
    if not raw.get("model", {}).get("base") and raw.get("model", {}).get("pretrained"):
        cfg.model.base = Path(cfg.model.pretrained).stem

    cfg.validate()
    return cfg


def default_config_dict() -> Dict[str, Any]:
    """最小可用配置（钢材缺陷检测场景）。"""
    return {
        "task": {
            "name": "steel-defect-detection",
            "dataset_type": "钢材表面缺陷检测",
            "dataset_name": "NEU-DET",
            "dataset_keywords": [
                "steel surface defect detection dataset",
                "NEU-DET steel surface defect database",
            ],
            "classes": [
                "crazing",
                "inclusion",
                "patches",
                "pitted_surface",
                "rolled-in-scale",
                "scratches",
            ],
            "description": "NEU-DET 钢材表面缺陷数据集上的 YOLOv8 注意力模块消融研究",
        },
        "model": {
            "base": "yolov8n",
            "pretrained": "yolov8n.pt",
            "variants": ["yolov8-se", "yolov8-ema", "yolov8-cbam"],
            "auto_variants": ["se+ema+cbam"],
            "source": "auto",
        },
        "data": {
            "source": "local",
            "local_dir": "../ultralytics-main/datasets/NEU-DET",
            "annotation_format": "auto",
            "split": {"train": 0.8, "val": 0.1, "test": 0.1},
        },
        "train": {
            "epochs": 300,
            "batch": 16,
            "imgsz": 640,
            "device": "0",
            "patience": 50,
            "auto_train": True,
        },
        "workspace": "./workspace/steel-defect",
        "project_root": "../ultralytics-main",
    }


def dump_config(cfg: AgentConfig, path: str | Path) -> Path:
    """把配置写回 yaml（用于 init 子命令生成任务模板）。"""
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    data: Dict[str, Any] = {
        "task": dataclasses.asdict(cfg.task),
        "model": dataclasses.asdict(cfg.model),
        "data": dataclasses.asdict(cfg.data),
        "train": dataclasses.asdict(cfg.train),
        "llm": dataclasses.asdict(cfg.llm),
        "search": dataclasses.asdict(cfg.search),
        "app": dataclasses.asdict(cfg.app),
        "report": dataclasses.asdict(cfg.report),
        "workspace": str(cfg.workspace),
        "project_root": str(cfg.project_root),
    }
    out.write_text(
        yaml.safe_dump(data, allow_unicode=True, sort_keys=False, default_flow_style=False),
        encoding="utf-8",
    )
    return out


def write_template_config(path: str | Path) -> Path:
    """生成一份带注释的任务配置模板。"""
    cfg_data = default_config_dict()
    header = (
        "# ==========================================================================\n"
        "# 视觉检测 Agent · 任务配置\n"
        "# 用法: python -m vision_agent run --task <本文件>\n"
        "#\n"
        "# 关键字段说明:\n"
        "#   task.dataset_type      必填, 数据集类型描述 (用于检索数据集与论文)\n"
        "#   task.dataset_keywords  检索关键词, 越具体命中率越高\n"
        "#   task.classes           类别名列表, 决定标签索引顺序\n"
        "#   model.variants         消融变体 (模型 YAML 名, 不含扩展名)\n"
        "#   model.auto_variants    自动生成的组合变体, 用 '+' 连接模块名, 如 se+ema+cbam\n"
        "#   data.local_dir         数据集原始目录 (VOC/COCO/YOLO 均可)\n"
        "#   data.annotation_format auto 会自动探测\n"
        "#   train.ablations        要跑的实验列表, 留空则用 model.variants + base\n"
        "#   workspace              Agent 工作目录, 所有产物写在这里\n"
        "#   project_root           本地 ultralytics 仓库根目录 (含 SE/EMA/CBAM 自定义模块时指向它)\n"
        "# ==========================================================================\n\n"
    )
    body = yaml.safe_dump(cfg_data, allow_unicode=True, sort_keys=False, default_flow_style=False)
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(header + body, encoding="utf-8")
    return out
