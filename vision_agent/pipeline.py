"""流水线编排：按顺序执行 S0-S8，维护状态，失败可续跑。"""

from __future__ import annotations

import traceback
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional

from .config import AgentConfig
from .llm import LLMClient
from .logging_utils import get_logger, stage_banner
from .state import PipelineState, STATUS_DONE, STATUS_FAILED, STATUS_SKIPPED
from .stages import base as stage_base
from .stages import (
    s0_setup,
    s1_research,
    s2_models,
    s3_data,
    s4_preprocess,
    s5_train,
    s6_visualize,
    s7_infer,
    s8_report,
)
from .workspace import Workspace


@dataclass
class StageSpec:
    key: str
    title: str
    subtitle: str
    func: Callable[[stage_base.StageContext], stage_base.StageResult]
    # S7（推理测试）可以与 S5（训练）并行，但为了状态清晰默认串行执行
    parallel_with: Optional[str] = None


STAGES: List[StageSpec] = [
    StageSpec("s0", "任务受理与工作区初始化", "建立目录结构与配置快照", s0_setup.run),
    StageSpec("s1", "数据集与论文调研", "权威站点检索 + 大模型归纳", s1_research.run),
    StageSpec("s2", "官方模型获取", "本地复用 / GitHub 检索克隆 / 生成组合变体", s2_models.run),
    StageSpec("s3", "数据集下载", "本地收拢 / 直链 / HuggingFace / Kaggle", s3_data.run),
    StageSpec("s4", "数据预处理", "标注转换、切分、校验、生成 data.yaml", s4_preprocess.run),
    StageSpec("s5", "训练与消融", "生成训练脚本并自动执行消融实验", s5_train.run),
    StageSpec("s6", "可视化系统", "上传检测下载 + 摄像头视频流", s6_visualize.run),
    StageSpec("s7", "推理测试", "最优权重评估与样本级预测", s7_infer.run),
    StageSpec("s8", "项目文档", "汇总全部产物生成交付报告", s8_report.run),
]

STAGE_BY_KEY = {s.key: s for s in STAGES}


def resolve_stage_keys(requested: Optional[List[str]]) -> List[StageSpec]:
    """把用户指定的阶段列表（支持 's4,s5' 或 's4-s7' 或 'all'）解析成阶段列表。"""
    if not requested:
        return list(STAGES)
    keys: List[str] = []
    for token in requested:
        token = (token or "").strip().lower()
        if not token:
            continue
        if token in {"all", "*"}:
            return list(STAGES)
        if "-" in token and not token.startswith("s"):
            start, _, end = token.partition("-")
            try:
                lo, hi = int(start), int(end)
            except ValueError:
                continue
            keys.extend(f"s{i}" for i in range(lo, hi + 1))
            continue
        keys.append(token)
    out = []
    for spec in STAGES:
        if spec.key in keys or spec.key[1:] in keys:
            out.append(spec)
    return out


class Pipeline:
    """流水线执行器。"""

    def __init__(
        self,
        cfg: AgentConfig,
        llm: Optional[LLMClient] = None,
        options: Optional[Dict[str, Any]] = None,
    ):
        self.cfg = cfg
        self.ws = Workspace(cfg.workspace)
        self.state = PipelineState(self.ws.state_file)
        self.llm = llm or LLMClient(cfg.llm)
        self.options = options or {}
        self.logger = get_logger("pipeline")

    def prepare(self) -> None:
        self.ws.ensure()
        self.state.meta.update(
            {
                "task": self.cfg.task.name,
                "dataset": self.cfg.task.dataset_name or self.cfg.task.dataset_type,
                "base_model": self.cfg.model.base,
                "config": str(self.cfg.config_path),
            }
        )
        self.state.save()

    def run(
        self,
        stage_keys: Optional[List[str]] = None,
        resume: bool = False,
        stop_on_error: bool = True,
    ) -> Dict[str, Any]:
        """执行流水线，返回汇总结果。"""
        self.prepare()
        specs = resolve_stage_keys(stage_keys)
        if not specs:
            self.logger.warning("没有匹配到任何阶段")
            return {"ok": False, "stages": []}

        summary: Dict[str, Any] = {"ok": True, "stages": []}
        for spec in specs:
            if resume and self.state.is_done(spec.key):
                self.logger.info(f"[{spec.key}] 已完成，跳过（--resume）")
                summary["stages"].append({"key": spec.key, "status": "skipped"})
                continue

            stage_banner(f"{spec.key.upper()} · {spec.title}", spec.subtitle)
            self.state.start(spec.key, spec.title)
            ctx = stage_base.StageContext(
                cfg=self.cfg,
                ws=self.ws,
                state=self.state,
                llm=self.llm,
                options=dict(self.options),
                logger=get_logger(spec.key),
            )
            try:
                result = spec.func(ctx)
            except Exception as exc:  # noqa: BLE001
                self.logger.error(f"[{spec.key}] 执行异常: {exc}")
                self.logger.debug(traceback.format_exc())
                self.state.finish(spec.key, STATUS_FAILED, note=str(exc)[:500])
                summary["ok"] = False
                summary["stages"].append({"key": spec.key, "status": "failed", "error": str(exc)})
                if stop_on_error:
                    self.logger.error("流水线中止。修复后可用 --resume 从该阶段继续。")
                    break
                continue

            status = STATUS_DONE
            if result.skipped:
                status = STATUS_SKIPPED
            elif not result.ok:
                status = STATUS_FAILED
            self.state.finish(
                spec.key,
                status,
                artifacts=result.artifacts,
                note=result.message,
                data={k: v for k, v in result.data.items() if isinstance(v, (str, int, float, bool, list, dict))},
            )
            self.logger.info(f"[{spec.key}] {result.message}")
            summary["stages"].append({"key": spec.key, "status": status, "message": result.message})
            if not result.ok and stop_on_error:
                summary["ok"] = False
                self.logger.error("流水线中止。修复后可用 --resume 从该阶段继续。")
                break

        return summary

    def status_table(self) -> List[Dict[str, Any]]:
        return self.state.summary()
