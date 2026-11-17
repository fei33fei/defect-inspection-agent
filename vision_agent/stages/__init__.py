"""S0-S8 阶段实现。每个阶段对应工作流中的一步，产物写入工作区约定目录。"""

from . import (  # noqa: F401
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

__all__ = [
    "s0_setup",
    "s1_research",
    "s2_models",
    "s3_data",
    "s4_preprocess",
    "s5_train",
    "s6_visualize",
    "s7_infer",
    "s8_report",
]
