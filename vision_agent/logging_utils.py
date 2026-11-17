"""统一日志与控制台输出。

Agent 在两种场景下运行：
    1. 用户在自己终端里手动执行 —— 需要清晰的阶段分隔与进度提示；
    2. 被上层程序调用 —— 需要结构化日志文件，便于事后排查。

这里统一用 stdlib logging，不引入第三方依赖，
同时提供 `stage_banner` 之类的辅助函数，让终端输出有清晰的阶段感。
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

_CONFIGURED = False

LOGGER_NAME = "vision_agent"


def setup_logging(log_dir: Path | None = None, verbose: bool = False) -> logging.Logger:
    """初始化根 logger。重复调用不会重复挂 handler。"""
    global _CONFIGURED
    logger = logging.getLogger(LOGGER_NAME)
    if _CONFIGURED:
        return logger

    logger.setLevel(logging.DEBUG if verbose else logging.INFO)
    logger.propagate = False

    fmt = logging.Formatter("%(asctime)s | %(levelname)-7s | %(message)s", "%H:%M:%S")
    console = logging.StreamHandler(sys.stdout)
    console.setLevel(logging.DEBUG if verbose else logging.INFO)
    console.setFormatter(fmt)
    logger.addHandler(console)

    if log_dir is not None:
        log_dir.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_dir / "agent.log", encoding="utf-8")
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(
            logging.Formatter("%(asctime)s | %(levelname)-7s | %(name)s | %(message)s")
        )
        logger.addHandler(file_handler)

    _CONFIGURED = True
    return logger


def get_logger(name: str | None = None) -> logging.Logger:
    """获取子 logger。"""
    base = logging.getLogger(LOGGER_NAME)
    return base.getChild(name) if name else base


def stage_banner(title: str, subtitle: str = "") -> None:
    """打印阶段横幅，让长流程在终端里可读。"""
    logger = get_logger("banner")
    line = "=" * 78
    logger.info(line)
    logger.info(f"  {title}" + (f"  ——  {subtitle}" if subtitle else ""))
    logger.info(line)


def bullet(items, indent: str = "  ") -> None:
    """以项目符号形式输出条目（空条目自动跳过）。"""
    logger = get_logger("bullet")
    for item in items:
        logger.info(f"{indent}- {item}")
