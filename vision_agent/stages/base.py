"""阶段基类与公共上下文。"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from ..config import AgentConfig
from ..llm import LLMClient
from ..logging_utils import get_logger
from ..state import PipelineState
from ..workspace import Workspace


@dataclass
class StageResult:
    ok: bool = True
    artifacts: List[Path] = field(default_factory=list)
    data: Dict[str, Any] = field(default_factory=dict)
    message: str = ""
    skipped: bool = False

    def add(self, path: Path) -> "StageResult":
        self.artifacts.append(Path(path))
        return self


@dataclass
class StageContext:
    cfg: AgentConfig
    ws: Workspace
    state: PipelineState
    llm: LLMClient
    options: Dict[str, Any] = field(default_factory=dict)
    logger: Any = field(default_factory=lambda: get_logger("stage"))

    @property
    def dry_run(self) -> bool:
        return bool(self.options.get("dry_run", False))

    def opt(self, key: str, default: Any = None) -> Any:
        return self.options.get(key, default)
