"""流水线状态与产物清单。

每个阶段执行完把自己的状态写进 state.json：
    {"stage": {"status": "done", "started_at": ..., "ended_at": ..., "artifacts": [...]}}

这样 `run --resume` 能跳过已完成的阶段，`status` 能一眼看到进度，
阶段之间也能通过 `state.get("s4")` 传递上一步的关键结果。
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

STATUS_PENDING = "pending"
STATUS_RUNNING = "running"
STATUS_DONE = "done"
STATUS_FAILED = "failed"
STATUS_SKIPPED = "skipped"


def _now() -> str:
    return datetime.now().isoformat(timespec="seconds")


@dataclass
class StageRecord:
    name: str
    status: str = STATUS_PENDING
    started_at: Optional[str] = None
    ended_at: Optional[str] = None
    artifacts: List[str] = field(default_factory=list)
    note: str = ""
    data: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return dataclasses_asdict(self)


def dataclasses_asdict(obj: StageRecord) -> Dict[str, Any]:
    return {
        "name": obj.name,
        "status": obj.status,
        "started_at": obj.started_at,
        "ended_at": obj.ended_at,
        "artifacts": list(obj.artifacts),
        "note": obj.note,
        "data": obj.data,
    }


class PipelineState:
    """state.json 的读写封装。"""

    def __init__(self, path: Path):
        self.path = Path(path)
        self.stages: Dict[str, StageRecord] = {}
        self.meta: Dict[str, Any] = {}
        self.load()

    # --- 读写 ---
    def load(self) -> "PipelineState":
        if not self.path.exists():
            return self
        try:
            raw = json.loads(self.path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            return self
        self.meta = raw.get("meta", {}) or {}
        for name, rec in (raw.get("stages", {}) or {}).items():
            self.stages[name] = StageRecord(
                name=rec.get("name", name),
                status=rec.get("status", STATUS_PENDING),
                started_at=rec.get("started_at"),
                ended_at=rec.get("ended_at"),
                artifacts=list(rec.get("artifacts", []) or []),
                note=rec.get("note", ""),
                data=rec.get("data", {}) or {},
            )
        return self

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "meta": self.meta,
            "updated_at": _now(),
            "stages": {k: v.to_dict() for k, v in self.stages.items()},
        }
        self.path.write_text(
            json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
        )

    # --- 阶段操作 ---
    def start(self, name: str, note: str = "") -> StageRecord:
        rec = self.stages.setdefault(name, StageRecord(name=name))
        rec.status = STATUS_RUNNING
        rec.started_at = _now()
        rec.ended_at = None
        rec.note = note
        self.save()
        return rec

    def finish(
        self,
        name: str,
        status: str = STATUS_DONE,
        artifacts: Optional[List[Path]] = None,
        note: str = "",
        data: Optional[Dict[str, Any]] = None,
    ) -> StageRecord:
        rec = self.stages.setdefault(name, StageRecord(name=name))
        rec.status = status
        rec.ended_at = _now()
        if artifacts:
            rec.artifacts = [str(a) for a in artifacts]
        if note:
            rec.note = note
        if data:
            rec.data.update(data)
        self.save()
        return rec

    def status_of(self, name: str) -> str:
        rec = self.stages.get(name)
        return rec.status if rec else STATUS_PENDING

    def is_done(self, name: str) -> bool:
        return self.status_of(name) in {STATUS_DONE, STATUS_SKIPPED}

    def get(self, name: str, key: str, default: Any = None) -> Any:
        rec = self.stages.get(name)
        if not rec:
            return default
        return rec.data.get(key, default)

    def summary(self) -> List[Dict[str, Any]]:
        rows = []
        for name, rec in self.stages.items():
            rows.append(
                {
                    "stage": name,
                    "status": rec.status,
                    "started_at": rec.started_at or "-",
                    "ended_at": rec.ended_at or "-",
                    "artifacts": len(rec.artifacts),
                    "note": rec.note,
                }
            )
        return rows
