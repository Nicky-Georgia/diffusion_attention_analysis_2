from __future__ import annotations

import json
import sys
import time
from dataclasses import dataclass, field
from typing import Any, Dict


@dataclass
class JsonProgress:
    stage_id: str = "unknown"
    stream: Any = sys.stdout
    start_time: float = field(default_factory=time.time)
    total: int | None = None
    current: int = 0

    def emit(self, event: str, **payload: Any) -> None:
        msg: Dict[str, Any] = {
            "event": event,
            "stage_id": self.stage_id,
            "time_sec": round(time.time() - self.start_time, 3),
        }
        if self.total is not None:
            msg["total"] = self.total
            msg["current"] = self.current
        msg.update(payload)
        print(json.dumps(msg, ensure_ascii=False), file=self.stream, flush=True)

    def start(self, total: int | None = None, message: str = "started", **payload: Any) -> None:
        self.total = total
        self.current = 0
        self.emit("start", message=message, **payload)

    def update(self, current: int | None = None, increment: int = 1, message: str = "progress", **payload: Any) -> None:
        if current is None:
            self.current += increment
        else:
            self.current = current
        self.emit("progress", message=message, **payload)

    def skip(self, message: str = "skipped", **payload: Any) -> None:
        self.emit("skip", message=message, **payload)

    def end(self, message: str = "finished", **payload: Any) -> None:
        if self.total is not None:
            self.current = self.total
        self.emit("end", message=message, **payload)
