from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal


StreamKind = Literal[
    "progress",
    "thought",
    "tool_start",
    "tool_done",
    "llm_start",
    "llm_done",
    "partial_answer",
    "final_answer",
    "warning",
    "error",
]

StreamStatus = Literal["pending", "done", "error"]


@dataclass(frozen=True)
class AgentStreamEvent:
    kind: StreamKind
    title: str
    message: str
    status: StreamStatus
    safe_for_user: bool = True
    debug_only: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)

    def safe_metadata(self) -> dict[str, Any]:
        allowed = {"id", "parent_id", "duration", "status", "log"}
        clean = {key: value for key, value in self.metadata.items() if key in allowed}
        clean["title"] = self.title
        clean["status"] = self.status
        clean["kind"] = self.kind
        return clean
