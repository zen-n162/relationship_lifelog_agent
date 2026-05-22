from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ReviewAction:
    event_id: int
    action: str
    note: str | None = None
