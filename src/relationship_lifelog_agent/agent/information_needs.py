from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


NeedStatus = Literal["available", "missing", "ambiguous", "optional"]


@dataclass(frozen=True)
class InformationNeed:
    name: str
    required: bool
    status: NeedStatus
    reason: str
    how_to_get: str | None = None


def split_needs(needs: tuple[InformationNeed, ...]) -> tuple[tuple[InformationNeed, ...], tuple[InformationNeed, ...]]:
    available = tuple(item for item in needs if item.status in {"available", "optional"})
    missing = tuple(item for item in needs if item.status in {"missing", "ambiguous"})
    return available, missing
