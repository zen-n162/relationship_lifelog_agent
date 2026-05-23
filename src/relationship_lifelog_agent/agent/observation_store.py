from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any
from uuid import uuid4


@dataclass(frozen=True)
class Observation:
    observation_id: str
    task_id: str
    observation_type: str
    summary: str
    data: dict[str, Any] = field(default_factory=dict)
    source_refs: tuple[str, ...] = ()
    confidence: float = 0.5

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ObservationStore:
    observations: tuple[Observation, ...] = ()

    def add(
        self,
        *,
        task_id: str,
        observation_type: str,
        summary: str,
        data: dict[str, Any] | None = None,
        source_refs: tuple[str, ...] = (),
        confidence: float = 0.5,
    ) -> "ObservationStore":
        observation = Observation(
            observation_id=f"obs-{uuid4().hex[:12]}",
            task_id=task_id,
            observation_type=observation_type,
            summary=summary,
            data=data or {},
            source_refs=source_refs,
            confidence=confidence,
        )
        return ObservationStore((*self.observations, observation))

    def by_task(self, task_id: str) -> tuple[Observation, ...]:
        return tuple(item for item in self.observations if item.task_id == task_id)

    def all(self) -> tuple[Observation, ...]:
        return self.observations

    def by_type(self, observation_type: str) -> tuple[Observation, ...]:
        return tuple(item for item in self.observations if item.observation_type == observation_type)

    def latest(self, observation_type: str) -> Observation | None:
        matches = self.by_type(observation_type)
        return matches[-1] if matches else None

    def to_debug_list(self) -> list[dict[str, Any]]:
        return [item.to_dict() for item in self.observations]
