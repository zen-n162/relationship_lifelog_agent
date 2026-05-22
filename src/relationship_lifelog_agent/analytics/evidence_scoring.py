from __future__ import annotations

from dataclasses import dataclass, field

from relationship_lifelog_agent.adapters.types import EvidenceItem


@dataclass(frozen=True)
class AnalysisResult:
    intent: str
    summary: str
    aggregate: dict[str, object] = field(default_factory=dict)
    examples: list[str] = field(default_factory=list)
    evidence: list[EvidenceItem] = field(default_factory=list)
    confidence: float = 0.5
    evidence_strength: float = 0.5
    cautions: list[str] = field(default_factory=list)


def confidence_label(value: float) -> str:
    if value >= 0.9:
        return "very strong"
    if value >= 0.7:
        return "strong"
    if value >= 0.5:
        return "medium"
    if value >= 0.3:
        return "weak"
    return "very weak"


def mean(values: list[float]) -> float:
    if not values:
        return 0.0
    return sum(values) / len(values)
