from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, ClassVar


@dataclass(frozen=True)
class EvidenceItem:
    source_type: str
    source_id: str
    date: str
    role: str
    summary: str
    excerpt: str | None = None
    confidence: float = 0.5
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class AdapterEvidence:
    source_id: str
    date: str
    summary: str
    role: str = "supporting"
    excerpt: str | None = None
    confidence: float = 0.5
    metadata: dict[str, Any] = field(default_factory=dict)

    source_type: ClassVar[str] = "manual"

    def as_evidence_item(self) -> EvidenceItem:
        return EvidenceItem(
            source_type=self.source_type,
            source_id=self.source_id,
            date=self.date,
            role=self.role,
            summary=self.summary,
            excerpt=self.excerpt,
            confidence=self.confidence,
            metadata=dict(self.metadata),
        )


@dataclass(frozen=True)
class LineEvidence(AdapterEvidence):
    source_type: ClassVar[str] = "line_message"
    speaker_id: str | None = None
    conversation_id: str | None = None


@dataclass(frozen=True)
class MediaEvidence(AdapterEvidence):
    source_type: ClassVar[str] = "media_caption"
    media_id: str | None = None
    media_type: str = "photo"
    place_id: str | None = None
    person_ids: tuple[str, ...] = ()


@dataclass(frozen=True)
class EventEvidence(AdapterEvidence):
    source_type: ClassVar[str] = "note_event"
    event_type: str = "other"
    review_status: str = "candidate"
    evidence_strength: float = 0.5
    severity: int = 0


@dataclass(frozen=True)
class PlaceEvidence(AdapterEvidence):
    source_type: ClassVar[str] = "place"
    place_id: str | None = None
    place_label: str | None = None
    area_label: str | None = None


@dataclass(frozen=True)
class NoteEvidence(AdapterEvidence):
    source_type: ClassVar[str] = "note"
    note_id: str | None = None
    category: str = "relationship_reflection"


@dataclass(frozen=True)
class ThoughtEvidence(AdapterEvidence):
    source_type: ClassVar[str] = "thought"
    thought_id: str | None = None
    thought_type: str = "own_thought"


@dataclass(frozen=True)
class MonthlyReflection(AdapterEvidence):
    source_type: ClassVar[str] = "monthly_reflection"
    month: str = ""


@dataclass(frozen=True)
class EvidenceBundle:
    line: tuple[LineEvidence, ...] = ()
    media: tuple[MediaEvidence, ...] = ()
    events: tuple[EventEvidence, ...] = ()
    places: tuple[PlaceEvidence, ...] = ()
    notes: tuple[NoteEvidence, ...] = ()
    thoughts: tuple[ThoughtEvidence, ...] = ()
    monthly_reflections: tuple[MonthlyReflection, ...] = ()

    def to_evidence_items(self) -> list[EvidenceItem]:
        evidence: list[AdapterEvidence] = [
            *self.line,
            *self.media,
            *self.events,
            *self.places,
            *self.notes,
            *self.thoughts,
            *self.monthly_reflections,
        ]
        return [item.as_evidence_item() for item in evidence]


@dataclass(frozen=True)
class RelationshipEvent:
    event_id: str
    event_type: str
    date: str
    summary: str
    review_status: str = "candidate"
    confidence: float = 0.5
    evidence_strength: float = 0.5
    severity: int = 0
    evidence: tuple[EvidenceItem, ...] = ()
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class PostConflictActivity:
    conflict_event_id: str
    activity_event_id: str
    date: str
    days_after_conflict: int
    place_label: str
    activity_type: str
    confidence: float
    evidence_strength: float
    evidence: tuple[EvidenceItem, ...] = ()
