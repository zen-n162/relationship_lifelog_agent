from __future__ import annotations

from collections import Counter
from datetime import date

from relationship_lifelog_agent.adapters.mock import MockRelationshipMemory
from relationship_lifelog_agent.adapters.types import AdapterEvidence, EventEvidence, EvidenceItem, LineEvidence, NoteEvidence, RelationshipEvent
from relationship_lifelog_agent.analytics.evidence_scoring import AnalysisResult, mean


CONFLICT_TYPES = {"conflict", "minor_misunderstanding"}
_STRONG_CONFLICT_TERMS = ("喧嘩", "けんか", "不満", "謝罪", "ごめん", "予定変更", "合わず", "再調整", "キャンセル")
_MINOR_TERMS = ("すれ違い", "軽い", "予定調整", "返信間隔")
_WEAK_ONLY_TERMS = ("返信遅延", "long silence", "delayed reply", "返信が少し空いた", "短文", "emoji")
_NOTE_SUPPORT_TERMS = ("反省", "不安", "後悔", "心配", "うまく伝えられ")


def conflict_frequency(
    memory: MockRelationshipMemory,
    date_from: str | None = None,
    date_to: str | None = None,
) -> AnalysisResult:
    events = [
        event
        for event in memory.search_events(date_from=date_from, date_to=date_to)
        if event.event_type in CONFLICT_TYPES
    ]
    counts = Counter(event.event_type for event in events)
    evidence = [item for event in events for item in event.evidence]
    examples = [
        f"{event.date}: {_severity_label(event.severity)} - {event.summary}"
        for event in events[:3]
    ]
    if not events:
        return AnalysisResult(
            intent="conflict_frequency",
            summary="mock データ上では喧嘩候補は見つかりませんでした。",
            aggregate={"conflict_candidates": 0},
            confidence=0.4,
            evidence_strength=0.0,
            cautions=["mock adapter の結果なので、実データ統合後に再確認が必要です。"],
        )
    return AnalysisResult(
        intent="conflict_frequency",
        summary=f"mock データ上では喧嘩候補が {len(events)} 件あります。確定ではなく、人間確認前の候補です。",
        aggregate={
            "conflict_candidates": counts["conflict"],
            "minor_misunderstanding_candidates": counts["minor_misunderstanding"],
            "total_candidates": len(events),
        },
        examples=examples,
        evidence=evidence,
        confidence=mean([event.confidence for event in events]),
        evidence_strength=mean([event.evidence_strength for event in events]),
        cautions=[
            "喧嘩は候補として扱います。相手の気持ちや関係状態は断定できません。",
            "返信遅延などの弱い信号だけでは喧嘩とは扱いません。",
        ],
    )


def conflict_timeline(
    memory: MockRelationshipMemory,
    date_from: str | None = None,
    date_to: str | None = None,
) -> AnalysisResult:
    events = [
        event
        for event in memory.search_events(date_from=date_from, date_to=date_to)
        if event.event_type in CONFLICT_TYPES
    ]
    examples = [
        f"{event.date}: {_severity_label(event.severity)} / confidence={event.confidence:.2f}"
        for event in events
    ]
    evidence = [item for event in events for item in event.evidence]
    return AnalysisResult(
        intent="conflict_timeline",
        summary="喧嘩候補と軽いすれ違い候補を時系列で整理しました。",
        aggregate={"total_candidates": len(events)},
        examples=examples,
        evidence=evidence,
        confidence=mean([event.confidence for event in events]),
        evidence_strength=mean([event.evidence_strength for event in events]),
        cautions=["時系列は mock データによる候補であり、実データの確定事実ではありません。"],
    )


def _severity_label(severity: int) -> str:
    if severity >= 3:
        return "強めの喧嘩候補"
    if severity == 2:
        return "中程度の喧嘩候補"
    if severity == 1:
        return "軽いすれ違い候補"
    return "関係イベント候補"


def build_conflict_candidates(
    *,
    line_evidence: list[LineEvidence],
    note_evidence: list[NoteEvidence],
    event_evidence: list[EventEvidence],
    date_from: str | None = None,
    date_to: str | None = None,
) -> list[RelationshipEvent]:
    """Build dry-run conflict candidates without writing to the relationship DB."""

    candidates: list[RelationshipEvent] = []
    seen_dates: set[tuple[str, str]] = set()
    for event in event_evidence:
        if event.event_type not in CONFLICT_TYPES:
            continue
        if not _in_range(event.date, date_from, date_to):
            continue
        key = (event.event_type, event.date)
        seen_dates.add(key)
        candidates.append(
            RelationshipEvent(
                event_id=f"dry-run-{event.event_type}-{event.source_id}",
                event_type=event.event_type,
                date=event.date,
                summary=_safe_conflict_summary(event.event_type, event.summary),
                review_status="candidate",
                confidence=min(event.confidence, 0.72),
                evidence_strength=min(max(event.evidence_strength, 0.35), 0.8),
                severity=event.severity,
                evidence=(event.as_evidence_item(),),
                metadata={"dry_run": True, "source_pointer": event.source_pointer},
            )
        )

    for line in line_evidence:
        if not _in_range(line.date, date_from, date_to):
            continue
        classification = classify_line_conflict_signal(line)
        if classification is None:
            continue
        event_type, severity, base_confidence, base_strength = classification
        key = (event_type, line.date)
        if key in seen_dates:
            continue
        related_notes = _related_notes(line.date, note_evidence)
        evidence_items = [line.as_evidence_item(), *[note.as_evidence_item() for note in related_notes[:2]]]
        support_boost = 0.08 if related_notes else 0.0
        candidates.append(
            RelationshipEvent(
                event_id=f"dry-run-{event_type}-{line.source_id}",
                event_type=event_type,
                date=line.date,
                summary=_safe_conflict_summary(event_type, line.summary),
                review_status="candidate",
                confidence=min(base_confidence + support_boost, 0.72),
                evidence_strength=min(base_strength + support_boost, 0.78),
                severity=severity,
                evidence=tuple(evidence_items),
                metadata={"dry_run": True, "source_pointer": line.source_pointer},
            )
        )
        seen_dates.add(key)
    return sorted(candidates, key=lambda item: item.date)


def classify_line_conflict_signal(line: LineEvidence) -> tuple[str, int, float, float] | None:
    text = _evidence_text(line)
    strong_count = sum(1 for term in _STRONG_CONFLICT_TERMS if term in text)
    minor_count = sum(1 for term in _MINOR_TERMS if term in text)
    weak_count = sum(1 for term in _WEAK_ONLY_TERMS if term in text)

    if strong_count >= 2:
        return "conflict", 2, max(0.52, min(line.confidence, 0.65)), max(0.5, min(line.evidence_strength, 0.7))
    if strong_count >= 1 and minor_count >= 1:
        return "minor_misunderstanding", 1, max(0.48, min(line.confidence, 0.6)), max(0.42, min(line.evidence_strength, 0.62))
    if minor_count >= 2 and weak_count == 0:
        return "minor_misunderstanding", 1, max(0.45, min(line.confidence, 0.56)), max(0.38, min(line.evidence_strength, 0.55))
    return None


def conflict_counts(candidates: list[RelationshipEvent]) -> dict[str, int]:
    counts = Counter(item.event_type for item in candidates)
    return {
        "conflict_candidates": counts["conflict"],
        "minor_misunderstanding_candidates": counts["minor_misunderstanding"],
        "total_candidates": len(candidates),
    }


def _related_notes(target_date: str, notes: list[NoteEvidence]) -> list[NoteEvidence]:
    try:
        current = date.fromisoformat(target_date)
    except ValueError:
        return []
    related = []
    for note in notes:
        if not _contains_any(_evidence_text(note), _NOTE_SUPPORT_TERMS):
            continue
        try:
            distance = abs((date.fromisoformat(note.date) - current).days)
        except ValueError:
            continue
        if distance <= 3:
            related.append(note)
    return related


def _safe_conflict_summary(event_type: str, source_summary: str) -> str:
    if event_type == "minor_misunderstanding":
        return f"軽いすれ違い候補: {_strip_certainty(source_summary)}"
    return f"喧嘩候補: {_strip_certainty(source_summary)}"


def _strip_certainty(text: str) -> str:
    return (
        text.replace("確実に", "")
        .replace("断定", "候補")
        .replace("怒って", "感情は不明で")
        .strip()
    )


def _evidence_text(item: AdapterEvidence | EvidenceItem) -> str:
    metadata_tags = getattr(item, "metadata", {}).get("tags", ())
    return " ".join([item.summary, item.excerpt or "", " ".join(str(tag) for tag in metadata_tags)])


def _contains_any(text: str, terms: tuple[str, ...]) -> bool:
    return any(term in text for term in terms)


def _in_range(value: str, date_from: str | None, date_to: str | None) -> bool:
    if date_from and value < date_from:
        return False
    if date_to and value > date_to:
        return False
    return True
