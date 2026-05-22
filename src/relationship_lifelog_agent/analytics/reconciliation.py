from __future__ import annotations

from datetime import date

from relationship_lifelog_agent.adapters.types import EventEvidence, LineEvidence, MediaEvidence, RelationshipEvent
from relationship_lifelog_agent.analytics.evidence_scoring import AnalysisResult


def reconciliation_placeholder() -> AnalysisResult:
    return AnalysisResult(
        intent="reconciliation_duration",
        summary="仲直り期間の分析は MVP では placeholder です。",
        aggregate={"status": "not_implemented"},
        confidence=0.0,
        evidence_strength=0.0,
        cautions=["仲直りは写真や場所だけでは断定しません。"],
    )


_RECONCILIATION_TERMS = ("仲直り", "通常会話", "大丈夫", "気にしない", "会う予定", "予定を確認", "謝罪後")


def build_reconciliation_candidates(
    *,
    conflict_candidates: list[RelationshipEvent],
    line_evidence: list[LineEvidence],
    event_evidence: list[EventEvidence],
    media_evidence: list[MediaEvidence],
    window_days: int = 14,
) -> list[RelationshipEvent]:
    candidates: list[RelationshipEvent] = []
    seen: set[tuple[str, str]] = set()
    for conflict in conflict_candidates:
        for line in line_evidence:
            days_after = _days_after(conflict.date, line.date)
            if days_after is None or not 0 <= days_after <= window_days:
                continue
            if not _has_reconciliation_signal(line.summary, line.excerpt):
                continue
            key = ("line", line.source_id)
            if key in seen:
                continue
            candidates.append(
                RelationshipEvent(
                    event_id=f"dry-run-reconciliation-{line.source_id}",
                    event_type="reconciliation",
                    date=line.date,
                    summary="仲直り候補: 謝罪後または喧嘩候補後に通常のやりとりへ戻った可能性があります。",
                    review_status="candidate",
                    confidence=min(max(line.confidence, 0.48), 0.66),
                    evidence_strength=min(max(line.evidence_strength, 0.42), 0.68),
                    severity=0,
                    evidence=(line.as_evidence_item(),),
                    metadata={"dry_run": True, "days_after_conflict": days_after},
                )
            )
            seen.add(key)
        for event in event_evidence:
            days_after = _days_after(conflict.date, event.date)
            if days_after is None or not 0 <= days_after <= window_days:
                continue
            if event.event_type not in {"reconciliation", "date_or_outing", "call_event"}:
                continue
            key = ("event", event.source_id)
            if key in seen:
                continue
            candidates.append(
                RelationshipEvent(
                    event_id=f"dry-run-reconciliation-{event.source_id}",
                    event_type="reconciliation",
                    date=event.date,
                    summary="仲直り候補: 喧嘩候補後に会話・通話・外出に関連する候補があります。",
                    review_status="candidate",
                    confidence=min(max(event.confidence, 0.42), 0.62),
                    evidence_strength=min(max(event.evidence_strength, 0.38), 0.64),
                    severity=0,
                    evidence=(event.as_evidence_item(),),
                    metadata={"dry_run": True, "days_after_conflict": days_after},
                )
            )
            seen.add(key)
        for media in media_evidence:
            days_after = _days_after(conflict.date, media.date)
            if days_after is None or not 0 <= days_after <= window_days:
                continue
            key = ("media", media.source_id)
            if key in seen:
                continue
            candidates.append(
                RelationshipEvent(
                    event_id=f"dry-run-reconciliation-media-{media.source_id}",
                    event_type="reconciliation",
                    date=media.date,
                    summary="仲直り候補: 喧嘩候補後に外出・場所に関する候補があります。",
                    review_status="candidate",
                    confidence=min(media.confidence, 0.52),
                    evidence_strength=min(media.evidence_strength, 0.55),
                    severity=0,
                    evidence=(media.as_evidence_item(),),
                    metadata={"dry_run": True, "days_after_conflict": days_after, "caution": "media alone is not definitive"},
                )
            )
            seen.add(key)
    return sorted(candidates, key=lambda item: item.date)


def _has_reconciliation_signal(summary: str, excerpt: str | None) -> bool:
    text = f"{summary} {excerpt or ''}"
    return any(term in text for term in _RECONCILIATION_TERMS)


def _days_after(base: str, later: str) -> int | None:
    try:
        return (date.fromisoformat(later) - date.fromisoformat(base)).days
    except ValueError:
        return None
