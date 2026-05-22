from __future__ import annotations

from datetime import date

from relationship_lifelog_agent.adapters.mock import MockRelationshipMemory
from relationship_lifelog_agent.adapters.types import EventEvidence, MediaEvidence, PostConflictActivity, RelationshipEvent
from relationship_lifelog_agent.analytics.evidence_scoring import AnalysisResult, mean


def post_conflict_activity(
    memory: MockRelationshipMemory,
    date_from: str | None = None,
    date_to: str | None = None,
) -> AnalysisResult:
    activities = memory.search_post_conflict_activities(date_from=date_from, date_to=date_to)
    evidence = [item for activity in activities for item in activity.evidence]
    examples = [
        f"{activity.date}: 喧嘩候補の {activity.days_after_conflict} 日後に {activity.place_label} の外出候補"
        for activity in activities
    ]
    return AnalysisResult(
        intent="post_conflict_activity",
        summary="喧嘩候補の後に外出候補がいくつかあります。ただし、それが仲直りだったとは断定しません。",
        aggregate={
            "post_conflict_window_days": 14,
            "activity_candidates": len(activities),
        },
        examples=examples,
        evidence=evidence,
        confidence=mean([activity.confidence for activity in activities]),
        evidence_strength=mean([activity.evidence_strength for activity in activities]),
        cautions=[
            "写真や場所だけでは仲直り確定とは言えません。",
            "外出候補は、LINE・場所・メモなどの追加根拠と合わせて確認する必要があります。",
        ],
    )


def build_post_conflict_outings(
    *,
    conflict_candidates: list[RelationshipEvent],
    media_evidence: list[MediaEvidence],
    event_evidence: list[EventEvidence],
    window_days: int = 14,
) -> list[PostConflictActivity]:
    activities: list[PostConflictActivity] = []
    for conflict in conflict_candidates:
        for event in event_evidence:
            if event.event_type != "date_or_outing":
                continue
            days_after = _days_after(conflict.date, event.date)
            if days_after is None or not 0 <= days_after <= window_days:
                continue
            activities.append(
                PostConflictActivity(
                    conflict_event_id=conflict.event_id,
                    activity_event_id=f"dry-run-outing-{event.source_id}",
                    date=event.date,
                    days_after_conflict=days_after,
                    place_label=_place_label(event.summary),
                    activity_type="outing_candidate",
                    confidence=min(event.confidence, 0.62),
                    evidence_strength=min(event.evidence_strength, 0.66),
                    evidence=(event.as_evidence_item(),),
                )
            )
        for media in media_evidence:
            days_after = _days_after(conflict.date, media.date)
            if days_after is None or not 0 <= days_after <= window_days:
                continue
            activities.append(
                PostConflictActivity(
                    conflict_event_id=conflict.event_id,
                    activity_event_id=f"dry-run-outing-media-{media.source_id}",
                    date=media.date,
                    days_after_conflict=days_after,
                    place_label=_place_label(media.summary),
                    activity_type="outing_candidate",
                    confidence=min(media.confidence, 0.58),
                    evidence_strength=min(media.evidence_strength, 0.6),
                    evidence=(media.as_evidence_item(),),
                )
            )
    return _dedupe_activities(sorted(activities, key=lambda item: (item.date, item.activity_event_id)))


def _dedupe_activities(activities: list[PostConflictActivity]) -> list[PostConflictActivity]:
    seen: set[tuple[str, str, str]] = set()
    unique: list[PostConflictActivity] = []
    for activity in activities:
        key = (activity.conflict_event_id, activity.activity_event_id, activity.date)
        if key in seen:
            continue
        seen.add(key)
        unique.append(activity)
    return unique


def _days_after(base: str, later: str) -> int | None:
    try:
        return (date.fromisoformat(later) - date.fromisoformat(base)).days
    except ValueError:
        return None


def _place_label(summary: str) -> str:
    for label in ("カフェ", "食事", "外出", "場所"):
        if label in summary:
            return f"{label}候補"
    return "外出候補"
