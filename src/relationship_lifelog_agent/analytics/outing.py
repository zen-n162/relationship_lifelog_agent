from __future__ import annotations

from relationship_lifelog_agent.adapters.mock import MockRelationshipMemory
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
