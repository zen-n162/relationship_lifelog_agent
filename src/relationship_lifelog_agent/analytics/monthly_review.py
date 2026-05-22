from __future__ import annotations

from relationship_lifelog_agent.adapters.mock import MockRelationshipMemory
from relationship_lifelog_agent.analytics.conflict import conflict_frequency
from relationship_lifelog_agent.analytics.evidence_scoring import AnalysisResult
from relationship_lifelog_agent.analytics.outing import post_conflict_activity


def monthly_relationship_review(memory: MockRelationshipMemory, month: str) -> AnalysisResult:
    date_from = f"{month}-01"
    date_to = f"{month}-31"
    conflicts = conflict_frequency(memory, date_from=date_from, date_to=date_to)
    outings = post_conflict_activity(memory, date_from=date_from, date_to=date_to)
    reflection = memory.get_monthly_reflection(month)
    examples = [
        *conflicts.examples[:2],
        *outings.examples[:2],
        f"{reflection.date}: 月次振り返り候補 - {reflection.summary}",
    ]
    return AnalysisResult(
        intent="monthly_relationship_review",
        summary=f"{month} は、喧嘩候補・軽いすれ違い候補・外出候補・自分側の振り返りを分けて見るのがよさそうです。",
        aggregate={
            "month": month,
            "conflict_total_candidates": conflicts.aggregate.get("total_candidates", 0),
            "post_conflict_activity_candidates": outings.aggregate.get("activity_candidates", 0),
        },
        examples=examples,
        evidence=[*conflicts.evidence, *outings.evidence, reflection],
        confidence=min(0.65, max(conflicts.confidence, outings.confidence, reflection.confidence)),
        evidence_strength=min(0.7, max(conflicts.evidence_strength, outings.evidence_strength, reflection.confidence)),
        cautions=[
            "月次レビューは関係の良し悪しを採点するものではありません。",
            "相手の内面や関係ラベルは記録から推定しません。",
        ],
    )
