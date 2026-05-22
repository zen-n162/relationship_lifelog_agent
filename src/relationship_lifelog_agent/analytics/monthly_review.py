from __future__ import annotations

from relationship_lifelog_agent.adapters.mock import MockRelationshipMemory
from relationship_lifelog_agent.adapters.types import MonthlyReflection, NoteEvidence, PostConflictActivity, RelationshipEvent
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


def build_monthly_relationship_summary(
    *,
    month: str,
    conflict_candidates: list[RelationshipEvent],
    reconciliation_candidates: list[RelationshipEvent],
    post_conflict_outings: list[PostConflictActivity],
    emotional_notes: list[NoteEvidence],
    monthly_reflection: MonthlyReflection | None = None,
) -> dict[str, object]:
    month_conflicts = [item for item in conflict_candidates if item.date.startswith(month)]
    month_reconciliations = [item for item in reconciliation_candidates if item.date.startswith(month)]
    month_outings = [item for item in post_conflict_outings if item.date.startswith(month)]
    month_notes = [item for item in emotional_notes if item.date.startswith(month)]
    summary_parts = [
        f"{month} の候補集計",
        f"喧嘩候補 {len([item for item in month_conflicts if item.event_type == 'conflict'])} 件",
        f"軽いすれ違い候補 {len([item for item in month_conflicts if item.event_type == 'minor_misunderstanding'])} 件",
        f"仲直り候補 {len(month_reconciliations)} 件",
        f"喧嘩後の外出候補 {len(month_outings)} 件",
        f"感情メモ候補 {len(month_notes)} 件",
    ]
    if monthly_reflection is not None:
        summary_parts.append("月次振り返り候補あり")
    return {
        "month": month,
        "summary": "、".join(summary_parts) + "。",
        "conflict_candidates": len([item for item in month_conflicts if item.event_type == "conflict"]),
        "minor_misunderstanding_candidates": len([item for item in month_conflicts if item.event_type == "minor_misunderstanding"]),
        "reconciliation_candidates": len(month_reconciliations),
        "post_conflict_outing_candidates": len(month_outings),
        "emotional_note_candidates": len(month_notes),
    }
