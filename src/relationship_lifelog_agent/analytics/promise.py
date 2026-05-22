from __future__ import annotations

from relationship_lifelog_agent.analytics.evidence_scoring import AnalysisResult


def promise_placeholder() -> AnalysisResult:
    return AnalysisResult(
        intent="promise_tracking",
        summary="約束トラッキングは MVP では placeholder です。",
        aggregate={"status": "not_implemented"},
        confidence=0.0,
        evidence_strength=0.0,
        cautions=["予定の実行状況は LINE・写真・場所などの根拠を分けて確認します。"],
    )
