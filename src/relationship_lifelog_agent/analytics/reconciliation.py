from __future__ import annotations

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
