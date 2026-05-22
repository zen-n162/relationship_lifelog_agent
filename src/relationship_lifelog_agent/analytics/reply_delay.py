from __future__ import annotations

from relationship_lifelog_agent.analytics.evidence_scoring import AnalysisResult


def reply_delay_placeholder() -> AnalysisResult:
    return AnalysisResult(
        intent="reply_delay_analysis",
        summary="返信遅延分析は MVP では placeholder です。",
        aggregate={"status": "not_implemented"},
        confidence=0.0,
        evidence_strength=0.0,
        cautions=["返信遅延から相手の感情や関係状態は推定しません。"],
    )
