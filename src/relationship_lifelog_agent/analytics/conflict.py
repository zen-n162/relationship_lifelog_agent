from __future__ import annotations

from collections import Counter

from relationship_lifelog_agent.adapters.mock import MockRelationshipMemory
from relationship_lifelog_agent.analytics.evidence_scoring import AnalysisResult, mean


CONFLICT_TYPES = {"conflict", "minor_misunderstanding"}


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
