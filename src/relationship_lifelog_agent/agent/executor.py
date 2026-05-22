from __future__ import annotations

from collections import Counter, defaultdict
from datetime import date
from typing import Any

from relationship_lifelog_agent.adapters.mock import MockRelationshipMemory
from relationship_lifelog_agent.adapters.types import (
    AdapterEvidence,
    EventEvidence,
    EvidenceItem,
    MediaEvidence,
    MonthlyReflection,
    NoteEvidence,
    ThoughtEvidence,
)
from relationship_lifelog_agent.agent.answer_composer import compose_answer
from relationship_lifelog_agent.agent.memory import build_mock_memory
from relationship_lifelog_agent.agent.planner import AdapterCall, QueryPlan, build_plan
from relationship_lifelog_agent.agent.router import route_question
from relationship_lifelog_agent.analytics.evidence_scoring import AnalysisResult, mean


CONFLICT_TYPES = {"conflict", "minor_misunderstanding"}


def answer_question(
    question: str,
    mode: str = "private",
    memory: MockRelationshipMemory | None = None,
) -> str:
    memory = memory or build_mock_memory()
    route = route_question(question)
    plan = build_plan(route)
    result = execute_plan(plan, memory)
    return compose_answer(result, mode=mode)


def execute_plan(plan: QueryPlan, memory: MockRelationshipMemory) -> AnalysisResult:
    results = _run_adapter_calls(plan, memory)
    intent = plan.primary_intent
    if intent == "conflict_frequency":
        return _conflict_frequency(results)
    if intent == "conflict_timeline":
        return _conflict_timeline(results)
    if intent == "post_conflict_activity":
        return _post_conflict_activity(results)
    if intent == "emotional_note_lookup":
        return _emotional_note_lookup(results)
    if intent == "monthly_relationship_review":
        return _monthly_relationship_review(results, month=plan.month or "2025-01")
    return _general_relationship_qa()


def _run_adapter_calls(plan: QueryPlan, memory: MockRelationshipMemory) -> dict[str, list[Any]]:
    results: dict[str, list[Any]] = defaultdict(list)
    for call in plan.calls:
        value = _execute_adapter_call(call, memory)
        key = f"{call.adapter}.{call.method}"
        if isinstance(value, list):
            results[key].extend(value)
        else:
            results[key].append(value)
    return dict(results)


def _execute_adapter_call(call: AdapterCall, memory: MockRelationshipMemory) -> Any:
    adapter = getattr(memory, call.adapter)
    method = getattr(adapter, call.method)
    if call.query is None:
        return method(**call.kwargs)
    return method(call.query, **call.kwargs)


def _conflict_frequency(results: dict[str, list[Any]]) -> AnalysisResult:
    events = _relationship_event_candidates(results)
    counts = Counter(event.event_type for event in events)
    evidence = _evidence_items([*events, *_adapter_items(results, "personal.search_line"), *_adapter_items(results, "notes.search_notes")])
    examples = [
        f"{event.date}: {_severity_label(event.severity)} - {event.summary}"
        for event in events[:4]
    ]
    if not events:
        return AnalysisResult(
            intent="conflict_frequency",
            summary="mock データ上では喧嘩候補は見つかりませんでした。",
            aggregate={"total_candidates": 0},
            confidence=0.3,
            evidence_strength=0.0,
            cautions=["実データ統合前なので、mock adapter の範囲だけの結果です。"],
        )
    return AnalysisResult(
        intent="conflict_frequency",
        summary=(
            f"mock データ上では喧嘩候補が {len(events)} 件あります。"
            "これは確定ではなく、人間確認前の候補です。"
        ),
        aggregate={
            "conflict_candidates": counts["conflict"],
            "minor_misunderstanding_candidates": counts["minor_misunderstanding"],
            "total_candidates": len(events),
            "observed_dates": ", ".join(event.date for event in events),
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


def _conflict_timeline(results: dict[str, list[Any]]) -> AnalysisResult:
    events = sorted(_relationship_event_candidates(results), key=lambda item: item.date)
    evidence = _evidence_items([*events, *_adapter_items(results, "personal.search_line"), *_adapter_items(results, "notes.search_notes")])
    return AnalysisResult(
        intent="conflict_timeline",
        summary="喧嘩候補と軽いすれ違い候補を時系列で整理しました。",
        aggregate={"total_candidates": len(events)},
        examples=[
            f"{event.date}: {_severity_label(event.severity)} / confidence={event.confidence:.2f}"
            for event in events
        ],
        evidence=evidence,
        confidence=mean([event.confidence for event in events]),
        evidence_strength=mean([event.evidence_strength for event in events]),
        cautions=[
            "時系列は mock データによる候補であり、実データの確定事実ではありません。",
            "相手の内面は記録から断定しません。",
        ],
    )


def _post_conflict_activity(results: dict[str, list[Any]]) -> AnalysisResult:
    events = _adapter_items(results, "personal.search_events")
    conflicts = sorted(
        [event for event in events if isinstance(event, EventEvidence) and event.event_type in CONFLICT_TYPES],
        key=lambda item: item.date,
    )
    outings = sorted(
        [event for event in events if isinstance(event, EventEvidence) and event.event_type == "date_or_outing"],
        key=lambda item: item.date,
    )
    media = [item for item in _adapter_items(results, "personal.search_media") if isinstance(item, MediaEvidence)]
    examples: list[str] = []
    days_after_values: list[int] = []
    for outing in outings:
        days_after = _days_after_latest_conflict(outing.date, conflicts)
        if days_after is None:
            examples.append(f"{outing.date}: {outing.summary}")
            continue
        days_after_values.append(days_after)
        examples.append(f"{outing.date}: 喧嘩候補の {days_after} 日後に外出候補 - {outing.summary}")
    evidence = _evidence_items([*conflicts, *outings, *_adapter_items(results, "personal.search_line"), *media, *_adapter_items(results, "notes.search_notes")])
    return AnalysisResult(
        intent="post_conflict_activity",
        summary="喧嘩候補の後に外出候補があります。ただし、それが仲直りだったとは断定しません。",
        aggregate={
            "post_conflict_window_days": 14,
            "conflict_candidates": len(conflicts),
            "activity_candidates": len(outings),
            "days_after_conflict": ", ".join(str(value) for value in days_after_values) or "unknown",
        },
        examples=examples,
        evidence=evidence,
        confidence=mean([item.confidence for item in [*outings, *media]]),
        evidence_strength=mean([item.evidence_strength for item in outings] + [item.confidence for item in media]),
        cautions=[
            "写真や場所だけでは仲直り確定とは言えません。",
            "外出候補は、LINE・場所・メモなどの追加根拠と合わせて確認する必要があります。",
            "相手の気持ちは記録からは断定できません。",
        ],
    )


def _emotional_note_lookup(results: dict[str, list[Any]]) -> AnalysisResult:
    notes = [item for item in _adapter_items(results, "notes.search_notes") if isinstance(item, NoteEvidence)]
    thoughts = [item for item in _adapter_items(results, "notes.search_thoughts") if isinstance(item, ThoughtEvidence)]
    evidence = _evidence_items([*notes, *thoughts])
    return AnalysisResult(
        intent="emotional_note_lookup",
        summary="自分側のメモ候補には、反省や不安を含む振り返りがあります。",
        aggregate={
            "note_candidates": len(notes),
            "thought_candidates": len(thoughts),
            "observed_dates": ", ".join(item.date for item in [*notes, *thoughts]),
        },
        examples=[
            f"{item.date}: {item.summary}"
            for item in [*notes, *thoughts][:4]
        ],
        evidence=evidence,
        confidence=mean([item.confidence for item in [*notes, *thoughts]]),
        evidence_strength=mean([item.confidence for item in [*notes, *thoughts]]),
        cautions=[
            "メモは自分側の記録候補として扱い、歌詞・引用・創作の可能性には注意します。",
            "相手の内面はメモから断定しません。",
        ],
    )


def _monthly_relationship_review(results: dict[str, list[Any]], month: str) -> AnalysisResult:
    events = _adapter_items(results, "personal.search_events")
    conflicts = [event for event in events if isinstance(event, EventEvidence) and event.event_type in CONFLICT_TYPES]
    outings = [event for event in events if isinstance(event, EventEvidence) and event.event_type == "date_or_outing"]
    media = _adapter_items(results, "personal.search_media")
    notes = _adapter_items(results, "notes.search_notes")
    reflections = [item for item in _adapter_items(results, "notes.get_monthly_reflection") if isinstance(item, MonthlyReflection)]
    month_summary = _month_summary_text(results)
    evidence = _evidence_items([*conflicts, *outings, *media, *notes, *reflections])
    examples = [
        *[f"{event.date}: {_severity_label(event.severity)} - {event.summary}" for event in conflicts[:2]],
        *[f"{event.date}: 外出候補 - {event.summary}" for event in outings[:2]],
        *[f"{item.date}: 月次振り返り候補 - {item.summary}" for item in reflections[:1]],
    ]
    return AnalysisResult(
        intent="monthly_relationship_review",
        summary=f"{month} は、喧嘩候補・外出候補・自分側の振り返りを分けて見るのがよさそうです。{month_summary}",
        aggregate={
            "month": month,
            "conflict_total_candidates": len(conflicts),
            "post_conflict_activity_candidates": len(outings),
            "note_candidates": len(notes),
        },
        examples=examples,
        evidence=evidence,
        confidence=mean([item.confidence for item in [*conflicts, *outings, *media, *notes, *reflections]]),
        evidence_strength=mean([getattr(item, "evidence_strength", item.confidence) for item in [*conflicts, *outings, *media, *notes, *reflections]]),
        cautions=[
            "月次レビューは関係の良し悪しを採点するものではありません。",
            "相手の内面や関係ラベルは記録から推定しません。",
        ],
    )


def _general_relationship_qa() -> AnalysisResult:
    return AnalysisResult(
        intent="general_relationship_qa",
        summary="この質問は MVP の汎用回答に回しました。mock データで答えられる範囲は、喧嘩候補、外出候補、自分側メモ、月次レビューです。",
        aggregate={"supported_mvp_intents": 4},
        confidence=0.35,
        evidence_strength=0.0,
        cautions=["実データ統合前なので、具体的な事実確認は mock データに限定されます。"],
    )


def _relationship_event_candidates(results: dict[str, list[Any]]) -> list[EventEvidence]:
    events = _adapter_items(results, "personal.search_events")
    return sorted(
        [
            event
            for event in events
            if isinstance(event, EventEvidence) and event.event_type in CONFLICT_TYPES
        ],
        key=lambda item: item.date,
    )


def _adapter_items(results: dict[str, list[Any]], key: str) -> list[Any]:
    return results.get(key, [])


def _evidence_items(items: list[Any]) -> list[EvidenceItem]:
    evidence: list[EvidenceItem] = []
    seen: set[tuple[str, str]] = set()
    for item in items:
        converted = _to_evidence_item(item)
        if converted is None:
            continue
        key = (converted.source_type, converted.source_id)
        if key in seen:
            continue
        seen.add(key)
        evidence.append(converted)
    return evidence


def _to_evidence_item(item: Any) -> EvidenceItem | None:
    if isinstance(item, EvidenceItem):
        return item
    if isinstance(item, AdapterEvidence):
        return item.as_evidence_item()
    return None


def _days_after_latest_conflict(activity_date: str, conflicts: list[EventEvidence]) -> int | None:
    activity = date.fromisoformat(activity_date)
    prior_conflicts = [date.fromisoformat(item.date) for item in conflicts if date.fromisoformat(item.date) <= activity]
    if not prior_conflicts:
        return None
    return (activity - max(prior_conflicts)).days


def _severity_label(severity: int) -> str:
    if severity >= 3:
        return "強めの喧嘩候補"
    if severity == 2:
        return "中程度の喧嘩候補"
    if severity == 1:
        return "軽いすれ違い候補"
    return "関係イベント候補"


def _month_summary_text(results: dict[str, list[Any]]) -> str:
    summaries = [
        item.get("summary", "")
        for item in _adapter_items(results, "personal.get_month_summary")
        if isinstance(item, dict)
    ]
    if not summaries:
        return ""
    return f"月次サマリ候補: {summaries[0]}"
