from __future__ import annotations

from collections import Counter, defaultdict
from datetime import date
from dataclasses import dataclass, replace
from typing import Any

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
from relationship_lifelog_agent.agent.memory import build_memory
from relationship_lifelog_agent.agent.planner import AdapterCall, QueryPlan, build_plan
from relationship_lifelog_agent.agent.router import route_question
from relationship_lifelog_agent.analytics.evidence_scoring import AnalysisResult, mean
from relationship_lifelog_agent.config import Settings
from relationship_lifelog_agent.db.repository import RelationshipRepository
from relationship_lifelog_agent.profiles import (
    ProfileContext,
    is_relationship_profile_question,
    load_profile_context,
    profile_missing_guidance,
)


CONFLICT_TYPES = {"conflict", "minor_misunderstanding"}
_PROFILE_ARG_UNSET = object()


@dataclass(frozen=True)
class ReviewTarget:
    event_id: int
    label: str
    event_type: str
    review_status: str


@dataclass(frozen=True)
class ChatAnswer:
    text: str
    review_targets: tuple[ReviewTarget, ...] = ()


def answer_question(
    question: str,
    mode: str = "private",
    memory: object | None = None,
    settings: Settings | None = None,
    profile_id: int | None | object = _PROFILE_ARG_UNSET,
    date_from: str | None = None,
    date_to: str | None = None,
    post_conflict_window_days: int | float | None = None,
) -> str:
    return answer_chat(
        question,
        mode=mode,
        memory=memory,
        settings=settings,
        profile_id=profile_id,
        date_from=date_from,
        date_to=date_to,
        post_conflict_window_days=post_conflict_window_days,
    ).text


def answer_chat(
    question: str,
    mode: str = "private",
    memory: object | None = None,
    settings: Settings | None = None,
    profile_id: int | None | object = _PROFILE_ARG_UNSET,
    date_from: str | None = None,
    date_to: str | None = None,
    post_conflict_window_days: int | float | None = None,
) -> ChatAnswer:
    route = route_question(question)
    profile = _resolve_profile(settings, profile_id)
    if settings is not None and profile is None and is_relationship_profile_question(question):
        return ChatAnswer(text=compose_answer(_profile_missing_result(route.intents[0]), mode=mode))
    memory = memory or build_memory(settings)
    clean_date_from = _clean_date(date_from)
    clean_date_to = _clean_date(date_to)
    plan = build_plan(route, date_from=clean_date_from, date_to=clean_date_to)
    result = execute_plan(
        plan,
        memory,
        mode=mode,
        profile=profile,
        settings=settings,
        date_from=clean_date_from,
        date_to=clean_date_to,
        post_conflict_window_days=post_conflict_window_days,
    )
    person_names = [profile.profile_name] if profile else None
    return ChatAnswer(
        text=compose_answer(result, mode=mode, person_names=person_names),
        review_targets=_review_targets_from_result(result, mode=mode, person_names=person_names),
    )


def execute_plan(
    plan: QueryPlan,
    memory: object,
    mode: str = "private",
    profile: ProfileContext | None = None,
    settings: Settings | None = None,
    date_from: str | None = None,
    date_to: str | None = None,
    post_conflict_window_days: int | float | None = None,
) -> AnalysisResult:
    results = _run_adapter_calls(plan, memory, mode=mode, profile=profile)
    results["_backend"] = [_backend(memory)]
    results["_window_days"] = [_resolve_window_days(settings, post_conflict_window_days)]
    if settings is not None and profile is not None:
        results["relationship_db.events"] = _load_saved_relationship_events(
            settings,
            profile=profile,
            mode=mode,
            date_from=date_from,
            date_to=date_to,
        )
    intent = plan.primary_intent
    if intent == "conflict_frequency":
        result = _conflict_frequency(results)
    elif intent == "conflict_timeline":
        result = _conflict_timeline(results)
    elif intent == "post_conflict_activity":
        result = _post_conflict_activity(results)
    elif intent == "emotional_note_lookup":
        result = _emotional_note_lookup(results)
    elif intent == "monthly_relationship_review":
        result = _monthly_relationship_review(results, month=plan.month or "2025-01")
    else:
        result = _general_relationship_qa(results)
    result = _with_profile_context(result, profile)
    return _with_adapter_warnings(result, _memory_warnings(memory), _backend(memory))


def _run_adapter_calls(
    plan: QueryPlan,
    memory: object,
    mode: str,
    profile: ProfileContext | None,
) -> dict[str, list[Any]]:
    results: dict[str, list[Any]] = defaultdict(list)
    for call in plan.calls:
        value = _execute_adapter_call(call, memory, mode=mode, profile=profile)
        key = f"{call.adapter}.{call.method}"
        if isinstance(value, list):
            results[key].extend(value)
        else:
            results[key].append(value)
    return dict(results)


def _execute_adapter_call(
    call: AdapterCall,
    memory: object,
    mode: str,
    profile: ProfileContext | None,
) -> Any:
    adapter = getattr(memory, call.adapter)
    method = getattr(adapter, call.method)
    kwargs = dict(call.kwargs)
    kwargs["mode"] = mode
    _apply_profile_filters(call, kwargs, profile)
    if call.query is None:
        return method(**kwargs)
    return method(call.query, **kwargs)


def _conflict_frequency(results: dict[str, list[Any]]) -> AnalysisResult:
    events = _relationship_event_candidates(results)
    counts = Counter(event.event_type for event in events)
    evidence = _evidence_items([*events, *_adapter_items(results, "personal.search_line"), *_adapter_items(results, "notes.search_notes")])
    examples = [
        f"{event.date}: {_severity_label(event.severity)} - {event.summary}"
        for event in events[:4]
    ]
    if not events:
        backend = _backend_from_results(results)
        return AnalysisResult(
            intent="conflict_frequency",
            summary=_no_candidate_summary(backend, "喧嘩候補"),
            aggregate={"total_candidates": 0, "adapter_backend": backend},
            confidence=0.3,
            evidence_strength=0.0,
            cautions=[_backend_caution(backend)],
        )
    backend_label = "mock データ上" if _backend_from_results(results) == "mock" else "上流read-only adapter"
    return AnalysisResult(
        intent="conflict_frequency",
        summary=(
            f"{backend_label}では喧嘩候補が {len(events)} 件あります。"
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
    window_days = _window_days_from_results(results)
    events = [*_adapter_items(results, "personal.search_events"), *_adapter_items(results, "relationship_db.events")]
    conflicts = sorted(
        [event for event in events if isinstance(event, EventEvidence) and event.event_type in CONFLICT_TYPES],
        key=lambda item: item.date,
    )
    if not conflicts:
        conflicts = _line_conflict_candidates(results)
    outings = sorted(
        [event for event in events if isinstance(event, EventEvidence) and event.event_type == "date_or_outing"],
        key=lambda item: item.date,
    )
    media = [item for item in _adapter_items(results, "personal.search_media") if isinstance(item, MediaEvidence)]
    if not outings:
        outings = sorted(_media_outing_candidates(results), key=lambda item: item.date)
    examples: list[str] = []
    days_after_values: list[int] = []
    displayed_outings: list[EventEvidence] = []
    for outing in outings:
        days_after = _days_after_latest_conflict(outing.date, conflicts)
        if days_after is None:
            examples.append(f"{outing.date}: {outing.summary}")
            displayed_outings.append(outing)
            continue
        if days_after > window_days:
            continue
        days_after_values.append(days_after)
        examples.append(f"{outing.date}: 喧嘩候補の {days_after} 日後に外出候補 - {outing.summary}")
        displayed_outings.append(outing)
    evidence = _evidence_items([*conflicts, *displayed_outings, *_adapter_items(results, "personal.search_line"), *media, *_adapter_items(results, "notes.search_notes")])
    summary = "喧嘩候補の後に外出候補があります。ただし、それが仲直りだったとは断定しません。"
    if not conflicts and not displayed_outings:
        summary = _no_candidate_summary(_backend_from_results(results), "喧嘩後の外出候補")
    return AnalysisResult(
        intent="post_conflict_activity",
        summary=summary,
        aggregate={
            "post_conflict_window_days": window_days,
            "conflict_candidates": len(conflicts),
            "activity_candidates": len(displayed_outings),
            "days_after_conflict": ", ".join(str(value) for value in days_after_values) or "unknown",
        },
        examples=examples,
        evidence=evidence,
        confidence=mean([item.confidence for item in [*displayed_outings, *media]]),
        evidence_strength=mean([item.evidence_strength for item in displayed_outings] + [item.confidence for item in media]),
        cautions=[
            "写真や場所だけでは仲直り確定とは言えません。",
            "外出候補は、LINE・場所・メモなどの追加根拠と合わせて確認する必要があります。",
            "相手の気持ちは記録からは断定できません。",
        ],
    )


def _emotional_note_lookup(results: dict[str, list[Any]]) -> AnalysisResult:
    notes = [item for item in _adapter_items(results, "notes.search_notes") if isinstance(item, NoteEvidence)]
    thoughts = [item for item in _adapter_items(results, "notes.search_thoughts") if isinstance(item, ThoughtEvidence)]
    saved_notes = [
        item
        for item in _adapter_items(results, "relationship_db.events")
        if isinstance(item, EventEvidence) and item.event_type == "emotional_note"
    ]
    evidence = _evidence_items([*notes, *thoughts, *saved_notes])
    return AnalysisResult(
        intent="emotional_note_lookup",
        summary="自分側のメモ候補には、反省や不安を含む振り返りがあります。",
        aggregate={
            "note_candidates": len(notes),
            "thought_candidates": len(thoughts),
            "saved_event_candidates": len(saved_notes),
            "observed_dates": ", ".join(item.date for item in [*notes, *thoughts, *saved_notes]),
        },
        examples=[
            f"{item.date}: {item.summary}"
            for item in [*notes, *thoughts, *saved_notes][:4]
        ],
        evidence=evidence,
        confidence=mean([item.confidence for item in [*notes, *thoughts, *saved_notes]]),
        evidence_strength=mean([item.evidence_strength for item in [*notes, *thoughts, *saved_notes]]),
        cautions=[
            "メモは自分側の記録候補として扱い、歌詞・引用・創作の可能性には注意します。",
            "相手の内面はメモから断定しません。",
        ],
    )


def _monthly_relationship_review(results: dict[str, list[Any]], month: str) -> AnalysisResult:
    events = [*_adapter_items(results, "personal.search_events"), *_adapter_items(results, "relationship_db.events")]
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


def _general_relationship_qa(results: dict[str, list[Any]] | None = None) -> AnalysisResult:
    backend = _backend_from_results(results or {})
    return AnalysisResult(
        intent="general_relationship_qa",
        summary="この質問は MVP の汎用回答に回しました。答えられる範囲は、喧嘩候補、外出候補、自分側メモ、月次レビューです。",
        aggregate={"supported_mvp_intents": 4, "adapter_backend": backend},
        confidence=0.35,
        evidence_strength=0.0,
        cautions=[
            _backend_caution(backend),
            "関係ラベルや親密度はAIで推定しません。必要な場合は手動profile設定だけを使います。",
            "相手の内心は記録から断定できません。",
        ],
    )


def _profile_missing_result(intent: str) -> AnalysisResult:
    return AnalysisResult(
        intent=intent,
        summary=profile_missing_guidance(),
        aggregate={
            "profile_configured": False,
            "required_setup": "manual relationship profile",
            "label_source": "user_manual only",
        },
        examples=[
            "CLIで profile create を使い、person_source_id と line_speaker_source_id を手動で設定してください。",
            "relationship_label はCLIで許可されたprivate labelから手動で選びます。",
        ],
        confidence=1.0,
        evidence_strength=0.0,
        cautions=[
            "AIは恋人ラベルを推定しません。",
            "AIはpersonとLINE speakerを自動リンクしません。",
            "public modeではrelationship_labelを表示しません。",
        ],
    )


def _resolve_profile(settings: Settings | None, profile_id: int | None | object) -> ProfileContext | None:
    if settings is None:
        return None
    if profile_id is _PROFILE_ARG_UNSET:
        profile_id = settings.relationship.default_profile_id
    if profile_id is None:
        return None
    return load_profile_context(settings, int(profile_id))


def _load_saved_relationship_events(
    settings: Settings,
    *,
    profile: ProfileContext,
    mode: str,
    date_from: str | None,
    date_to: str | None,
) -> list[EventEvidence]:
    repo = RelationshipRepository(settings.paths.relationship_db)
    rows = repo.list_events(
        profile_id=profile.id,
        status="candidate",
        date_from=date_from,
        date_to=date_to,
        limit=100,
    )
    events: list[EventEvidence] = []
    for row in rows:
        if row["review_status"] == "rejected":
            continue
        evidence_rows = repo.list_evidence(event_id=int(row["id"]), limit=8)
        pointer = _first_source_pointer(evidence_rows, fallback=f"relationship_event:{row['id']}")
        source_type = _first_source_type(evidence_rows, fallback="manual")
        summary = _review_status_summary_prefix(str(row["review_status"])) + str(row["summary"])
        events.append(
            EventEvidence(
                source_id=f"relationship-db-event-{row['id']}",
                date=str(row["event_date"]),
                role="primary",
                summary=_safe_db_text(summary, mode=mode, person_names=[profile.profile_name]),
                excerpt=None,
                confidence=float(row["confidence"]),
                evidence_strength=float(row["evidence_strength"]),
                sensitivity="private",
                source_pointer=pointer,
                event_type=str(row["event_type"]),
                review_status=str(row["review_status"]),
                severity=int(row["severity"]),
                metadata={
                    "source": "relationship_db",
                    "relationship_event_id": int(row["id"]),
                    "event_type": str(row["event_type"]),
                    "review_status": str(row["review_status"]),
                    "event_date": str(row["event_date"]),
                    "primary_source_type": source_type,
                },
            )
        )
    return events


def _apply_profile_filters(call: AdapterCall, kwargs: dict[str, Any], profile: ProfileContext | None) -> None:
    if profile is None:
        return
    if call.method == "search_line" and profile.line_speaker_source_id:
        kwargs.setdefault("speaker_id", profile.line_speaker_source_id)
    if call.method in {"search_media", "search_events"} and profile.person_source_id:
        kwargs.setdefault("person_id", profile.person_source_id)


def _with_profile_context(result: AnalysisResult, profile: ProfileContext | None) -> AnalysisResult:
    if profile is None:
        return result
    aggregate = {
        **result.aggregate,
        "profile_configured": True,
        "profile_id": profile.id,
    }
    cautions = [
        *result.cautions,
        "選択profileはユーザー手動設定のみを使います。関係ラベルやID対応はAIで推定しません。",
    ]
    return replace(result, aggregate=aggregate, cautions=cautions)


def _review_targets_from_result(
    result: AnalysisResult,
    *,
    mode: str,
    person_names: list[str] | None,
) -> tuple[ReviewTarget, ...]:
    targets: list[ReviewTarget] = []
    seen: set[int] = set()
    for item in result.evidence:
        event_id = item.metadata.get("relationship_event_id")
        if event_id is None:
            continue
        parsed_id = int(event_id)
        if parsed_id in seen:
            continue
        seen.add(parsed_id)
        label = _safe_db_text(
            f"{parsed_id}: {item.date} {item.summary}",
            mode=mode,
            person_names=person_names or [],
        )
        targets.append(
            ReviewTarget(
                event_id=parsed_id,
                label=label,
                event_type=str(item.metadata.get("event_type", "other")),
                review_status=str(item.metadata.get("review_status", "unreviewed")),
            )
        )
    return tuple(targets)


def _relationship_event_candidates(results: dict[str, list[Any]]) -> list[EventEvidence]:
    events = [*_adapter_items(results, "personal.search_events"), *_adapter_items(results, "relationship_db.events")]
    candidates = [
            event
            for event in events
            if isinstance(event, EventEvidence) and event.event_type in CONFLICT_TYPES
        ]
    if not candidates:
        candidates = _line_conflict_candidates(results)
    return sorted(candidates, key=lambda item: item.date)


def _line_conflict_candidates(results: dict[str, list[Any]]) -> list[EventEvidence]:
    candidates: list[EventEvidence] = []
    for item in _adapter_items(results, "personal.search_line"):
        if not isinstance(item, AdapterEvidence):
            continue
        text = f"{item.summary} {item.excerpt or ''}"
        if not _text_has_any(text, ("喧嘩", "けんか", "すれ違い", "謝罪", "不満", "予定変更")):
            continue
        event_type = "minor_misunderstanding" if _text_has_any(text, ("すれ違い", "返信")) else "conflict"
        candidates.append(
            EventEvidence(
                source_id=f"line-candidate-{item.source_id}",
                date=item.date,
                role="primary",
                summary=item.summary,
                confidence=min(item.confidence, 0.62),
                evidence_strength=min(item.evidence_strength, 0.55),
                sensitivity=item.sensitivity,
                source_pointer=item.source_pointer,
                event_type=event_type,
                review_status="candidate",
                severity=1 if event_type == "minor_misunderstanding" else 2,
                metadata={"derived_from": item.source_pointer, "adapter_backend": _backend_from_results(results)},
            )
        )
    return candidates


def _media_outing_candidates(results: dict[str, list[Any]]) -> list[EventEvidence]:
    candidates: list[EventEvidence] = []
    for item in _adapter_items(results, "personal.search_media"):
        if not isinstance(item, MediaEvidence):
            continue
        candidates.append(
            EventEvidence(
                source_id=f"media-outing-{item.source_id}",
                date=item.date,
                role="after_event",
                summary=item.summary,
                confidence=min(item.confidence, 0.6),
                evidence_strength=min(item.evidence_strength, 0.55),
                sensitivity=item.sensitivity,
                source_pointer=item.source_pointer,
                event_type="date_or_outing",
                review_status="candidate",
                severity=0,
                metadata={"derived_from": item.source_pointer, "adapter_backend": _backend_from_results(results)},
            )
        )
    return candidates


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


def _backend(memory: object) -> str:
    return str(getattr(memory, "backend", "mock"))


def _backend_from_results(results: dict[str, list[Any]]) -> str:
    values = results.get("_backend", [])
    if values:
        return str(values[0])
    return "mock"


def _window_days_from_results(results: dict[str, list[Any]]) -> int:
    values = results.get("_window_days", [])
    if not values:
        return 14
    try:
        return max(1, int(values[0]))
    except (TypeError, ValueError):
        return 14


def _resolve_window_days(settings: Settings | None, value: int | float | None) -> int:
    if value is not None:
        return max(1, int(value))
    if settings is not None:
        return max(1, int(settings.relationship.post_conflict_window_days))
    return 14


def _clean_date(value: str | None) -> str | None:
    if value is None:
        return None
    stripped = value.strip()
    return stripped or None


def _first_source_pointer(evidence_rows: list[dict[str, Any]], *, fallback: str) -> str:
    for row in evidence_rows:
        pointer = row.get("source_pointer")
        if pointer:
            return str(pointer)
    return fallback


def _first_source_type(evidence_rows: list[dict[str, Any]], *, fallback: str) -> str:
    for row in evidence_rows:
        source_type = row.get("source_type")
        if source_type:
            return str(source_type)
    return fallback


def _safe_db_text(text: str, *, mode: str, person_names: list[str]) -> str:
    from relationship_lifelog_agent.privacy.guard import sanitize_answer

    return sanitize_answer(text, mode=mode, person_names=person_names)


def _review_status_summary_prefix(review_status: str) -> str:
    if review_status == "verified":
        return "人間確認済み: "
    if review_status == "corrected":
        return "人間修正済み: "
    return ""


def _memory_warnings(memory: object) -> list[str]:
    warnings = getattr(memory, "warnings", None)
    if callable(warnings):
        return warnings()
    collected: list[str] = []
    for adapter_name in ("personal", "notes"):
        adapter = getattr(memory, adapter_name, None)
        adapter_warnings = getattr(adapter, "warnings", None)
        if callable(adapter_warnings):
            collected.extend(adapter_warnings())
    return list(dict.fromkeys(collected))


def _with_adapter_warnings(result: AnalysisResult, warnings: list[str], backend: str) -> AnalysisResult:
    cautions = [*result.cautions]
    for warning in warnings:
        if warning not in cautions:
            cautions.append(warning)
    if backend == "upstream_readonly" and not warnings:
        cautions.append("上流DBはread-only接続で参照し、relationship DBへのrawデータ大量コピーは行いません。")
    return replace(result, cautions=cautions)


def _no_candidate_summary(backend: str, target: str) -> str:
    if backend == "upstream_readonly":
        return f"上流read-only adapterでは、現在の設定と検索範囲で{target}は見つかりませんでした。"
    return f"mock データ上では{target}は見つかりませんでした。"


def _backend_caution(backend: str) -> str:
    if backend == "upstream_readonly":
        return "上流adapterが未設定または根拠不足の場合は、source pointer と短いexcerptのみを安全に扱います。"
    return "実データ統合前なので、具体的な事実確認は mock データに限定されます。"


def _text_has_any(value: object, needles: tuple[str, ...]) -> bool:
    text = str(value or "")
    return any(needle in text for needle in needles)
