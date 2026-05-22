from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from relationship_lifelog_agent.adapters.types import (
    AdapterEvidence,
    EventEvidence,
    EvidenceItem,
    LineEvidence,
    MediaEvidence,
    MonthlyReflection,
    NoteEvidence,
    PostConflictActivity,
    RelationshipEvent,
)
from relationship_lifelog_agent.analytics.conflict import build_conflict_candidates, conflict_counts
from relationship_lifelog_agent.analytics.monthly_review import build_monthly_relationship_summary
from relationship_lifelog_agent.analytics.outing import build_post_conflict_outings
from relationship_lifelog_agent.analytics.reconciliation import build_reconciliation_candidates
from relationship_lifelog_agent.config import Settings
from relationship_lifelog_agent.privacy.guard import sanitize_answer
from relationship_lifelog_agent.privacy.redaction import redact_evidence_for_mode
from relationship_lifelog_agent.profiles import ProfileContext


@dataclass(frozen=True)
class DryRunResult:
    report_markdown: str
    source_counts: dict[str, int]
    conflict_candidates: list[RelationshipEvent]
    minor_misunderstanding_candidates: list[RelationshipEvent]
    reconciliation_candidates: list[RelationshipEvent]
    post_conflict_outings: list[PostConflictActivity]
    emotional_notes: list[NoteEvidence]
    warnings: list[str]


def run_relationship_dry_run(
    *,
    memory: object,
    settings: Settings,
    profile: ProfileContext | None,
    date_from: str,
    date_to: str,
    backend: str,
    mode: str = "private",
    output_path: str | Path | None = None,
) -> DryRunResult:
    fetched = _fetch_sources(memory, profile=profile, date_from=date_from, date_to=date_to, backend=backend, mode=mode)
    conflicts = build_conflict_candidates(
        line_evidence=fetched["line"],
        note_evidence=fetched["notes"],
        event_evidence=fetched["events"],
        date_from=date_from,
        date_to=date_to,
    )
    reconciliations = build_reconciliation_candidates(
        conflict_candidates=conflicts,
        line_evidence=fetched["line"],
        event_evidence=fetched["events"],
        media_evidence=fetched["media"],
        window_days=settings.relationship.post_conflict_window_days,
    )
    outings = build_post_conflict_outings(
        conflict_candidates=conflicts,
        media_evidence=fetched["media"],
        event_evidence=fetched["events"],
        window_days=settings.relationship.post_conflict_window_days,
    )
    emotional_notes = _emotional_notes(fetched["notes"])
    monthly_summaries = [
        build_monthly_relationship_summary(
            month=month,
            conflict_candidates=conflicts,
            reconciliation_candidates=reconciliations,
            post_conflict_outings=outings,
            emotional_notes=emotional_notes,
            monthly_reflection=fetched["monthly_reflections"].get(month),
        )
        for month in _months_between(date_from, date_to)
    ]
    warnings = _memory_warnings(memory)
    report = _render_report(
        date_from=date_from,
        date_to=date_to,
        backend=backend,
        mode=mode,
        profile=profile,
        source_counts=_source_counts(fetched),
        conflicts=conflicts,
        reconciliations=reconciliations,
        outings=outings,
        emotional_notes=emotional_notes,
        monthly_summaries=monthly_summaries,
        warnings=warnings,
    )
    if output_path is not None:
        path = Path(output_path).expanduser()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(report, encoding="utf-8")
    conflict_candidates = [item for item in conflicts if item.event_type == "conflict"]
    minor_candidates = [item for item in conflicts if item.event_type == "minor_misunderstanding"]
    return DryRunResult(
        report_markdown=report,
        source_counts=_source_counts(fetched),
        conflict_candidates=conflict_candidates,
        minor_misunderstanding_candidates=minor_candidates,
        reconciliation_candidates=reconciliations,
        post_conflict_outings=outings,
        emotional_notes=emotional_notes,
        warnings=warnings,
    )


def _fetch_sources(
    memory: object,
    *,
    profile: ProfileContext | None,
    date_from: str,
    date_to: str,
    backend: str,
    mode: str,
) -> dict[str, Any]:
    speaker_id = profile.line_speaker_source_id if profile and backend != "mock" else None
    person_id = profile.person_source_id if profile and backend != "mock" else None
    personal = getattr(memory, "personal")
    notes = getattr(memory, "notes")
    line = personal.search_line("喧嘩 すれ違い 謝罪 仲直り 通常会話", date_from=date_from, date_to=date_to, speaker_id=speaker_id, limit=100, mode=mode)
    media = personal.search_media("喧嘩後 外出 場所 カフェ 食事", date_from=date_from, date_to=date_to, person_id=person_id, limit=100, mode=mode)
    events = personal.search_events(date_from=date_from, date_to=date_to, person_id=person_id, limit=100, mode=mode)
    note_items = notes.search_notes("反省 不安 後悔 感謝 関係", date_from=date_from, date_to=date_to, limit=100, mode=mode)
    thought_items = notes.search_thoughts("反省 不安 自分側", date_from=date_from, date_to=date_to, limit=100, mode=mode)
    monthly_reflections: dict[str, MonthlyReflection] = {}
    for month in _months_between(date_from, date_to):
        reflection = notes.get_monthly_reflection(month, mode=mode)
        if reflection.confidence > 0 or reflection.evidence_strength > 0:
            monthly_reflections[month] = reflection
    return {
        "line": line,
        "media": media,
        "events": events,
        "notes": note_items,
        "thoughts": thought_items,
        "monthly_reflections": monthly_reflections,
    }


def _render_report(
    *,
    date_from: str,
    date_to: str,
    backend: str,
    mode: str,
    profile: ProfileContext | None,
    source_counts: dict[str, int],
    conflicts: list[RelationshipEvent],
    reconciliations: list[RelationshipEvent],
    outings: list[PostConflictActivity],
    emotional_notes: list[NoteEvidence],
    monthly_summaries: list[dict[str, object]],
    warnings: list[str],
) -> str:
    counts = conflict_counts(conflicts)
    sections = [
        "# Relationship Event Candidates Dry Run",
        "## Dry-run status\n- relationship DBには候補を書き込んでいません。\n- 上流データはread-onlyで参照します。\n- raw LINE全文、raw note全文、正確GPS、顔情報、写真実体は出力しません。",
        f"## 対象期間\n- date_from: {date_from}\n- date_to: {date_to}\n- backend: {backend}\n- mode: {mode}",
        "## 使用profile\n" + _format_profile(profile, mode),
        "## 取得できたsource counts\n" + _format_dict(source_counts),
        "## conflict candidates\n" + _format_events([item for item in conflicts if item.event_type == "conflict"], mode),
        "## minor misunderstanding candidates\n" + _format_events([item for item in conflicts if item.event_type == "minor_misunderstanding"], mode),
        "## reconciliation candidates\n" + _format_events(reconciliations, mode),
        "## post-conflict outing candidates\n" + _format_outings(outings, mode),
        "## emotional note candidates\n" + _format_adapter_evidence(emotional_notes, mode),
        "## monthly relationship summary\n" + _format_monthly(monthly_summaries),
        "## evidence strength 分布\n" + _format_strength_distribution([*conflicts, *reconciliations, *emotional_notes]),
        "## candidate counts\n" + _format_dict({**counts, "reconciliation_candidates": len(reconciliations), "post_conflict_outing_candidates": len(outings), "emotional_note_candidates": len(emotional_notes)}),
        "## warnings\n" + _format_warnings(warnings),
    ]
    return sanitize_answer("\n\n".join(sections), mode=mode)


def _format_profile(profile: ProfileContext | None, mode: str) -> str:
    if profile is None:
        return "- profile未設定。関係ラベルやID対応は推定していません。"
    if mode == "public":
        return "\n".join(
            [
                f"- profile_id: {profile.id}",
                "- profile_name: 人物A",
                "- manual_profile_configured: true",
                "- private_label: [relationship label redacted]",
            ]
        )
    lines = [
        f"- profile_id: {profile.id}",
        f"- profile_name: {profile.profile_name}",
        f"- person_source_id: {profile.person_source_id or 'unset'}",
        f"- line_speaker_source_id: {profile.line_speaker_source_id or 'unset'}",
        "- label_source: user_manual",
    ]
    if mode == "private" and profile.relationship_label:
        lines.append(f"- relationship_label: {profile.relationship_label}")
    return "\n".join(lines)


def _format_events(events: list[RelationshipEvent], mode: str) -> str:
    if not events:
        return "- 候補はありません。"
    return "\n".join(
        f"- {event.date}: {event.summary} "
        f"(confidence={event.confidence:.2f}, evidence_strength={event.evidence_strength:.2f}, source={_source_pointer(event.evidence, mode)})"
        for event in events
    )


def _format_outings(outings: list[PostConflictActivity], mode: str) -> str:
    if not outings:
        return "- 候補はありません。"
    lines = []
    for outing in outings:
        evidence = [redact_evidence_for_mode(item, mode=mode) for item in outing.evidence]
        lines.append(
            f"- {outing.date}: 喧嘩候補の{outing.days_after_conflict}日後の外出候補 "
            f"(place={outing.place_label}, confidence={outing.confidence:.2f}, evidence_strength={outing.evidence_strength:.2f}, source={_source_pointer(evidence, mode)})"
        )
    return "\n".join(lines)


def _format_adapter_evidence(items: list[AdapterEvidence], mode: str) -> str:
    if not items:
        return "- 候補はありません。"
    lines = []
    for item in items:
        evidence = redact_evidence_for_mode(item.as_evidence_item(), mode=mode)
        lines.append(
            f"- {evidence.date}: {evidence.summary} "
            f"(confidence={evidence.confidence:.2f}, evidence_strength={evidence.evidence_strength:.2f}, source={evidence.source_pointer})"
        )
    return "\n".join(lines)


def _format_monthly(monthly_summaries: list[dict[str, object]]) -> str:
    if not monthly_summaries:
        return "- 月次summaryはありません。"
    return "\n".join(f"- {item['month']}: {item['summary']}" for item in monthly_summaries)


def _format_strength_distribution(items: list[Any]) -> str:
    values = [float(getattr(item, "evidence_strength", 0.0)) for item in items]
    buckets = {
        "0.00-0.29": sum(1 for value in values if value < 0.3),
        "0.30-0.49": sum(1 for value in values if 0.3 <= value < 0.5),
        "0.50-0.69": sum(1 for value in values if 0.5 <= value < 0.7),
        "0.70-1.00": sum(1 for value in values if value >= 0.7),
    }
    return _format_dict(buckets)


def _format_dict(values: dict[str, object]) -> str:
    if not values:
        return "- none"
    return "\n".join(f"- {key}: {value}" for key, value in values.items())


def _format_warnings(warnings: list[str]) -> str:
    if not warnings:
        return "- 追加warningはありません。"
    return "\n".join(f"- {warning}" for warning in warnings)


def _source_pointer(evidence: tuple[EvidenceItem, ...] | list[EvidenceItem], mode: str) -> str:
    if not evidence:
        return "none"
    safe = redact_evidence_for_mode(evidence[0], mode=mode)
    return str(safe.source_pointer)


def _emotional_notes(notes: list[NoteEvidence]) -> list[NoteEvidence]:
    terms = ("反省", "不安", "後悔", "感謝", "関係")
    return [note for note in notes if any(term in f"{note.summary} {note.excerpt or ''}" for term in terms)]


def _source_counts(fetched: dict[str, Any]) -> dict[str, int]:
    return {
        "line": len(fetched["line"]),
        "media": len(fetched["media"]),
        "events": len(fetched["events"]),
        "notes": len(fetched["notes"]),
        "thoughts": len(fetched["thoughts"]),
        "monthly_reflections": len(fetched["monthly_reflections"]),
    }


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


def _months_between(date_from: str, date_to: str) -> list[str]:
    start_year, start_month = int(date_from[:4]), int(date_from[5:7])
    end_year, end_month = int(date_to[:4]), int(date_to[5:7])
    months: list[str] = []
    year, month = start_year, start_month
    while (year, month) <= (end_year, end_month):
        months.append(f"{year:04d}-{month:02d}")
        month += 1
        if month == 13:
            month = 1
            year += 1
    return months
