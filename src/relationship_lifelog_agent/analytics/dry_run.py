from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import sqlite3
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
from relationship_lifelog_agent.db.repository import RelationshipRepository
from relationship_lifelog_agent.privacy.guard import detect_answer_safety_violations, sanitize_answer
from relationship_lifelog_agent.privacy.redaction import (
    FACE_CROP_RE,
    GPS_RE,
    LINE_FULL_TEXT_RE,
    NOTE_FULL_TEXT_RE,
    PHOTO_PATH_RE,
    PRIVATE_PATH_RE,
    SOURCE_PATH_RE,
    redact_evidence_for_mode,
    redact_public_text,
)
from relationship_lifelog_agent.profiles import ProfileContext


MAX_STORED_SUMMARY_CHARS = 240
MAX_STORED_EXCERPT_CHARS = 120
PRIVACY_LEVELS = ("counts-only", "redacted", "private")


@dataclass(frozen=True)
class DryRunResult:
    report_markdown: str
    source_counts: dict[str, int]
    privacy_level: str
    conflict_candidates: list[RelationshipEvent]
    minor_misunderstanding_candidates: list[RelationshipEvent]
    reconciliation_candidates: list[RelationshipEvent]
    post_conflict_outings: list[PostConflictActivity]
    emotional_notes: list[NoteEvidence]
    warnings: list[str]


@dataclass(frozen=True)
class WriteResult:
    events_written: int
    evidence_written: int
    post_conflict_activities_written: int
    duplicates: int
    warnings: list[str]
    created_event_ids: tuple[int, ...] = ()
    created_activity_ids: tuple[int, ...] = ()


@dataclass(frozen=True)
class WriteSafetyReport:
    candidate_events: int
    candidate_evidence: int
    candidate_post_conflict_activities: int
    duplicate_candidates: int
    raw_leakage_issues: tuple[str, ...]
    forbidden_phrase_issues: tuple[str, ...]
    warnings: tuple[str, ...] = ()

    @property
    def ok(self) -> bool:
        return not self.raw_leakage_issues and not self.forbidden_phrase_issues


def run_relationship_dry_run(
    *,
    memory: object,
    settings: Settings,
    profile: ProfileContext | None,
    date_from: str,
    date_to: str,
    backend: str,
    mode: str = "private",
    privacy_level: str = "redacted",
    output_path: str | Path | None = None,
) -> DryRunResult:
    _validate_privacy_level(privacy_level, mode=mode)
    adapter_mode = _adapter_mode_for_privacy(mode=mode, privacy_level=privacy_level)
    fetched = _fetch_sources(memory, profile=profile, date_from=date_from, date_to=date_to, backend=backend, mode=adapter_mode)
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
    reviewed_snapshot = _load_reviewed_event_snapshot(
        settings=settings,
        profile=profile,
        date_from=date_from,
        date_to=date_to,
    )
    report = _render_report(
        date_from=date_from,
        date_to=date_to,
        backend=backend,
        mode=mode,
        privacy_level=privacy_level,
        profile=profile,
        source_counts=_source_counts(fetched),
        conflicts=conflicts,
        reconciliations=reconciliations,
        outings=outings,
        emotional_notes=emotional_notes,
        monthly_summaries=monthly_summaries,
        reviewed_snapshot=reviewed_snapshot,
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
        privacy_level=privacy_level,
        conflict_candidates=conflict_candidates,
        minor_misunderstanding_candidates=minor_candidates,
        reconciliation_candidates=reconciliations,
        post_conflict_outings=outings,
        emotional_notes=emotional_notes,
        warnings=warnings,
    )


def write_dry_run_candidates(
    *,
    repo: RelationshipRepository,
    result: DryRunResult,
    profile_id: int,
    mode: str = "private",
    privacy_level: str | None = None,
) -> WriteResult:
    """Persist dry-run candidates only when explicitly requested by the user."""

    privacy_level = privacy_level or result.privacy_level
    _validate_privacy_level(privacy_level, mode=mode)
    events_written = 0
    evidence_written = 0
    activities_written = 0
    duplicate_count = 0
    warnings: list[str] = []
    event_id_map: dict[str, int] = {}
    created_event_ids: list[int] = []
    created_activity_ids: list[int] = []
    events = _write_events(result)

    for event in events:
        source_pointer = _event_source_pointer(event)
        existing_id = _find_duplicate_event(
            repo,
            profile_id=profile_id,
            event_type=event.event_type,
            event_date=event.date,
            source_pointer=source_pointer,
            event=event,
        )
        if existing_id is not None:
            event_id_map[event.event_id] = existing_id
            duplicate_count += 1
            warnings.append(
                "duplicate skipped: "
                f"{event.event_type} {event.date} "
                f"{_storage_text(source_pointer, mode=mode, privacy_level=privacy_level, max_chars=180)}"
            )
            continue
        event_id = repo.create_event(
            profile_id=profile_id,
            event_type=event.event_type,
            event_date=event.date,
            summary=_storage_text(
                _event_storage_summary(event, privacy_level=privacy_level),
                mode=mode,
                privacy_level=privacy_level,
                max_chars=MAX_STORED_SUMMARY_CHARS,
            ),
            status="candidate",
            review_status="unreviewed",
            confidence=event.confidence,
            evidence_strength=event.evidence_strength,
            severity=event.severity,
            generated_by_model=None,
            prompt_version="dry_run_v1",
        )
        event_id_map[event.event_id] = event_id
        events_written += 1
        created_event_ids.append(event_id)
        for item in event.evidence:
            safe = _storage_evidence(item, mode=mode, privacy_level=privacy_level)
            repo.create_evidence(
                event_id=event_id,
                source_type=safe.source_type,
                source_id=safe.source_id,
                source_pointer=safe.source_pointer,
                source_date=safe.date,
                role=safe.role,
                summary=safe.summary,
                excerpt=safe.excerpt,
                confidence=safe.confidence,
                evidence_strength=safe.evidence_strength,
            )
            evidence_written += 1

    for activity in result.post_conflict_outings:
        conflict_event_id = event_id_map.get(activity.conflict_event_id)
        if _post_conflict_activity_exists(
            repo,
            conflict_event_id=conflict_event_id,
            activity=activity,
            mode=mode,
            privacy_level=privacy_level,
        ):
            duplicate_count += 1
            warnings.append(
                "duplicate skipped: "
                f"post_conflict_activity {activity.date} "
                f"{_storage_text(activity.place_label, mode=mode, privacy_level=privacy_level, max_chars=80)}"
            )
            continue
        activity_id = repo.create_post_conflict_activity(
            conflict_event_id=conflict_event_id,
            activity_event_id=None,
            activity_date=activity.date,
            days_after_conflict=activity.days_after_conflict,
            place_label=_storage_text(
                _activity_storage_label(activity, privacy_level=privacy_level),
                mode=mode,
                privacy_level=privacy_level,
                max_chars=80,
            ),
            activity_type=activity.activity_type,
            confidence=activity.confidence,
            evidence_strength=activity.evidence_strength,
        )
        activities_written += 1
        created_activity_ids.append(activity_id)

    return WriteResult(
        events_written=events_written,
        evidence_written=evidence_written,
        post_conflict_activities_written=activities_written,
        duplicates=duplicate_count,
        warnings=warnings,
        created_event_ids=tuple(created_event_ids),
        created_activity_ids=tuple(created_activity_ids),
    )


def assess_write_safety(
    *,
    repo: RelationshipRepository,
    result: DryRunResult,
    profile_id: int,
    date_from: str,
    date_to: str,
    mode: str,
    privacy_level: str,
) -> WriteSafetyReport:
    _validate_privacy_level(privacy_level, mode=mode)
    warnings: list[str] = []
    if not date_from or not date_to:
        warnings.append("date range is required for write safety")
    if profile_id is None:
        warnings.append("profile_id is required for write safety")

    events = _write_events(result)
    duplicate_candidates = 0
    planned_records: list[tuple[str, str]] = []
    planned_evidence_count = 0
    for event in events:
        source_pointer = _event_source_pointer(event)
        if _find_duplicate_event(
            repo,
            profile_id=profile_id,
            event_type=event.event_type,
            event_date=event.date,
            source_pointer=source_pointer,
            event=event,
        ):
            duplicate_candidates += 1
        planned_records.append(
            (
                f"event:{event.event_type}:{event.date}",
                _storage_text(
                    _event_storage_summary(event, privacy_level=privacy_level),
                    mode=mode,
                    privacy_level=privacy_level,
                    max_chars=MAX_STORED_SUMMARY_CHARS,
                ),
            )
        )
        for item in event.evidence:
            safe = _storage_evidence(item, mode=mode, privacy_level=privacy_level)
            planned_evidence_count += 1
            planned_records.extend(
                [
                    (f"evidence:{safe.source_type}:{safe.source_id}:summary", safe.summary),
                    (f"evidence:{safe.source_type}:{safe.source_id}:excerpt", safe.excerpt or ""),
                    (f"evidence:{safe.source_type}:{safe.source_id}:source_pointer", safe.source_pointer or ""),
                ]
            )

    for activity in result.post_conflict_outings:
        planned_records.append(
            (
                f"post_conflict_activity:{activity.date}",
                _storage_text(
                    _activity_storage_label(activity, privacy_level=privacy_level),
                    mode=mode,
                    privacy_level=privacy_level,
                    max_chars=80,
                ),
            )
        )

    raw_issues, forbidden_issues = _scan_text_records(planned_records, mode=mode)
    return WriteSafetyReport(
        candidate_events=len(events),
        candidate_evidence=planned_evidence_count,
        candidate_post_conflict_activities=len(result.post_conflict_outings),
        duplicate_candidates=duplicate_candidates,
        raw_leakage_issues=tuple(raw_issues),
        forbidden_phrase_issues=tuple(forbidden_issues),
        warnings=tuple(warnings),
    )


def scan_created_write_rows(
    *,
    repo: RelationshipRepository,
    write_result: WriteResult,
    mode: str,
) -> WriteSafetyReport:
    records: list[tuple[str, str]] = []
    evidence_count = 0
    for event_id in write_result.created_event_ids:
        event = repo.get_event(event_id)
        if event:
            records.append((f"event:{event_id}:summary", str(event.get("summary") or "")))
        for evidence in repo.list_evidence(event_id=event_id, limit=200):
            evidence_count += 1
            records.extend(
                [
                    (f"evidence:{event_id}:summary", str(evidence.get("summary") or "")),
                    (f"evidence:{event_id}:excerpt", str(evidence.get("excerpt") or "")),
                    (f"evidence:{event_id}:source_pointer", str(evidence.get("source_pointer") or "")),
                ]
            )
    for activity_id in write_result.created_activity_ids:
        activity = repo.get_post_conflict_activity(activity_id)
        if activity:
            records.append((f"post_conflict_activity:{activity_id}:place_label", str(activity.get("place_label") or "")))
    raw_issues, forbidden_issues = _scan_text_records(records, mode=mode)
    return WriteSafetyReport(
        candidate_events=write_result.events_written,
        candidate_evidence=evidence_count,
        candidate_post_conflict_activities=write_result.post_conflict_activities_written,
        duplicate_candidates=write_result.duplicates,
        raw_leakage_issues=tuple(raw_issues),
        forbidden_phrase_issues=tuple(forbidden_issues),
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
    speaker_id = profile.target_line_speaker_source_id if profile and backend != "mock" else None
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
    privacy_level: str,
    profile: ProfileContext | None,
    source_counts: dict[str, int],
    conflicts: list[RelationshipEvent],
    reconciliations: list[RelationshipEvent],
    outings: list[PostConflictActivity],
    emotional_notes: list[NoteEvidence],
    monthly_summaries: list[dict[str, object]],
    reviewed_snapshot: dict[str, Any],
    warnings: list[str],
) -> str:
    counts = conflict_counts(conflicts)
    person_names = _profile_person_names(profile)
    sections = [
        "# Relationship Event Candidates Dry Run",
        "## Dry-run status\n"
        "- relationship DBには候補を書き込んでいません。\n"
        "- 上流データはread-onlyで参照します。\n"
        "- raw LINE全文、raw note全文、正確GPS、顔情報、写真実体は出力しません。\n"
        f"- privacy_level: {privacy_level}",
        f"## 対象期間\n- date_from: {date_from}\n- date_to: {date_to}\n- backend: {backend}\n- mode: {mode}",
        "## 使用profile\n" + _format_profile(profile, mode, privacy_level),
        "## 取得できたsource counts\n" + _format_dict(source_counts),
        "## reviewed relationship DB events\n"
        + _format_reviewed_snapshot(reviewed_snapshot, mode, privacy_level, person_names),
        "## new conflict candidates\n"
        + _format_events([item for item in conflicts if item.event_type == "conflict"], mode, privacy_level, person_names),
        "## new minor misunderstanding candidates\n"
        + _format_events(
            [item for item in conflicts if item.event_type == "minor_misunderstanding"],
            mode,
            privacy_level,
            person_names,
        ),
        "## new reconciliation candidates\n" + _format_events(reconciliations, mode, privacy_level, person_names),
        "## new post-conflict outing candidates\n" + _format_outings(outings, mode, privacy_level, person_names),
        "## new emotional note candidates\n" + _format_adapter_evidence(emotional_notes, mode, privacy_level, person_names),
        "## monthly relationship summary\n" + _format_monthly(monthly_summaries, mode, privacy_level, person_names),
        "## evidence strength 分布\n" + _format_strength_distribution([*conflicts, *reconciliations, *emotional_notes]),
        "## candidate counts\n" + _format_dict({**counts, "reconciliation_candidates": len(reconciliations), "post_conflict_outing_candidates": len(outings), "emotional_note_candidates": len(emotional_notes)}),
        "## warnings\n" + _format_warnings(warnings, mode, privacy_level, person_names),
    ]
    return _display_text("\n\n".join(sections), mode=mode, privacy_level=privacy_level, person_names=person_names)


def _format_profile(profile: ProfileContext | None, mode: str, privacy_level: str) -> str:
    if profile is None:
        return "- profile未設定。関係ラベルやID対応は推定していません。"
    if mode == "public" or privacy_level in {"counts-only", "redacted"}:
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
        f"- line_speaker_group_source_id: {profile.line_speaker_group_source_id or 'unset'}",
        f"- self_person_source_id: {profile.self_person_source_id or 'unset'}",
        f"- self_line_speaker_source_id: {profile.self_line_speaker_source_id or 'unset'}",
        f"- self_line_speaker_group_source_id: {profile.self_line_speaker_group_source_id or 'unset'}",
        "- label_source: user_manual",
    ]
    if mode == "private" and profile.relationship_label:
        lines.append(f"- relationship_label: {profile.relationship_label}")
    return "\n".join(lines)


def _format_events(events: list[RelationshipEvent], mode: str, privacy_level: str, person_names: list[str]) -> str:
    if not events:
        return "- 候補はありません。"
    if privacy_level == "counts-only":
        return f"- count: {len(events)}\n- detail: counts-only privacy levelのため候補本文は省略しています。"
    return "\n".join(
        f"- {event.date}: {_display_text(event.summary, mode=mode, privacy_level=privacy_level, person_names=person_names)} "
        f"(confidence={event.confidence:.2f}, evidence_strength={event.evidence_strength:.2f}, "
        f"source={_source_pointer(event.evidence, mode, privacy_level, person_names)})"
        for event in events
    )


def _format_outings(outings: list[PostConflictActivity], mode: str, privacy_level: str, person_names: list[str]) -> str:
    if not outings:
        return "- 候補はありません。"
    if privacy_level == "counts-only":
        return f"- count: {len(outings)}\n- detail: counts-only privacy levelのため候補本文は省略しています。"
    lines = []
    for outing in outings:
        evidence = [redact_evidence_for_mode(item, mode=mode) for item in outing.evidence]
        place_label = _display_text(outing.place_label, mode=mode, privacy_level=privacy_level, person_names=person_names)
        lines.append(
            f"- {outing.date}: 喧嘩候補の{outing.days_after_conflict}日後の外出候補 "
            f"(place={place_label}, confidence={outing.confidence:.2f}, evidence_strength={outing.evidence_strength:.2f}, "
            f"source={_source_pointer(evidence, mode, privacy_level, person_names)})"
        )
    return "\n".join(lines)


def _format_adapter_evidence(items: list[AdapterEvidence], mode: str, privacy_level: str, person_names: list[str]) -> str:
    if not items:
        return "- 候補はありません。"
    if privacy_level == "counts-only":
        return f"- count: {len(items)}\n- detail: counts-only privacy levelのため候補本文は省略しています。"
    lines = []
    for item in items:
        evidence = redact_evidence_for_mode(item.as_evidence_item(), mode=mode)
        summary = _display_text(evidence.summary, mode=mode, privacy_level=privacy_level, person_names=person_names)
        lines.append(
            f"- {evidence.date}: {summary} "
            f"(confidence={evidence.confidence:.2f}, evidence_strength={evidence.evidence_strength:.2f}, "
            f"source={_display_source_pointer(evidence.source_pointer, mode, privacy_level, person_names)})"
        )
    return "\n".join(lines)


def _format_reviewed_snapshot(
    snapshot: dict[str, Any],
    mode: str,
    privacy_level: str,
    person_names: list[str],
) -> str:
    if not snapshot:
        return "- relationship DBのreview済みeventは確認できませんでした。"
    counts = snapshot.get("counts", {})
    lines = [
        "- review済みeventと新規dry-run candidateは分けて扱います。",
        f"- 人間確認済み件数: {counts.get('human_verified', 0)}",
        f"- AI推定のみ件数: {counts.get('ai_only', 0)}",
        f"- 除外済み件数: {counts.get('excluded', 0)}",
        f"- 要再分析件数: {counts.get('needs_reanalysis', 0)}",
    ]
    examples = snapshot.get("examples", [])
    if not examples:
        return "\n".join(lines)
    if privacy_level == "counts-only":
        lines.append("- detail: counts-only privacy levelのためreview済みevent本文は省略しています。")
        return "\n".join(lines)
    lines.append("- examples:")
    for item in examples[:5]:
        status = item.get("review_status", "unreviewed")
        event_type = item.get("event_type", "other")
        date = item.get("event_date", "unknown")
        summary = _display_text(str(item.get("summary", "")), mode=mode, privacy_level=privacy_level, person_names=person_names)
        lines.append(f"  - {date}: {status}/{event_type} - {summary}")
    return "\n".join(lines)


def _format_monthly(
    monthly_summaries: list[dict[str, object]],
    mode: str,
    privacy_level: str,
    person_names: list[str],
) -> str:
    if not monthly_summaries:
        return "- 月次summaryはありません。"
    if privacy_level == "counts-only":
        return "\n".join(
            "- {month}: conflict_candidates={conflict_candidates}, "
            "minor_misunderstanding_candidates={minor_misunderstanding_candidates}, "
            "reconciliation_candidates={reconciliation_candidates}, "
            "post_conflict_outing_candidates={post_conflict_outing_candidates}, "
            "emotional_note_candidates={emotional_note_candidates}".format(**item)
            for item in monthly_summaries
        )
    return "\n".join(
        f"- {item['month']}: "
        f"{_display_text(str(item['summary']), mode=mode, privacy_level=privacy_level, person_names=person_names)}"
        for item in monthly_summaries
    )


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


def _format_warnings(warnings: list[str], mode: str, privacy_level: str, person_names: list[str]) -> str:
    if not warnings:
        return "- 追加warningはありません。"
    return "\n".join(
        f"- {_display_text(warning, mode=mode, privacy_level=privacy_level, person_names=person_names)}"
        for warning in warnings
    )


def _source_pointer(
    evidence: tuple[EvidenceItem, ...] | list[EvidenceItem],
    mode: str,
    privacy_level: str,
    person_names: list[str],
) -> str:
    if not evidence:
        return "none"
    if privacy_level == "counts-only":
        return "omitted"
    safe = redact_evidence_for_mode(evidence[0], mode=mode)
    return _display_source_pointer(safe.source_pointer, mode, privacy_level, person_names)


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


def _load_reviewed_event_snapshot(
    *,
    settings: Settings,
    profile: ProfileContext | None,
    date_from: str,
    date_to: str,
) -> dict[str, Any]:
    if profile is None:
        return {}
    db_path = Path(settings.paths.relationship_db).expanduser()
    if not db_path.exists():
        return {}
    uri = f"file:{db_path}?mode=ro"
    try:
        with sqlite3.connect(uri, uri=True) as conn:
            conn.row_factory = sqlite3.Row
            table = conn.execute(
                "SELECT name FROM sqlite_master WHERE type = 'table' AND name = 'relationship_events'"
            ).fetchone()
            if table is None:
                return {}
            rows = conn.execute(
                """
                SELECT id, event_type, event_date, summary, status, review_status
                FROM relationship_events
                WHERE profile_id = ?
                  AND event_date >= ?
                  AND event_date <= ?
                ORDER BY
                  CASE review_status
                    WHEN 'verified' THEN 0
                    WHEN 'corrected' THEN 0
                    WHEN 'unreviewed' THEN 1
                    WHEN 'needs_reanalysis' THEN 2
                    ELSE 3
                  END,
                  event_date,
                  id
                LIMIT 200
                """,
                (profile.id, date_from, date_to),
            ).fetchall()
    except sqlite3.Error:
        return {}
    counts = {"human_verified": 0, "ai_only": 0, "excluded": 0, "needs_reanalysis": 0}
    examples: list[dict[str, str]] = []
    for row in rows:
        status = str(row["status"] or "")
        review_status = str(row["review_status"] or "")
        if status != "candidate" or review_status == "rejected":
            counts["excluded"] += 1
        elif review_status in {"verified", "corrected"}:
            counts["human_verified"] += 1
        elif review_status == "needs_reanalysis":
            counts["needs_reanalysis"] += 1
        else:
            counts["ai_only"] += 1
        examples.append(
            {
                "event_type": str(row["event_type"]),
                "event_date": str(row["event_date"]),
                "summary": str(row["summary"]),
                "status": status,
                "review_status": review_status,
            }
        )
    return {"counts": counts, "examples": examples}


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


def _emotional_note_event(note: NoteEvidence) -> RelationshipEvent:
    return RelationshipEvent(
        event_id=f"dry-run-emotional-note-{note.source_id}",
        event_type="emotional_note",
        date=note.date,
        summary=f"感情メモ候補: {note.summary}",
        review_status="candidate",
        confidence=min(note.confidence, 0.62),
        evidence_strength=min(note.evidence_strength, 0.6),
        severity=0,
        evidence=(note.as_evidence_item(),),
        metadata={"dry_run": True, "source_pointer": note.source_pointer},
    )


def _write_events(result: DryRunResult) -> list[RelationshipEvent]:
    return [
        *result.conflict_candidates,
        *result.minor_misunderstanding_candidates,
        *result.reconciliation_candidates,
        *[_emotional_note_event(note) for note in result.emotional_notes],
    ]


def _event_source_pointer(event: RelationshipEvent) -> str:
    pointer = event.metadata.get("source_pointer")
    if pointer:
        return _storage_text(str(pointer), mode="private", max_chars=180)
    if event.evidence:
        return _storage_text(str(event.evidence[0].source_pointer), mode="private", max_chars=180)
    return _storage_text(event.event_id, mode="private", max_chars=180)


def _find_duplicate_event(
    repo: RelationshipRepository,
    *,
    profile_id: int,
    event_type: str,
    event_date: str,
    source_pointer: str,
    event: RelationshipEvent,
) -> int | None:
    incoming_signature = _event_evidence_signature(event.evidence)
    with repo.connect() as conn:
        row = conn.execute(
            """
            SELECT relationship_events.id
            FROM relationship_events
            JOIN relationship_event_evidence
              ON relationship_event_evidence.event_id = relationship_events.id
            WHERE relationship_events.profile_id = ?
              AND relationship_events.event_type = ?
              AND relationship_events.event_date = ?
              AND relationship_event_evidence.source_pointer = ?
            LIMIT 1
            """,
            (profile_id, event_type, event_date, source_pointer),
        ).fetchone()
        if row:
            return int(row[0])
        rows = conn.execute(
            """
            SELECT id
            FROM relationship_events
            WHERE profile_id = ?
              AND event_type = ?
              AND event_date = ?
            """,
            (profile_id, event_type, event_date),
        ).fetchall()
        for candidate in rows:
            evidence_rows = conn.execute(
                """
                SELECT source_type, source_id, source_pointer
                FROM relationship_event_evidence
                WHERE event_id = ?
                """,
                (candidate["id"],),
            ).fetchall()
            existing_signature = _stored_evidence_signature(evidence_rows)
            if incoming_signature and incoming_signature == existing_signature:
                return int(candidate["id"])
    return None


def _post_conflict_activity_exists(
    repo: RelationshipRepository,
    *,
    conflict_event_id: int | None,
    activity: PostConflictActivity,
    mode: str,
    privacy_level: str,
) -> bool:
    conflict_clause = "conflict_event_id = ?" if conflict_event_id is not None else "conflict_event_id IS NULL"
    params: list[object] = []
    if conflict_event_id is not None:
        params.append(conflict_event_id)
    stored_place_label = _storage_text(
        _activity_storage_label(activity, privacy_level=privacy_level),
        mode=mode,
        privacy_level=privacy_level,
        max_chars=80,
    )
    params.extend([activity.date, activity.activity_type, stored_place_label])
    with repo.connect() as conn:
        row = conn.execute(
            f"""
            SELECT id
            FROM post_conflict_activities
            WHERE {conflict_clause}
              AND activity_date = ?
              AND activity_type = ?
              AND COALESCE(place_label, '') = COALESCE(?, '')
            LIMIT 1
            """,
            params,
        ).fetchone()
    return row is not None


def _event_evidence_signature(evidence: tuple[EvidenceItem, ...] | list[EvidenceItem]) -> tuple[str, ...]:
    return tuple(
        sorted(
            f"{item.source_type}:{_storage_text(str(item.source_pointer or item.source_id), mode='private', max_chars=180)}"
            for item in evidence
        )
    )


def _stored_evidence_signature(rows: list[Any]) -> tuple[str, ...]:
    return tuple(
        sorted(
            f"{row['source_type']}:{_storage_text(str(row['source_pointer'] or row['source_id']), mode='private', max_chars=180)}"
            for row in rows
        )
    )


def _storage_evidence(item: EvidenceItem, *, mode: str, privacy_level: str) -> EvidenceItem:
    safe = redact_evidence_for_mode(item, mode=mode)
    excerpt = (
        None
        if mode == "public" or privacy_level in {"counts-only", "redacted"} or safe.excerpt is None
        else _storage_text(
            safe.excerpt,
            mode=mode,
            privacy_level=privacy_level,
            max_chars=MAX_STORED_EXCERPT_CHARS,
        )
    )
    summary = _evidence_storage_summary(safe, privacy_level=privacy_level)
    return EvidenceItem(
        source_type=safe.source_type,
        source_id=_storage_text(safe.source_id, mode=mode, privacy_level=privacy_level, max_chars=120),
        date=safe.date,
        role=safe.role,
        summary=_storage_text(summary, mode=mode, privacy_level=privacy_level, max_chars=MAX_STORED_SUMMARY_CHARS),
        excerpt=excerpt,
        confidence=safe.confidence,
        evidence_strength=safe.evidence_strength,
        sensitivity=safe.sensitivity,
        source_pointer=_storage_text(str(safe.source_pointer), mode=mode, privacy_level=privacy_level, max_chars=180),
        metadata={},
    )


def _storage_text(
    text: str | None,
    *,
    mode: str,
    max_chars: int,
    privacy_level: str = "private",
    person_names: list[str] | None = None,
) -> str:
    if text is None:
        return ""
    safe = _display_text(text, mode=mode, privacy_level=privacy_level, person_names=person_names or [])
    safe = " ".join(safe.split())
    if len(safe) <= max_chars:
        return safe
    return safe[: max_chars - 1].rstrip() + "…"


def _validate_privacy_level(privacy_level: str, *, mode: str) -> None:
    if privacy_level not in PRIVACY_LEVELS:
        choices = ", ".join(PRIVACY_LEVELS)
        raise ValueError(f"privacy_level must be one of: {choices}")
    if mode == "public" and privacy_level == "private":
        raise ValueError("privacy-level private is not allowed in public mode")


def _adapter_mode_for_privacy(*, mode: str, privacy_level: str) -> str:
    if mode == "public" or privacy_level in {"counts-only", "redacted"}:
        return "public"
    return "private"


def _display_text(
    text: str,
    *,
    mode: str,
    privacy_level: str,
    person_names: list[str],
) -> str:
    if mode == "public" or privacy_level in {"counts-only", "redacted"}:
        return redact_public_text(sanitize_answer(text, mode="public", person_names=person_names), person_names=person_names)
    return _redact_always_private_hazards(sanitize_answer(text, mode="private"))


def _display_source_pointer(
    source_pointer: str | None,
    mode: str,
    privacy_level: str,
    person_names: list[str],
) -> str:
    if not source_pointer:
        return "none"
    if privacy_level == "redacted" or mode == "public":
        return "[redacted_source_pointer]"
    if privacy_level == "counts-only":
        return "omitted"
    return _display_text(str(source_pointer), mode=mode, privacy_level=privacy_level, person_names=person_names)


def _redact_always_private_hazards(text: str) -> str:
    safe = LINE_FULL_TEXT_RE.sub("[LINE full text redacted]", text)
    safe = NOTE_FULL_TEXT_RE.sub("[note full text redacted]", safe)
    safe = SOURCE_PATH_RE.sub("[source file path redacted]", safe)
    safe = FACE_CROP_RE.sub("[face data redacted]", safe)
    safe = PHOTO_PATH_RE.sub("[real photo redacted]", safe)
    safe = GPS_RE.sub("[exact GPS redacted]", safe)
    safe = PRIVATE_PATH_RE.sub("[private path redacted]", safe)
    return safe


def _scan_text_records(records: list[tuple[str, str]], *, mode: str) -> tuple[list[str], list[str]]:
    raw_issues: list[str] = []
    forbidden_issues: list[str] = []
    for label, text in records:
        if not text:
            continue
        for code, pattern in _RAW_LEAKAGE_PATTERNS:
            if pattern.search(text):
                raw_issues.append(f"{label}:{code}")
        for violation in detect_answer_safety_violations(text, mode=mode):
            forbidden_issues.append(f"{label}:{violation.code}")
    return sorted(set(raw_issues)), sorted(set(forbidden_issues))


_RAW_LEAKAGE_PATTERNS = (
    ("line_full_text", LINE_FULL_TEXT_RE),
    ("note_full_text", NOTE_FULL_TEXT_RE),
    ("exact_gps", GPS_RE),
    ("face_crop", FACE_CROP_RE),
    ("photo_path", PHOTO_PATH_RE),
    ("private_path", PRIVATE_PATH_RE),
    ("source_path", SOURCE_PATH_RE),
)


def _profile_person_names(profile: ProfileContext | None) -> list[str]:
    if profile is None or not profile.profile_name:
        return []
    return [profile.profile_name]


def _event_storage_summary(event: RelationshipEvent, *, privacy_level: str) -> str:
    if privacy_level == "counts-only":
        return f"{_event_type_label(event.event_type)} ({event.date})"
    return event.summary


def _evidence_storage_summary(item: EvidenceItem, *, privacy_level: str) -> str:
    if privacy_level == "counts-only":
        return f"{item.source_type} evidence ({item.role})"
    return item.summary


def _activity_storage_label(activity: PostConflictActivity, *, privacy_level: str) -> str:
    if privacy_level == "counts-only":
        return "post_conflict_activity_candidate"
    return activity.place_label


def _event_type_label(event_type: str) -> str:
    labels = {
        "conflict": "喧嘩候補",
        "minor_misunderstanding": "軽いすれ違い候補",
        "reconciliation": "仲直り候補",
        "emotional_note": "感情メモ候補",
    }
    return labels.get(event_type, "関係イベント候補")
