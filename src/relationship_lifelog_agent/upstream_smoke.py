from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from pathlib import Path
import json
import sqlite3
from urllib.parse import quote

from relationship_lifelog_agent.agent.memory import build_memory
from relationship_lifelog_agent.config import Settings, load_config


@dataclass(frozen=True)
class SmokeProfileStatus:
    requested_profile_id: int | None
    configured: bool
    has_person_source_id: bool
    has_line_speaker_source_id: bool
    label_source: str | None


@dataclass(frozen=True)
class SmokeReport:
    status: str
    adapter_backend: str
    date_from: str
    date_to: str
    profile: SmokeProfileStatus
    source_counts: dict[str, int]
    coverage: dict[str, str]
    monthly_reflection_available: bool
    warnings: list[str]
    unsupported_methods: list[str]
    redaction_status: str
    write_count: int
    next_recommended_action: str
    raw_data_included: bool = False

    @property
    def has_errors(self) -> bool:
        return self.status == "ERROR"


def run_upstream_smoke(
    *,
    date_from: str,
    date_to: str,
    profile_id: int | None = None,
    backend: str = "mock",
    config_path: str | Path | None = None,
    settings: Settings | None = None,
) -> SmokeReport:
    resolved = settings or load_config(config_path)
    resolved = replace(resolved, adapter=replace(resolved.adapter, backend=backend))
    memory = build_memory(resolved)
    profile = _read_profile_status(resolved.paths.relationship_db, profile_id)
    month = _month_from_range(date_from, date_to)
    warnings: list[str] = []

    line = _safe_call(
        lambda: _personal(memory).search_line(
            "喧嘩 すれ違い 謝罪 予定",
            date_from=date_from,
            date_to=date_to,
            speaker_id=None,
            mode="public",
        ),
        warnings,
        "search_line",
    )
    media = _safe_call(
        lambda: _personal(memory).search_media(
            "外出 場所 カフェ 食事 喧嘩後",
            date_from=date_from,
            date_to=date_to,
            person_id=None,
            mode="public",
        ),
        warnings,
        "search_media",
    )
    events = _safe_call(
        lambda: _personal(memory).search_events(date_from=date_from, date_to=date_to, mode="public"),
        warnings,
        "search_events",
    )
    notes = _safe_call(
        lambda: _notes(memory).search_notes("反省 不安 感謝 関係", date_from=date_from, date_to=date_to, mode="public"),
        warnings,
        "search_notes",
    )
    thoughts = _safe_call(
        lambda: _notes(memory).search_thoughts("反省 不安 感謝 関係", date_from=date_from, date_to=date_to, mode="public"),
        warnings,
        "search_thoughts",
    )
    monthly = _safe_call(lambda: [_notes(memory).get_monthly_reflection(month, mode="public")], warnings, "get_monthly_reflection")
    adapter_warnings = getattr(memory, "warnings", None)
    if callable(adapter_warnings):
        warnings.extend(adapter_warnings())

    source_counts = {
        "line_evidence": len(line),
        "media_evidence": len(media),
        "event_evidence": len(events),
        "note_evidence": len(notes),
        "thought_evidence": len(thoughts),
    }
    coverage = {
        "search_line": _coverage_for_count(len(line), warnings, "search_line"),
        "search_media": _coverage_for_count(len(media), warnings, "search_media"),
        "search_events": _coverage_for_count(len(events), warnings, "search_events"),
        "search_notes": _coverage_for_count(len(notes), warnings, "search_notes"),
        "search_thoughts": _coverage_for_count(len(thoughts), warnings, "search_thoughts"),
        "get_monthly_reflection": _coverage_for_monthly(monthly, warnings),
    }
    unsupported = sorted(method for method, status in coverage.items() if status == "unsupported")
    deduped_warnings = list(dict.fromkeys(warnings))
    status = "WARN" if deduped_warnings or unsupported or not profile.configured else "OK"
    return SmokeReport(
        status=status,
        adapter_backend=backend,
        date_from=date_from,
        date_to=date_to,
        profile=profile,
        source_counts=source_counts,
        coverage=coverage,
        monthly_reflection_available=_monthly_available(monthly),
        warnings=deduped_warnings,
        unsupported_methods=unsupported,
        redaction_status="counts_only_redacted",
        write_count=0,
        next_recommended_action=_next_action(status, profile, unsupported, source_counts),
    )


def render_smoke_markdown(report: SmokeReport) -> str:
    lines = [
        "# Upstream Read-only Smoke Report",
        "",
        f"- status: {report.status}",
        f"- adapter_backend: {report.adapter_backend}",
        f"- date_from: {report.date_from}",
        f"- date_to: {report.date_to}",
        f"- profile_configured: {str(report.profile.configured).lower()}",
        f"- requested_profile_id: {report.profile.requested_profile_id if report.profile.requested_profile_id is not None else '-'}",
        f"- has_person_source_id: {str(report.profile.has_person_source_id).lower()}",
        f"- has_line_speaker_source_id: {str(report.profile.has_line_speaker_source_id).lower()}",
        f"- label_source: {report.profile.label_source or '-'}",
        f"- redaction_status: {report.redaction_status}",
        f"- write_count: {report.write_count}",
        f"- raw_data_included: {str(report.raw_data_included).lower()}",
        "",
        "Raw LINE text, raw note text, exact GPS, face data, photo paths, and private paths are not included.",
        "",
        "## Source Counts",
        "",
        "| source | count |",
        "| --- | ---: |",
    ]
    for key, value in report.source_counts.items():
        lines.append(f"| {key} | {value} |")
    lines.extend(
        [
            "",
            "## Coverage",
            "",
            "| method | status |",
            "| --- | --- |",
        ]
    )
    for key, value in report.coverage.items():
        lines.append(f"| {key} | {value} |")
    lines.extend(
        [
            "",
            f"- monthly_reflection_available: {str(report.monthly_reflection_available).lower()}",
            f"- unsupported_methods: {', '.join(report.unsupported_methods) if report.unsupported_methods else '-'}",
            "",
            "## Warnings",
            "",
        ]
    )
    lines.extend(f"- {warning}" for warning in report.warnings) if report.warnings else lines.append("- none")
    lines.extend(["", "## Next Recommended Action", "", report.next_recommended_action])
    return "\n".join(lines).rstrip() + "\n"


def render_smoke_json(report: SmokeReport) -> str:
    return json.dumps(asdict(report), ensure_ascii=False, indent=2)


def write_smoke_report(report: SmokeReport, output_path: str | Path, *, output_format: str = "markdown") -> Path:
    path = Path(output_path).expanduser()
    path.parent.mkdir(parents=True, exist_ok=True)
    rendered = render_smoke_json(report) if output_format == "json" else render_smoke_markdown(report)
    path.write_text(rendered, encoding="utf-8")
    return path


def _read_profile_status(db_path: str | None, profile_id: int | None) -> SmokeProfileStatus:
    if profile_id is None:
        return SmokeProfileStatus(None, False, False, False, None)
    if not db_path:
        return SmokeProfileStatus(profile_id, False, False, False, None)
    path = Path(db_path).expanduser()
    if not path.exists():
        return SmokeProfileStatus(profile_id, False, False, False, None)
    try:
        conn = sqlite3.connect(f"file:{quote(str(path), safe='/:')}?mode=ro", uri=True)
        conn.row_factory = sqlite3.Row
        row = conn.execute(
            """
            SELECT person_source_id, line_speaker_source_id, label_source
            FROM relationship_profiles
            WHERE id = ?
            """,
            (profile_id,),
        ).fetchone()
    except sqlite3.Error:
        return SmokeProfileStatus(profile_id, False, False, False, None)
    finally:
        try:
            conn.close()
        except UnboundLocalError:
            pass
    if row is None:
        return SmokeProfileStatus(profile_id, False, False, False, None)
    return SmokeProfileStatus(
        profile_id,
        True,
        bool(row["person_source_id"]),
        bool(row["line_speaker_source_id"]),
        str(row["label_source"]) if row["label_source"] else None,
    )


def _personal(memory: object) -> object:
    return getattr(memory, "personal", memory)


def _notes(memory: object) -> object:
    return getattr(memory, "notes", memory)


def _safe_call(factory, warnings: list[str], method_name: str) -> list[object]:
    try:
        result = factory()
    except Exception as exc:  # pragma: no cover - defensive for upstream schema drift
        warnings.append(f"{method_name} failed safely: {exc.__class__.__name__}")
        return []
    return list(result or [])


def _coverage_for_count(count: int, warnings: list[str], method_name: str) -> str:
    if any(method_name in warning for warning in warnings):
        return "unsupported"
    return "supported" if count > 0 else "partial"


def _coverage_for_monthly(items: list[object], warnings: list[str]) -> str:
    if any("get_monthly_reflection" in warning for warning in warnings):
        return "unsupported"
    return "supported" if _monthly_available(items) else "partial"


def _monthly_available(items: list[object]) -> bool:
    if not items:
        return False
    item = items[0]
    confidence = float(getattr(item, "confidence", 0.0) or 0.0)
    evidence_strength = float(getattr(item, "evidence_strength", 0.0) or 0.0)
    return confidence > 0.0 or evidence_strength > 0.0


def _month_from_range(date_from: str, date_to: str) -> str:
    del date_to
    return date_from[:7]


def _next_action(
    status: str,
    profile: SmokeProfileStatus,
    unsupported: list[str],
    source_counts: dict[str, int],
) -> str:
    if not profile.configured:
        return "Create or select a manual relationship profile before using relationship-specific answers."
    if unsupported:
        return "Run upstream inspect and adjust the read-only adapter mapping for unsupported methods."
    if not any(source_counts.values()):
        return "Check date range and upstream DB paths; the smoke found no evidence rows in the requested range."
    if status == "WARN":
        return "Review warnings, then rerun smoke before enabling real-data QA."
    return "Smoke passed. Next, run a private dry-run analysis without --write."
