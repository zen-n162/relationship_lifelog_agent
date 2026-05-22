from __future__ import annotations

from dataclasses import asdict, dataclass, replace
import json
from pathlib import Path
import sqlite3
from typing import Any

from relationship_lifelog_agent.adapters.upstream_sqlite import UpstreamReadOnlyError, open_readonly_sqlite
from relationship_lifelog_agent.config import Settings, load_config


STATUS_ORDER = {"OK": 0, "WARN": 1, "ERROR": 2}


@dataclass(frozen=True)
class ColumnGroupResult:
    name: str
    candidates: list[str]
    matched: str | None
    required: bool


@dataclass(frozen=True)
class TableMappingResult:
    key: str
    purpose: str
    table_candidates: list[str]
    matched_table: str | None
    table_exists: bool
    row_count: int | None
    columns: list[str]
    column_groups: list[ColumnGroupResult]
    mapping_status: str
    mapping_confidence: float
    warnings: list[str]
    unsupported_fields: list[str]


@dataclass(frozen=True)
class MethodCoverageResult:
    adapter_method: str
    status: str
    mapping_confidence: float
    source_tables: list[str]
    warnings: list[str]


@dataclass(frozen=True)
class DatabaseInspectionResult:
    app_name: str
    status: str
    configured: bool
    connection_status: str
    table_count: int
    table_mappings: list[TableMappingResult]
    method_coverage: list[MethodCoverageResult]
    warnings: list[str]


@dataclass(frozen=True)
class UpstreamInspectionReport:
    status: str
    backend: str
    databases: list[DatabaseInspectionResult]
    warnings: list[str]

    @property
    def has_errors(self) -> bool:
        return self.status == "ERROR"


@dataclass(frozen=True)
class ColumnGroupSpec:
    name: str
    candidates: tuple[str, ...]
    required: bool = True


@dataclass(frozen=True)
class TableSpec:
    key: str
    purpose: str
    candidates: tuple[str, ...]
    column_groups: tuple[ColumnGroupSpec, ...]


PERSONAL_TABLE_SPECS = (
    TableSpec(
        key="line_messages",
        purpose="LINE messages",
        candidates=("line_messages", "line_message", "messages", "line_logs"),
        column_groups=(
            ColumnGroupSpec("id", ("id", "message_id")),
            ColumnGroupSpec("date", ("sent_at", "timestamp", "created_at", "date")),
            ColumnGroupSpec("text", ("text", "message_text", "body", "content"), required=False),
            ColumnGroupSpec("speaker", ("sender", "speaker_id", "sender_id"), required=False),
            ColumnGroupSpec("conversation", ("chat_id", "conversation_id", "room_id"), required=False),
            ColumnGroupSpec("message_type", ("message_type", "type"), required=False),
        ),
    ),
    TableSpec(
        key="media_items",
        purpose="media/photos",
        candidates=("media_items", "photos", "media", "photo_items"),
        column_groups=(
            ColumnGroupSpec("id", ("id", "media_id")),
            ColumnGroupSpec("date", ("captured_at", "fallback_captured_at", "created_at", "date")),
            ColumnGroupSpec("caption", ("caption", "vlm_caption", "description"), required=False),
            ColumnGroupSpec("ocr", ("ocr_text", "redacted_ocr_text"), required=False),
            ColumnGroupSpec("media_type", ("media_type", "type"), required=False),
            ColumnGroupSpec("place_id", ("place_id",), required=False),
        ),
    ),
    TableSpec(
        key="events",
        purpose="events",
        candidates=("events", "timeline_events", "event_timeline"),
        column_groups=(
            ColumnGroupSpec("id", ("id", "event_id")),
            ColumnGroupSpec("date", ("date", "event_date", "start_date", "created_at")),
            ColumnGroupSpec("title", ("title", "name"), required=False),
            ColumnGroupSpec("summary", ("summary", "description"), required=False),
            ColumnGroupSpec("event_type", ("event_type", "type"), required=False),
            ColumnGroupSpec("confidence", ("confidence", "score"), required=False),
        ),
    ),
    TableSpec(
        key="places",
        purpose="places",
        candidates=("places", "place_visits", "location_clusters"),
        column_groups=(
            ColumnGroupSpec("id", ("id", "place_id")),
            ColumnGroupSpec("label", ("place_label", "label", "name"), required=False),
            ColumnGroupSpec("area", ("area_label", "area", "city"), required=False),
        ),
    ),
    TableSpec(
        key="people",
        purpose="people/person links",
        candidates=("people", "persons", "person_links", "media_people", "event_people"),
        column_groups=(
            ColumnGroupSpec("id", ("id", "person_id")),
            ColumnGroupSpec("source_id", ("source_id", "person_source_id", "face_cluster_id"), required=False),
            ColumnGroupSpec("review_status", ("review_status", "status"), required=False),
        ),
    ),
    TableSpec(
        key="captions_ocr",
        purpose="captions/OCR",
        candidates=("media_captions", "ocr_results", "captions", "media_items"),
        column_groups=(
            ColumnGroupSpec("id", ("id", "media_id", "caption_id", "ocr_id")),
            ColumnGroupSpec("text", ("caption", "vlm_caption", "ocr_text", "text", "content"), required=False),
            ColumnGroupSpec("media_id", ("media_id",), required=False),
        ),
    ),
)

NOTES_TABLE_SPECS = (
    TableSpec(
        key="notes",
        purpose="notes",
        candidates=("notes", "apple_notes", "note_records"),
        column_groups=(
            ColumnGroupSpec("id", ("id", "note_id")),
            ColumnGroupSpec("date", ("note_date", "date", "created_at", "updated_at"), required=False),
            ColumnGroupSpec("title", ("title", "generated_title"), required=False),
            ColumnGroupSpec("summary", ("summary", "generated_summary"), required=False),
            ColumnGroupSpec("body", ("body", "text", "content"), required=False),
        ),
    ),
    TableSpec(
        key="note_chunks",
        purpose="note chunks",
        candidates=("note_chunks", "chunks", "note_embeddings"),
        column_groups=(
            ColumnGroupSpec("id", ("id", "chunk_id")),
            ColumnGroupSpec("note_id", ("note_id",), required=False),
            ColumnGroupSpec("text", ("text", "content", "chunk_text"), required=False),
        ),
    ),
    TableSpec(
        key="thoughts",
        purpose="thoughts",
        candidates=("thoughts", "thought_records", "reflections"),
        column_groups=(
            ColumnGroupSpec("id", ("id", "thought_id")),
            ColumnGroupSpec("date", ("date", "date_label", "created_at"), required=False),
            ColumnGroupSpec("summary", ("summary", "text"), required=False),
            ColumnGroupSpec("thought_type", ("thought_type", "type"), required=False),
            ColumnGroupSpec("themes", ("themes", "emotion_labels", "tags"), required=False),
            ColumnGroupSpec("confidence", ("confidence", "score"), required=False),
        ),
    ),
    TableSpec(
        key="events",
        purpose="note events",
        candidates=("events", "note_events", "timeline_events"),
        column_groups=(
            ColumnGroupSpec("id", ("id", "event_id")),
            ColumnGroupSpec("date", ("date", "event_date", "start_date", "created_at"), required=False),
            ColumnGroupSpec("summary", ("summary", "description", "title"), required=False),
        ),
    ),
    TableSpec(
        key="monthly_reflections",
        purpose="monthly reflections",
        candidates=("monthly_reflections", "monthly_reviews", "month_reflections"),
        column_groups=(
            ColumnGroupSpec("month", ("month", "target_month")),
            ColumnGroupSpec("summary", ("summary", "summary_json", "reflection", "body"), required=False),
            ColumnGroupSpec("confidence", ("confidence", "importance", "score"), required=False),
        ),
    ),
    TableSpec(
        key="suggestions",
        purpose="suggestions",
        candidates=("suggestions", "note_suggestions", "reflection_suggestions"),
        column_groups=(
            ColumnGroupSpec("id", ("id", "suggestion_id")),
            ColumnGroupSpec("date", ("date", "created_at"), required=False),
            ColumnGroupSpec("summary", ("summary", "text", "content"), required=False),
        ),
    ),
)


def run_upstream_inspection(
    *,
    config_path: str | Path | None = None,
    backend: str = "mock",
    settings: Settings | None = None,
) -> UpstreamInspectionReport:
    resolved = settings or load_config(config_path)
    resolved = replace(resolved, adapter=replace(resolved.adapter, backend=backend))
    if backend == "mock":
        databases = [
            _mock_database("personal_lifelog_rag", PERSONAL_TABLE_SPECS),
            _mock_database("notes_lifelog_rag", NOTES_TABLE_SPECS),
        ]
        return UpstreamInspectionReport(status="OK", backend=backend, databases=databases, warnings=[])
    if backend != "upstream_readonly":
        return UpstreamInspectionReport(
            status="ERROR",
            backend=backend,
            databases=[],
            warnings=[f"unsupported backend: {backend}"],
        )
    databases = [
        _inspect_database(
            app_name="personal_lifelog_rag",
            db_path=resolved.paths.personal_lifelog_db,
            table_specs=PERSONAL_TABLE_SPECS,
            method_specs=_personal_method_specs(),
        ),
        _inspect_database(
            app_name="notes_lifelog_rag",
            db_path=resolved.paths.notes_lifelog_db,
            table_specs=NOTES_TABLE_SPECS,
            method_specs=_notes_method_specs(),
        ),
    ]
    warnings = [warning for database in databases for warning in database.warnings]
    return UpstreamInspectionReport(status=_overall_status([database.status for database in databases]), backend=backend, databases=databases, warnings=warnings)


def render_inspection_json(report: UpstreamInspectionReport) -> str:
    return json.dumps(asdict(report), ensure_ascii=False, indent=2)


def render_inspection_markdown(report: UpstreamInspectionReport) -> str:
    lines = [
        "# Upstream Schema Inspection",
        "",
        f"- status: {report.status}",
        f"- backend: {report.backend}",
        "- privacy: output is schema-only, counts-only, and private paths are redacted",
        "",
    ]
    for database in report.databases:
        lines.extend(
            [
                f"## {database.app_name}",
                "",
                f"- status: {database.status}",
                f"- configured: {str(database.configured).lower()}",
                f"- connection_status: {database.connection_status}",
                f"- table_count: {database.table_count}",
                "",
                "### Adapter Coverage",
                "",
                "| method | status | confidence | source_tables | warnings |",
                "| --- | --- | ---: | --- | --- |",
            ]
        )
        for item in database.method_coverage:
            lines.append(
                f"| {item.adapter_method} | {item.status} | {item.mapping_confidence:.2f} | "
                f"{', '.join(item.source_tables) or '-'} | {_join(item.warnings)} |"
            )
        lines.extend(["", "### Table Mapping", ""])
        for table in database.table_mappings:
            lines.extend(
                [
                    f"#### {table.key}",
                    "",
                    f"- purpose: {table.purpose}",
                    f"- table_exists: {str(table.table_exists).lower()}",
                    f"- matched_table: {table.matched_table or '-'}",
                    f"- row_count: {table.row_count if table.row_count is not None else '-'}",
                    f"- mapping_status: {table.mapping_status}",
                    f"- mapping_confidence: {table.mapping_confidence:.2f}",
                    f"- columns: {', '.join(table.columns) if table.columns else '-'}",
                    f"- unsupported_fields: {', '.join(table.unsupported_fields) if table.unsupported_fields else '-'}",
                    f"- warnings: {_join(table.warnings)}",
                    "",
                    "| column_group | required | matched | candidates |",
                    "| --- | --- | --- | --- |",
                ]
            )
            for group in table.column_groups:
                lines.append(
                    f"| {group.name} | {str(group.required).lower()} | {group.matched or '-'} | {', '.join(group.candidates)} |"
                )
            lines.append("")
    if report.warnings:
        lines.extend(["## Warnings", ""])
        lines.extend(f"- {warning}" for warning in report.warnings)
    return "\n".join(lines).rstrip() + "\n"


def write_inspection_report(report: UpstreamInspectionReport, output_path: str | Path, *, output_format: str) -> Path:
    path = Path(output_path).expanduser()
    path.parent.mkdir(parents=True, exist_ok=True)
    rendered = render_inspection_json(report) if output_format == "json" else render_inspection_markdown(report)
    path.write_text(rendered, encoding="utf-8")
    return path


def _inspect_database(
    *,
    app_name: str,
    db_path: str | None,
    table_specs: tuple[TableSpec, ...],
    method_specs: dict[str, list[str]],
) -> DatabaseInspectionResult:
    if not db_path:
        mappings = [_missing_mapping(spec, "upstream DB path is not configured") for spec in table_specs]
        methods = [_method_coverage(method, keys, mappings) for method, keys in method_specs.items()]
        warning = f"{app_name} DB path is not configured"
        return DatabaseInspectionResult(
            app_name=app_name,
            status="WARN",
            configured=False,
            connection_status="not_configured",
            table_count=0,
            table_mappings=mappings,
            method_coverage=methods,
            warnings=[warning],
        )
    try:
        conn = open_readonly_sqlite(db_path)
    except UpstreamReadOnlyError:
        mappings = [_missing_mapping(spec, "read-only connection failed") for spec in table_specs]
        methods = [_method_coverage(method, keys, mappings) for method, keys in method_specs.items()]
        return DatabaseInspectionResult(
            app_name=app_name,
            status="ERROR",
            configured=True,
            connection_status="read_only_failed",
            table_count=0,
            table_mappings=mappings,
            method_coverage=methods,
            warnings=[f"{app_name} DB could not be opened read-only"],
        )
    except sqlite3.Error:
        mappings = [_missing_mapping(spec, "SQLite connection failed") for spec in table_specs]
        methods = [_method_coverage(method, keys, mappings) for method, keys in method_specs.items()]
        return DatabaseInspectionResult(
            app_name=app_name,
            status="ERROR",
            configured=True,
            connection_status="sqlite_error",
            table_count=0,
            table_mappings=mappings,
            method_coverage=methods,
            warnings=[f"{app_name} SQLite read-only inspection failed"],
        )
    try:
        available = _table_columns_by_name(conn)
        mappings = [_inspect_table(conn, spec, available) for spec in table_specs]
    finally:
        conn.close()
    methods = [_method_coverage(method, keys, mappings) for method, keys in method_specs.items()]
    status = _overall_status([_status_for_mapping(item.mapping_status) for item in mappings])
    warnings = [warning for mapping in mappings for warning in mapping.warnings]
    return DatabaseInspectionResult(
        app_name=app_name,
        status=status,
        configured=True,
        connection_status="read_only_ok",
        table_count=len(available),
        table_mappings=mappings,
        method_coverage=methods,
        warnings=warnings,
    )


def _inspect_table(conn: sqlite3.Connection, spec: TableSpec, available: dict[str, list[str]]) -> TableMappingResult:
    matched_table = next((candidate for candidate in spec.candidates if candidate in available), None)
    if matched_table is None:
        return _missing_mapping(spec, "candidate table was not found")
    columns = sorted(available[matched_table])
    groups = [_column_group_result(group, columns) for group in spec.column_groups]
    missing_required = [group.name for group in groups if group.required and not group.matched]
    unsupported = [group.name for group in groups if not group.required and not group.matched]
    row_count = _safe_count(conn, matched_table)
    if missing_required:
        status = "partial"
        warnings = [f"missing required column group: {name}" for name in missing_required]
    else:
        status = "supported"
        warnings = []
    confidence = _mapping_confidence(groups, table_exists=True)
    return TableMappingResult(
        key=spec.key,
        purpose=spec.purpose,
        table_candidates=list(spec.candidates),
        matched_table=matched_table,
        table_exists=True,
        row_count=row_count,
        columns=columns,
        column_groups=groups,
        mapping_status=status,
        mapping_confidence=confidence,
        warnings=warnings,
        unsupported_fields=unsupported,
    )


def _missing_mapping(spec: TableSpec, warning: str) -> TableMappingResult:
    groups = [ColumnGroupResult(group.name, list(group.candidates), None, group.required) for group in spec.column_groups]
    return TableMappingResult(
        key=spec.key,
        purpose=spec.purpose,
        table_candidates=list(spec.candidates),
        matched_table=None,
        table_exists=False,
        row_count=None,
        columns=[],
        column_groups=groups,
        mapping_status="unsupported",
        mapping_confidence=0.0,
        warnings=[warning],
        unsupported_fields=[group.name for group in groups],
    )


def _mock_database(app_name: str, specs: tuple[TableSpec, ...]) -> DatabaseInspectionResult:
    mappings: list[TableMappingResult] = []
    for spec in specs:
        groups = [ColumnGroupResult(group.name, list(group.candidates), group.candidates[0], group.required) for group in spec.column_groups]
        mappings.append(
            TableMappingResult(
                key=spec.key,
                purpose=spec.purpose,
                table_candidates=list(spec.candidates),
                matched_table=f"mock_{spec.key}",
                table_exists=True,
                row_count=0,
                columns=[group.matched for group in groups if group.matched],
                column_groups=groups,
                mapping_status="supported",
                mapping_confidence=1.0,
                warnings=[],
                unsupported_fields=[],
            )
        )
    method_specs = _personal_method_specs() if app_name == "personal_lifelog_rag" else _notes_method_specs()
    methods = [_method_coverage(method, keys, mappings) for method, keys in method_specs.items()]
    return DatabaseInspectionResult(
        app_name=app_name,
        status="OK",
        configured=True,
        connection_status="mock",
        table_count=len(mappings),
        table_mappings=mappings,
        method_coverage=methods,
        warnings=[],
    )


def _column_group_result(group: ColumnGroupSpec, columns: list[str]) -> ColumnGroupResult:
    column_set = set(columns)
    matched = next((candidate for candidate in group.candidates if candidate in column_set), None)
    return ColumnGroupResult(group.name, list(group.candidates), matched, group.required)


def _mapping_confidence(groups: list[ColumnGroupResult], *, table_exists: bool) -> float:
    if not table_exists:
        return 0.0
    if not groups:
        return 0.5
    required = [group for group in groups if group.required]
    optional = [group for group in groups if not group.required]
    required_score = sum(1 for group in required if group.matched) / max(1, len(required))
    optional_score = sum(1 for group in optional if group.matched) / max(1, len(optional))
    return round((required_score * 0.75) + (optional_score * 0.25), 2)


def _method_coverage(method: str, table_keys: list[str], mappings: list[TableMappingResult]) -> MethodCoverageResult:
    mapping_by_key = {mapping.key: mapping for mapping in mappings}
    selected = [mapping_by_key[key] for key in table_keys if key in mapping_by_key]
    if not selected:
        return MethodCoverageResult(method, "unsupported", 0.0, [], ["no mapping spec"])
    if all(item.mapping_status == "supported" for item in selected):
        status = "supported"
    elif any(item.mapping_status in {"supported", "partial"} for item in selected):
        status = "partial"
    else:
        status = "unsupported"
    confidence = round(sum(item.mapping_confidence for item in selected) / len(selected), 2)
    source_tables = [item.matched_table or item.key for item in selected]
    warnings = [warning for item in selected for warning in item.warnings]
    return MethodCoverageResult(method, status, confidence, source_tables, warnings)


def _personal_method_specs() -> dict[str, list[str]]:
    return {
        "search_line": ["line_messages"],
        "search_media": ["media_items"],
        "search_events": ["events"],
        "get_day_summary": ["line_messages", "media_items", "events"],
        "get_month_summary": ["line_messages", "media_items", "events"],
    }


def _notes_method_specs() -> dict[str, list[str]]:
    return {
        "search_notes": ["notes"],
        "search_thoughts": ["thoughts"],
        "get_monthly_reflection": ["monthly_reflections"],
    }


def _table_columns_by_name(conn: sqlite3.Connection) -> dict[str, list[str]]:
    rows = conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%' ORDER BY name"
    ).fetchall()
    tables: dict[str, list[str]] = {}
    for row in rows:
        table_name = str(row[0])
        columns = conn.execute(f"PRAGMA table_info({_quote_identifier(table_name)})").fetchall()
        tables[table_name] = [str(column[1]) for column in columns]
    return tables


def _safe_count(conn: sqlite3.Connection, table_name: str) -> int:
    try:
        row = conn.execute(f"SELECT COUNT(*) FROM {_quote_identifier(table_name)}").fetchone()
    except sqlite3.Error:
        return 0
    return int(row[0]) if row else 0


def _status_for_mapping(mapping_status: str) -> str:
    if mapping_status == "supported":
        return "OK"
    if mapping_status == "partial":
        return "WARN"
    return "WARN"


def _overall_status(statuses: list[str]) -> str:
    value = max((STATUS_ORDER.get(status, 2) for status in statuses), default=0)
    for status, order in STATUS_ORDER.items():
        if order == value:
            return status
    return "ERROR"


def _quote_identifier(value: str) -> str:
    return '"' + value.replace('"', '""') + '"'


def _join(values: list[str]) -> str:
    return "<br>".join(values) if values else "-"
