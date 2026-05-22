from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
import sqlite3
from typing import Any

from relationship_lifelog_agent.adapters.upstream_sqlite import (
    UpstreamReadOnlyError,
    first_existing_column,
    normalize_date,
    open_readonly_sqlite,
    table_columns,
    table_exists,
)
from relationship_lifelog_agent.config import Settings, load_config


IDENTITY_KINDS = ("people", "line-speakers", "all")
IDENTITY_PRIVACY_LEVELS = ("redacted", "private")


@dataclass(frozen=True)
class PeopleIdentity:
    identity_label: str
    source_pointer: str
    person_source_id: str
    verification_status: str
    media_count: int
    first_seen_date: str | None
    last_seen_date: str | None
    manual_label: str | None = None
    warning: str | None = None


@dataclass(frozen=True)
class LineSpeakerIdentity:
    identity_label: str
    source_pointer: str
    line_speaker_source_id: str
    message_count: int
    first_message_date: str | None
    last_message_date: str | None
    verification_status: str
    display_name: str | None = None
    warning: str | None = None


@dataclass(frozen=True)
class UpstreamIdentityReport:
    status: str
    backend: str
    kind: str
    privacy_level: str
    people: list[PeopleIdentity]
    line_speakers: list[LineSpeakerIdentity]
    warnings: list[str]

    @property
    def has_errors(self) -> bool:
        return self.status == "ERROR"


def run_upstream_identities(
    *,
    config_path: str | Path | None = None,
    backend: str = "mock",
    kind: str = "all",
    privacy_level: str = "redacted",
    settings: Settings | None = None,
) -> UpstreamIdentityReport:
    if kind not in IDENTITY_KINDS:
        return _error_report(backend, kind, privacy_level, f"unsupported identity kind: {kind}")
    if privacy_level not in IDENTITY_PRIVACY_LEVELS:
        return _error_report(backend, kind, privacy_level, f"unsupported privacy level: {privacy_level}")
    if backend == "mock":
        return _mock_report(kind, privacy_level)
    if backend != "upstream_readonly":
        return _error_report(backend, kind, privacy_level, f"unsupported backend: {backend}")

    resolved = settings or load_config(config_path)
    warnings: list[str] = []
    people: list[PeopleIdentity] = []
    line_speakers: list[LineSpeakerIdentity] = []

    if kind in {"people", "all"}:
        people, people_warnings = _load_people_identities(
            resolved.paths.personal_lifelog_db,
            privacy_level=privacy_level,
        )
        warnings.extend(people_warnings)
    if kind in {"line-speakers", "all"}:
        line_speakers, speaker_warnings = _load_line_speaker_identities(
            resolved.paths.personal_lifelog_db,
            privacy_level=privacy_level,
        )
        warnings.extend(speaker_warnings)

    status = "WARN" if warnings else "OK"
    return UpstreamIdentityReport(
        status=status,
        backend=backend,
        kind=kind,
        privacy_level=privacy_level,
        people=people,
        line_speakers=line_speakers,
        warnings=list(dict.fromkeys(warnings)),
    )


def render_identities_json(report: UpstreamIdentityReport) -> str:
    return json.dumps(asdict(report), ensure_ascii=False, indent=2)


def render_identities_markdown(report: UpstreamIdentityReport) -> str:
    lines = [
        "# Upstream Identity Inventory",
        "",
        f"- status: {report.status}",
        f"- backend: {report.backend}",
        f"- kind: {report.kind}",
        f"- privacy_level: {report.privacy_level}",
        "- raw_data_included: false",
        "",
        "LINE本文、メモ本文、正確GPS、顔情報、顔crop、写真パス、private path、AI推定の関係ラベルは含みません。",
        "",
    ]
    if report.kind in {"people", "all"}:
        lines.extend(_people_table(report.people, private=report.privacy_level == "private"))
    if report.kind in {"line-speakers", "all"}:
        lines.extend(_line_speaker_table(report.line_speakers, private=report.privacy_level == "private"))
    if report.warnings:
        lines.extend(["## Warnings", ""])
        lines.extend(f"- {warning}" for warning in report.warnings)
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def write_identities_report(report: UpstreamIdentityReport, output_path: str | Path, *, output_format: str) -> Path:
    path = Path(output_path).expanduser()
    path.parent.mkdir(parents=True, exist_ok=True)
    rendered = render_identities_json(report) if output_format == "json" else render_identities_markdown(report)
    path.write_text(rendered, encoding="utf-8")
    return path


def _load_people_identities(
    db_path: str | None,
    *,
    privacy_level: str,
) -> tuple[list[PeopleIdentity], list[str]]:
    if not db_path:
        return [], ["personal_lifelog_db is not configured"]
    try:
        conn = open_readonly_sqlite(db_path)
    except (UpstreamReadOnlyError, sqlite3.Error):
        return [], ["personal_lifelog_db could not be opened read-only"]
    try:
        if not table_exists(conn, "persons"):
            return [], ["persons table was not found"]
        people = _query_people(conn, privacy_level=privacy_level)
    finally:
        conn.close()
    if not people:
        return [], ["no upstream people identities found"]
    return people, []


def _query_people(conn: sqlite3.Connection, *, privacy_level: str) -> list[PeopleIdentity]:
    columns = table_columns(conn, "persons")
    id_col = first_existing_column(columns, ("id", "person_id"))
    if not id_col:
        return []
    select_columns = [id_col]
    display_col = first_existing_column(columns, ("display_name", "public_name", "name"))
    manual_verified_col = first_existing_column(columns, ("manual_verified", "verified_by_user", "review_status", "status"))
    hidden_col = first_existing_column(columns, ("hidden",))
    deleted_col = first_existing_column(columns, ("deleted_at",))
    for column in (display_col, manual_verified_col, hidden_col, deleted_col):
        if column and column not in select_columns:
            select_columns.append(column)

    sql = f"SELECT {', '.join(_q(column) for column in select_columns)} FROM persons"
    where: list[str] = []
    if hidden_col:
        where.append(f"COALESCE({_q(hidden_col)}, 0) = 0")
    if deleted_col:
        where.append(f"{_q(deleted_col)} IS NULL")
    if where:
        sql += " WHERE " + " AND ".join(where)
    sql += f" ORDER BY {_q(id_col)} ASC LIMIT 200"

    identities: list[PeopleIdentity] = []
    for index, row in enumerate(conn.execute(sql).fetchall(), start=1):
        person_id = _safe_id(row[id_col])
        media_count = _person_media_count(conn, person_id)
        first_seen, last_seen = _person_seen_dates(conn, person_id)
        identities.append(
            PeopleIdentity(
                identity_label=_anonymous_label("人物", index),
                source_pointer=f"personal_lifelog_rag:persons:{person_id}",
                person_source_id=f"plr:person:{person_id}",
                verification_status=_verification_status(row[manual_verified_col] if manual_verified_col else None),
                media_count=media_count,
                first_seen_date=first_seen,
                last_seen_date=last_seen,
                manual_label=_private_text(row[display_col], privacy_level) if display_col else None,
                warning=None,
            )
        )
    return identities


def _person_media_count(conn: sqlite3.Connection, person_id: str) -> int:
    if not table_exists(conn, "media_people") or "person_id" not in table_columns(conn, "media_people"):
        return 0
    where = '"person_id" = ?'
    if "hidden" in table_columns(conn, "media_people"):
        where += ' AND COALESCE("hidden", 0) = 0'
    return _count_where(conn, "media_people", where, (person_id,))


def _person_seen_dates(conn: sqlite3.Connection, person_id: str) -> tuple[str | None, str | None]:
    dates: list[str] = []
    if table_exists(conn, "media_people") and table_exists(conn, "media_items"):
        media_people_cols = table_columns(conn, "media_people")
        media_cols = table_columns(conn, "media_items")
        media_id_col = first_existing_column(media_people_cols, ("media_id",))
        item_id_col = first_existing_column(media_cols, ("id", "media_id"))
        date_col = first_existing_column(media_cols, ("captured_at", "fallback_captured_at", "created_at", "date"))
        if media_id_col and item_id_col and date_col:
            sql = (
                f"SELECT MIN(substr(mi.{_q(date_col)}, 1, 10)) AS first_date, "
                f"MAX(substr(mi.{_q(date_col)}, 1, 10)) AS last_date "
                f"FROM media_people mp JOIN media_items mi ON mp.{_q(media_id_col)} = mi.{_q(item_id_col)} "
                'WHERE mp."person_id" = ?'
            )
            row = conn.execute(sql, (person_id,)).fetchone()
            dates.extend(_date_values(row))
    if table_exists(conn, "event_people") and table_exists(conn, "events"):
        event_people_cols = table_columns(conn, "event_people")
        event_cols = table_columns(conn, "events")
        event_id_col = first_existing_column(event_people_cols, ("event_id",))
        item_id_col = first_existing_column(event_cols, ("id", "event_id"))
        date_col = first_existing_column(event_cols, ("date", "event_date", "start_date", "created_at"))
        if event_id_col and item_id_col and date_col:
            sql = (
                f"SELECT MIN(substr(e.{_q(date_col)}, 1, 10)) AS first_date, "
                f"MAX(substr(e.{_q(date_col)}, 1, 10)) AS last_date "
                f"FROM event_people ep JOIN events e ON ep.{_q(event_id_col)} = e.{_q(item_id_col)} "
                'WHERE ep."person_id" = ?'
            )
            row = conn.execute(sql, (person_id,)).fetchone()
            dates.extend(_date_values(row))
    clean = sorted(date for date in dates if date and date != "1970-01-01")
    if not clean:
        return None, None
    return clean[0], clean[-1]


def _load_line_speaker_identities(
    db_path: str | None,
    *,
    privacy_level: str,
) -> tuple[list[LineSpeakerIdentity], list[str]]:
    if not db_path:
        return [], ["personal_lifelog_db is not configured"]
    try:
        conn = open_readonly_sqlite(db_path)
    except (UpstreamReadOnlyError, sqlite3.Error):
        return [], ["personal_lifelog_db could not be opened read-only"]
    try:
        if table_exists(conn, "line_speaker_links"):
            speakers = _query_line_speaker_links(conn, privacy_level=privacy_level)
        else:
            speakers = _query_line_speaker_groups(conn, privacy_level=privacy_level)
    finally:
        conn.close()
    if not speakers:
        return [], ["no upstream LINE speaker identities found"]
    return speakers, []


def _query_line_speaker_links(conn: sqlite3.Connection, *, privacy_level: str) -> list[LineSpeakerIdentity]:
    columns = table_columns(conn, "line_speaker_links")
    id_col = first_existing_column(columns, ("id",))
    speaker_col = first_existing_column(columns, ("speaker_name", "display_name", "sender", "name"))
    chat_col = first_existing_column(columns, ("chat_id", "conversation_id", "room_id"))
    verified_col = first_existing_column(columns, ("verified_by_user", "manual_verified", "review_status", "status"))
    if not id_col:
        return []
    select_columns = [id_col]
    for column in (speaker_col, chat_col, verified_col):
        if column and column not in select_columns:
            select_columns.append(column)
    sql = f"SELECT {', '.join(_q(column) for column in select_columns)} FROM line_speaker_links ORDER BY {_q(id_col)} ASC LIMIT 200"
    identities: list[LineSpeakerIdentity] = []
    for index, row in enumerate(conn.execute(sql).fetchall(), start=1):
        link_id = _safe_id(row[id_col])
        speaker_name = str(row[speaker_col]) if speaker_col and row[speaker_col] is not None else None
        chat_id = str(row[chat_col]) if chat_col and row[chat_col] is not None else None
        message_count, first_date, last_date = _speaker_message_metrics(conn, speaker_name=speaker_name, chat_id=chat_id)
        identities.append(
            LineSpeakerIdentity(
                identity_label=_anonymous_label("話者", index),
                source_pointer=f"personal_lifelog_rag:line_speaker_links:{link_id}",
                line_speaker_source_id=f"plr:line_speaker:{link_id}",
                message_count=message_count,
                first_message_date=first_date,
                last_message_date=last_date,
                verification_status=_verification_status(row[verified_col] if verified_col else None),
                display_name=_private_text(speaker_name, privacy_level),
                warning=None,
            )
        )
    return identities


def _query_line_speaker_groups(conn: sqlite3.Connection, *, privacy_level: str) -> list[LineSpeakerIdentity]:
    if not table_exists(conn, "line_messages"):
        return []
    columns = table_columns(conn, "line_messages")
    sender_col = first_existing_column(columns, ("sender", "speaker_id", "sender_id"))
    chat_col = first_existing_column(columns, ("chat_id", "conversation_id", "room_id"))
    date_col = first_existing_column(columns, ("sent_at", "timestamp", "created_at", "date"))
    if not sender_col:
        return []
    group_columns = [sender_col]
    if chat_col:
        group_columns.insert(0, chat_col)
    select_parts = [f"{_q(column)} AS {_q(column)}" for column in group_columns]
    if date_col:
        select_parts.extend(
            [
                f"MIN(substr({_q(date_col)}, 1, 10)) AS first_date",
                f"MAX(substr({_q(date_col)}, 1, 10)) AS last_date",
            ]
        )
    select_parts.append("COUNT(*) AS message_count")
    group_by = ", ".join(_q(column) for column in group_columns)
    sql = (
        f"SELECT {', '.join(select_parts)} FROM line_messages "
        f"GROUP BY {group_by} ORDER BY message_count DESC LIMIT 200"
    )
    identities: list[LineSpeakerIdentity] = []
    for index, row in enumerate(conn.execute(sql).fetchall(), start=1):
        speaker_name = str(row[sender_col]) if row[sender_col] is not None else ""
        chat_id = str(row[chat_col]) if chat_col and row[chat_col] is not None else ""
        source_hash = hashlib.sha256(f"{chat_id}|{speaker_name}".encode("utf-8")).hexdigest()[:12]
        identities.append(
            LineSpeakerIdentity(
                identity_label=_anonymous_label("話者", index),
                source_pointer=f"personal_lifelog_rag:line_messages:speaker_group:{source_hash}",
                line_speaker_source_id=f"plr:line_speaker_group:{source_hash}",
                message_count=int(row["message_count"] or 0),
                first_message_date=normalize_date(row["first_date"]) if date_col and row["first_date"] else None,
                last_message_date=normalize_date(row["last_date"]) if date_col and row["last_date"] else None,
                verification_status="unlinked",
                display_name=_private_text(speaker_name, privacy_level),
                warning="line_speaker_links table not found; grouped by speaker only",
            )
        )
    return identities


def _speaker_message_metrics(conn: sqlite3.Connection, *, speaker_name: str | None, chat_id: str | None) -> tuple[int, str | None, str | None]:
    if not speaker_name or not table_exists(conn, "line_messages"):
        return 0, None, None
    columns = table_columns(conn, "line_messages")
    sender_col = first_existing_column(columns, ("sender", "speaker_id", "sender_id"))
    chat_col = first_existing_column(columns, ("chat_id", "conversation_id", "room_id"))
    date_col = first_existing_column(columns, ("sent_at", "timestamp", "created_at", "date"))
    if not sender_col:
        return 0, None, None
    where = [f"{_q(sender_col)} = ?"]
    params: list[Any] = [speaker_name]
    if chat_id and chat_col:
        where.append(f"{_q(chat_col)} = ?")
        params.append(chat_id)
    if date_col:
        sql = (
            f"SELECT COUNT(*) AS message_count, MIN(substr({_q(date_col)}, 1, 10)) AS first_date, "
            f"MAX(substr({_q(date_col)}, 1, 10)) AS last_date FROM line_messages WHERE {' AND '.join(where)}"
        )
    else:
        sql = f"SELECT COUNT(*) AS message_count FROM line_messages WHERE {' AND '.join(where)}"
    row = conn.execute(sql, params).fetchone()
    first_date = normalize_date(row["first_date"]) if date_col and row["first_date"] else None
    last_date = normalize_date(row["last_date"]) if date_col and row["last_date"] else None
    return int(row["message_count"] or 0), first_date, last_date


def _people_table(people: list[PeopleIdentity], *, private: bool) -> list[str]:
    lines = ["## People", ""]
    if not people:
        return [*lines, "No people identities found.", ""]
    headers = [
        "label",
        "source_pointer",
        "person_source_id",
        "verification_status",
        "media_count",
        "first_seen_date",
        "last_seen_date",
    ]
    if private:
        headers.append("manual_label")
    headers.append("warning")
    lines.extend(_markdown_table(headers, [_people_row(item, private=private) for item in people]))
    lines.append("")
    return lines


def _line_speaker_table(speakers: list[LineSpeakerIdentity], *, private: bool) -> list[str]:
    lines = ["## LINE Speakers", ""]
    if not speakers:
        return [*lines, "No LINE speaker identities found.", ""]
    headers = [
        "label",
        "source_pointer",
        "line_speaker_source_id",
        "message_count",
        "first_message_date",
        "last_message_date",
        "verification_status",
    ]
    if private:
        headers.append("display_name")
    headers.append("warning")
    lines.extend(_markdown_table(headers, [_speaker_row(item, private=private) for item in speakers]))
    lines.append("")
    return lines


def _people_row(item: PeopleIdentity, *, private: bool) -> list[Any]:
    row: list[Any] = [
        item.identity_label,
        item.source_pointer,
        item.person_source_id,
        item.verification_status,
        item.media_count,
        item.first_seen_date,
        item.last_seen_date,
    ]
    if private:
        row.append(item.manual_label)
    row.append(item.warning)
    return row


def _speaker_row(item: LineSpeakerIdentity, *, private: bool) -> list[Any]:
    row: list[Any] = [
        item.identity_label,
        item.source_pointer,
        item.line_speaker_source_id,
        item.message_count,
        item.first_message_date,
        item.last_message_date,
        item.verification_status,
    ]
    if private:
        row.append(item.display_name)
    row.append(item.warning)
    return row


def _markdown_table(headers: list[str], rows: list[list[Any]]) -> list[str]:
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(_markdown_cell(value) for value in row) + " |")
    return lines


def _mock_report(kind: str, privacy_level: str) -> UpstreamIdentityReport:
    people = []
    speakers = []
    if kind in {"people", "all"}:
        people = [
            PeopleIdentity(
                identity_label="人物A",
                source_pointer="mock:persons:1",
                person_source_id="mock:person:1",
                verification_status="manual_verified",
                media_count=3,
                first_seen_date="2025-01-01",
                last_seen_date="2025-01-31",
                manual_label="Mock Person" if privacy_level == "private" else None,
            )
        ]
    if kind in {"line-speakers", "all"}:
        speakers = [
            LineSpeakerIdentity(
                identity_label="話者A",
                source_pointer="mock:line_speaker_links:1",
                line_speaker_source_id="mock:line_speaker:1",
                message_count=12,
                first_message_date="2025-01-01",
                last_message_date="2025-01-31",
                verification_status="manual_verified",
                display_name="Mock Speaker" if privacy_level == "private" else None,
            )
        ]
    return UpstreamIdentityReport(
        status="OK",
        backend="mock",
        kind=kind,
        privacy_level=privacy_level,
        people=people,
        line_speakers=speakers,
        warnings=[],
    )


def _error_report(backend: str, kind: str, privacy_level: str, warning: str) -> UpstreamIdentityReport:
    return UpstreamIdentityReport(
        status="ERROR",
        backend=backend,
        kind=kind,
        privacy_level=privacy_level,
        people=[],
        line_speakers=[],
        warnings=[warning],
    )


def _anonymous_label(prefix: str, index: int) -> str:
    return f"{prefix}{_alphabet_label(index)}"


def _alphabet_label(index: int) -> str:
    alphabet = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    if index <= len(alphabet):
        return alphabet[index - 1]
    return str(index)


def _verification_status(value: Any) -> str:
    if value is None:
        return "unknown"
    if isinstance(value, bool):
        return "manual_verified" if value else "unverified"
    text = str(value).strip().lower()
    if text in {"1", "true", "yes", "verified", "manual_verified", "accepted"}:
        return "manual_verified"
    if text in {"0", "false", "no", "unverified", "none", ""}:
        return "unverified"
    return text


def _private_text(value: Any, privacy_level: str) -> str | None:
    if privacy_level != "private" or value is None:
        return None
    text = " ".join(str(value).split())
    return text[:80] if text else None


def _count_where(conn: sqlite3.Connection, table_name: str, where: str, params: tuple[Any, ...]) -> int:
    row = conn.execute(f"SELECT COUNT(*) AS count FROM {_q(table_name)} WHERE {where}", params).fetchone()
    return int(row["count"] or 0) if row else 0


def _date_values(row: sqlite3.Row | None) -> list[str]:
    if row is None:
        return []
    values: list[str] = []
    for key in ("first_date", "last_date"):
        value = row[key]
        if value:
            values.append(normalize_date(value))
    return values


def _safe_id(value: Any) -> str:
    return str(value).strip()


def _markdown_cell(value: Any) -> str:
    if value in (None, ""):
        return "-"
    text = str(value)
    return text.replace("|", "\\|").replace("\n", " ")


def _q(value: str) -> str:
    return '"' + value.replace('"', '""') + '"'
