from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from relationship_lifelog_agent.adapters.types import (
    EventEvidence,
    LineEvidence,
    MediaEvidence,
)
from relationship_lifelog_agent.adapters.upstream_sqlite import (
    UpstreamReadOnlyError,
    first_existing_column,
    normalize_date,
    open_readonly_sqlite,
    row_value,
    select_rows,
    short_excerpt,
    table_columns,
    table_exists,
    where_date_range,
    where_text_search,
)


class PersonalLifelogAdapter(Protocol):
    """Read-only interface for personal_lifelog_rag.

    Implementations must not mutate upstream DBs. The default backend is mock;
    upstream adapters must use read-only access.
    """

    def search_line(
        self,
        query: str,
        date_from: str | None = None,
        date_to: str | None = None,
        speaker_id: str | None = None,
        limit: int = 50,
        mode: str = "private",
    ) -> list[LineEvidence]:
        ...

    def search_media(
        self,
        query: str,
        date_from: str | None = None,
        date_to: str | None = None,
        person_id: str | None = None,
        place_id: str | None = None,
        limit: int = 50,
        mode: str = "private",
    ) -> list[MediaEvidence]:
        ...

    def search_events(
        self,
        date_from: str | None = None,
        date_to: str | None = None,
        event_type: str | None = None,
        person_id: str | None = None,
        place_id: str | None = None,
        limit: int = 50,
        mode: str = "private",
    ) -> list[EventEvidence]:
        ...

    def get_day_summary(self, date: str, mode: str = "private") -> dict[str, str]:
        ...

    def get_month_summary(self, month: str, mode: str = "private") -> dict[str, str]:
        ...


@dataclass(frozen=True)
class DisabledPersonalLifelogAdapter:
    root_path: str
    warning: str = "上流adapter未設定: personal_lifelog_rag は有効化されていません。"

    def warnings(self) -> list[str]:
        return [self.warning]

    def search_line(
        self,
        query: str,
        date_from: str | None = None,
        date_to: str | None = None,
        speaker_id: str | None = None,
        limit: int = 50,
        mode: str = "private",
    ) -> list[LineEvidence]:
        del query, date_from, date_to, speaker_id, limit, mode
        return []

    def search_media(
        self,
        query: str,
        date_from: str | None = None,
        date_to: str | None = None,
        person_id: str | None = None,
        place_id: str | None = None,
        limit: int = 50,
        mode: str = "private",
    ) -> list[MediaEvidence]:
        del query, date_from, date_to, person_id, place_id, limit, mode
        return []

    def search_events(
        self,
        date_from: str | None = None,
        date_to: str | None = None,
        event_type: str | None = None,
        person_id: str | None = None,
        place_id: str | None = None,
        limit: int = 50,
        mode: str = "private",
    ) -> list[EventEvidence]:
        del date_from, date_to, event_type, person_id, place_id, limit, mode
        return []

    def get_day_summary(self, date: str, mode: str = "private") -> dict[str, str]:
        del date, mode
        return {"warning": self.warning, "summary": "上流adapter未設定です。"}

    def get_month_summary(self, month: str, mode: str = "private") -> dict[str, str]:
        del month, mode
        return {"warning": self.warning, "summary": "上流adapter未設定です。"}


class PersonalSqliteReadOnlyAdapter:
    """Read-only SQLite adapter for personal_lifelog_rag.

    This adapter opens the upstream database with ``mode=ro`` for each call,
    returns source pointers and short excerpts only, and records warnings
    instead of raising UI-breaking errors for missing optional tables.
    """

    source_namespace = "personal_lifelog_rag"

    def __init__(self, db_path: str | Path | None, *, excerpt_chars: int = 120) -> None:
        self.db_path = str(db_path) if db_path else None
        self.excerpt_chars = excerpt_chars
        self._warnings: list[str] = []

    def warnings(self) -> list[str]:
        return list(self._warnings)

    def search_line(
        self,
        query: str,
        date_from: str | None = None,
        date_to: str | None = None,
        speaker_id: str | None = None,
        limit: int = 50,
        mode: str = "private",
    ) -> list[LineEvidence]:
        conn = self._connect()
        if conn is None:
            return []
        try:
            if not table_exists(conn, "line_messages"):
                self._warn("上流personal adapter: line_messages テーブルが見つかりません。")
                return []
            columns = table_columns(conn, "line_messages")
            id_col = first_existing_column(columns, ("id", "message_id"))
            date_col = first_existing_column(columns, ("sent_at", "timestamp", "created_at", "date"))
            text_col = first_existing_column(columns, ("text", "message_text", "body", "content"))
            sender_col = first_existing_column(columns, ("sender", "speaker_id", "sender_id"))
            chat_col = first_existing_column(columns, ("chat_id", "conversation_id", "room_id"))
            type_col = first_existing_column(columns, ("message_type", "type"))
            if not id_col or not date_col:
                self._warn("上流personal adapter: line_messages の必須列が不足しています。")
                return []
            where: list[str] = []
            params: list[object] = []
            where_date_range(where, params, date_col, date_from, date_to)
            where_text_search(where, params, [col for col in (text_col,) if col], query)
            if speaker_id and sender_col:
                where.append(f'"{sender_col}" = ?')
                params.append(speaker_id)
            selected = _selected_columns(id_col, date_col, text_col, sender_col, chat_col, type_col)
            rows = select_rows(conn, "line_messages", selected, where, params, order_by=date_col, limit=limit)
            return [
                LineEvidence(
                    source_id=str(row[id_col]),
                    date=normalize_date(row[date_col]),
                    role="primary" if _looks_conflict_related(row_value(row, text_col), query) else "supporting",
                    summary=_line_summary(row_value(row, text_col), query),
                    excerpt=short_excerpt(row_value(row, text_col), mode=mode, max_chars=self.excerpt_chars),
                    confidence=0.55 if text_col else 0.35,
                    evidence_strength=0.5 if text_col else 0.3,
                    sensitivity="private",
                    source_pointer=self._pointer("line_messages", row[id_col]),
                    speaker_id=str(row[sender_col]) if sender_col and row[sender_col] is not None else None,
                    conversation_id=str(row[chat_col]) if chat_col and row[chat_col] is not None else None,
                    metadata=_metadata("line_messages", message_type=row_value(row, type_col)),
                )
                for row in rows
            ]
        finally:
            conn.close()

    def search_media(
        self,
        query: str,
        date_from: str | None = None,
        date_to: str | None = None,
        person_id: str | None = None,
        place_id: str | None = None,
        limit: int = 50,
        mode: str = "private",
    ) -> list[MediaEvidence]:
        del person_id
        conn = self._connect()
        if conn is None:
            return []
        try:
            if not table_exists(conn, "media_items"):
                self._warn("上流personal adapter: media_items テーブルが見つかりません。")
                return []
            columns = table_columns(conn, "media_items")
            id_col = first_existing_column(columns, ("id", "media_id"))
            date_col = first_existing_column(columns, ("captured_at", "fallback_captured_at", "created_at", "date"))
            caption_col = first_existing_column(columns, ("caption", "vlm_caption", "description"))
            ocr_col = first_existing_column(columns, ("ocr_text", "redacted_ocr_text"))
            type_col = first_existing_column(columns, ("media_type", "type"))
            place_col = first_existing_column(columns, ("place_id",))
            if not id_col or not date_col:
                self._warn("上流personal adapter: media_items の必須列が不足しています。")
                return []
            where: list[str] = []
            params: list[object] = []
            where_date_range(where, params, date_col, date_from, date_to)
            where_text_search(where, params, [col for col in (caption_col, ocr_col) if col], query)
            if place_id and place_col:
                where.append(f'"{place_col}" = ?')
                params.append(place_id)
            selected = _selected_columns(id_col, date_col, caption_col, ocr_col, type_col, place_col)
            rows = select_rows(conn, "media_items", selected, where, params, order_by=date_col, limit=limit)
            return [
                MediaEvidence(
                    source_id=str(row[id_col]),
                    date=normalize_date(row[date_col]),
                    role="after_event" if _contains_any(row_value(row, caption_col), ("外出", "場所", "カフェ", "食事")) else "supporting",
                    summary="上流メディア記録に外出・場所に関する候補があります。",
                    excerpt=short_excerpt(row_value(row, caption_col) or row_value(row, ocr_col), mode=mode, max_chars=self.excerpt_chars),
                    confidence=0.5 if caption_col or ocr_col else 0.35,
                    evidence_strength=0.45 if caption_col or ocr_col else 0.3,
                    sensitivity="sensitive",
                    source_pointer=self._pointer("media_items", row[id_col]),
                    media_id=str(row[id_col]),
                    media_type=str(row[type_col]) if type_col and row[type_col] else "photo",
                    place_id=str(row[place_col]) if place_col and row[place_col] else None,
                    metadata=_metadata("media_items"),
                )
                for row in rows
            ]
        finally:
            conn.close()

    def search_events(
        self,
        date_from: str | None = None,
        date_to: str | None = None,
        event_type: str | None = None,
        person_id: str | None = None,
        place_id: str | None = None,
        limit: int = 50,
        mode: str = "private",
    ) -> list[EventEvidence]:
        del person_id, place_id
        conn = self._connect()
        if conn is None:
            return []
        try:
            if not table_exists(conn, "events"):
                self._warn("上流personal adapter: events テーブルが見つかりません。")
                return []
            columns = table_columns(conn, "events")
            id_col = first_existing_column(columns, ("id", "event_id"))
            date_col = first_existing_column(columns, ("date", "event_date", "start_date", "created_at"))
            title_col = first_existing_column(columns, ("title", "name"))
            summary_col = first_existing_column(columns, ("summary", "description"))
            type_col = first_existing_column(columns, ("event_type", "type"))
            confidence_col = first_existing_column(columns, ("confidence", "score"))
            if not id_col or not date_col:
                self._warn("上流personal adapter: events の必須列が不足しています。")
                return []
            where: list[str] = []
            params: list[object] = []
            where_date_range(where, params, date_col, date_from, date_to)
            if event_type and type_col:
                where.append(f'"{type_col}" = ?')
                params.append(event_type)
            selected = _selected_columns(id_col, date_col, title_col, summary_col, type_col, confidence_col)
            rows = select_rows(conn, "events", selected, where, params, order_by=date_col, limit=limit)
            events = [
                self._event_from_row(row, id_col, date_col, title_col, summary_col, type_col, confidence_col, mode)
                for row in rows
            ]
            if event_type:
                events = [item for item in events if item.event_type == event_type]
            return events
        finally:
            conn.close()

    def get_day_summary(self, date: str, mode: str = "private") -> dict[str, str]:
        del mode
        conn = self._connect()
        if conn is None:
            return {"date": date, "summary": "上流adapter未設定です。", "warning": self._warnings[-1]}
        try:
            counts = self._counts_for_range(conn, date, date)
            return {"date": date, "summary": _count_summary(counts)}
        finally:
            conn.close()

    def get_month_summary(self, month: str, mode: str = "private") -> dict[str, str]:
        del mode
        conn = self._connect()
        if conn is None:
            return {"month": month, "summary": "上流adapter未設定です。", "warning": self._warnings[-1]}
        try:
            date_from = f"{month}-01"
            date_to = f"{month}-31"
            counts = self._counts_for_range(conn, date_from, date_to)
            return {"month": month, "summary": _count_summary(counts)}
        finally:
            conn.close()

    def _connect(self) -> sqlite3.Connection | None:
        if not self.db_path:
            self._warn("上流adapter未設定: personal_lifelog_db が未設定です。")
            return None
        try:
            return open_readonly_sqlite(self.db_path)
        except UpstreamReadOnlyError:
            self._warn("上流adapter未設定: personal_lifelog_db を read-only で開けません。")
            return None
        except sqlite3.Error:
            self._warn("上流personal adapter: SQLite read-only 接続に失敗しました。")
            return None

    def _event_from_row(
        self,
        row: sqlite3.Row,
        id_col: str,
        date_col: str,
        title_col: str | None,
        summary_col: str | None,
        type_col: str | None,
        confidence_col: str | None,
        mode: str,
    ) -> EventEvidence:
        text = " ".join(str(value) for value in (row_value(row, title_col), row_value(row, summary_col), row_value(row, type_col)) if value)
        mapped_type = _map_event_type(text, row_value(row, type_col))
        confidence = _safe_score(row_value(row, confidence_col), default=0.45)
        return EventEvidence(
            source_id=str(row[id_col]),
            date=normalize_date(row[date_col]),
            role="primary" if mapped_type in {"conflict", "minor_misunderstanding"} else "supporting",
            summary=_event_summary(mapped_type),
            excerpt=short_excerpt(text, mode=mode, max_chars=self.excerpt_chars),
            confidence=confidence,
            evidence_strength=max(0.35, min(confidence, 0.7)),
            sensitivity="private",
            source_pointer=self._pointer("events", row[id_col]),
            event_type=mapped_type,
            review_status="candidate",
            severity=_severity_for_type(mapped_type),
            metadata=_metadata("events", upstream_event_type=row_value(row, type_col)),
        )

    def _counts_for_range(self, conn: sqlite3.Connection, date_from: str, date_to: str) -> dict[str, int]:
        counts: dict[str, int] = {}
        for table_name, date_candidates in {
            "line_messages": ("sent_at", "timestamp", "created_at", "date"),
            "media_items": ("captured_at", "fallback_captured_at", "created_at", "date"),
            "events": ("date", "event_date", "start_date", "created_at"),
        }.items():
            if not table_exists(conn, table_name):
                counts[table_name] = 0
                continue
            columns = table_columns(conn, table_name)
            date_col = first_existing_column(columns, date_candidates)
            if not date_col:
                counts[table_name] = 0
                continue
            counts[table_name] = int(
                conn.execute(
                    f'SELECT COUNT(*) AS count FROM "{table_name}" WHERE substr("{date_col}", 1, 10) >= ? AND substr("{date_col}", 1, 10) <= ?',
                    (date_from, date_to),
                ).fetchone()["count"]
            )
        return counts

    def _pointer(self, table_name: str, row_id: object) -> str:
        return f"{self.source_namespace}:{table_name}:{row_id}"

    def _warn(self, message: str) -> None:
        if message not in self._warnings:
            self._warnings.append(message)


def _selected_columns(*columns: str | None) -> list[str]:
    return list(dict.fromkeys(column for column in columns if column))


def _line_summary(text: object, query: str) -> str:
    if _looks_conflict_related(text, query):
        return "上流LINE記録に喧嘩候補・すれ違い候補に関係する表現があります。"
    return "上流LINE記録に検索条件と一致するメッセージ候補があります。"


def _event_summary(event_type: str) -> str:
    labels = {
        "conflict": "上流イベントに中程度の喧嘩候補があります。",
        "minor_misunderstanding": "上流イベントに軽いすれ違い候補があります。",
        "date_or_outing": "上流イベントに外出候補があります。",
        "reconciliation": "上流イベントに通常のやりとりへ戻った可能性のある候補があります。",
    }
    return labels.get(event_type, "上流イベントに関係イベント候補があります。")


def _map_event_type(text: str, upstream_type: object) -> str:
    if _contains_any(text, ("喧嘩", "けんか", "強い不満")):
        return "conflict"
    if _contains_any(text, ("すれ違い", "返信遅延", "予定変更")):
        return "minor_misunderstanding"
    if _contains_any(text, ("仲直り", "通常会話", "謝罪後")):
        return "reconciliation"
    if _contains_any(text, ("外出", "カフェ", "食事", "場所", "訪問")):
        return "date_or_outing"
    if upstream_type:
        return str(upstream_type)
    return "other"


def _looks_conflict_related(text: object, query: str) -> bool:
    haystack = f"{text or ''} {query}"
    return _contains_any(haystack, ("喧嘩", "けんか", "すれ違い", "謝罪", "不満", "予定変更"))


def _contains_any(value: object, needles: tuple[str, ...]) -> bool:
    text = str(value or "")
    return any(needle in text for needle in needles)


def _severity_for_type(event_type: str) -> int:
    if event_type == "conflict":
        return 2
    if event_type == "minor_misunderstanding":
        return 1
    return 0


def _safe_score(value: object, *, default: float) -> float:
    try:
        score = float(value)
    except (TypeError, ValueError):
        return default
    return max(0.0, min(score, 1.0))


def _metadata(table_name: str, **values: object) -> dict[str, object]:
    metadata: dict[str, object] = {"upstream_table": table_name, "adapter_backend": "upstream_readonly"}
    metadata.update({key: value for key, value in values.items() if value is not None})
    return metadata


def _count_summary(counts: dict[str, int]) -> str:
    return (
        "上流DBをread-onlyで集計しました: "
        f"LINE {counts.get('line_messages', 0)} 件、"
        f"media {counts.get('media_items', 0)} 件、"
        f"event {counts.get('events', 0)} 件。"
    )
