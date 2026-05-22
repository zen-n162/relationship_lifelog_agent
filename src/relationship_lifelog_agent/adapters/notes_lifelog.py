from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from relationship_lifelog_agent.adapters.types import MonthlyReflection, NoteEvidence, ThoughtEvidence
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


class NotesLifelogAdapter(Protocol):
    """Read-only interface for notes_lifelog_rag."""

    def search_notes(
        self,
        query: str,
        date_from: str | None = None,
        date_to: str | None = None,
        limit: int = 50,
        mode: str = "private",
    ) -> list[NoteEvidence]:
        ...

    def search_thoughts(
        self,
        query: str,
        date_from: str | None = None,
        date_to: str | None = None,
        limit: int = 50,
        mode: str = "private",
    ) -> list[ThoughtEvidence]:
        ...

    def get_monthly_reflection(self, month: str, mode: str = "private") -> MonthlyReflection:
        ...


@dataclass(frozen=True)
class DisabledNotesLifelogAdapter:
    root_path: str
    warning: str = "上流adapter未設定: notes_lifelog_rag は有効化されていません。"

    def warnings(self) -> list[str]:
        return [self.warning]

    def search_notes(
        self,
        query: str,
        date_from: str | None = None,
        date_to: str | None = None,
        limit: int = 50,
        mode: str = "private",
    ) -> list[NoteEvidence]:
        del query, date_from, date_to, limit, mode
        return []

    def search_thoughts(
        self,
        query: str,
        date_from: str | None = None,
        date_to: str | None = None,
        limit: int = 50,
        mode: str = "private",
    ) -> list[ThoughtEvidence]:
        del query, date_from, date_to, limit, mode
        return []

    def get_monthly_reflection(self, month: str, mode: str = "private") -> MonthlyReflection:
        del mode
        return _missing_reflection(month, self.warning)


class NotesSqliteReadOnlyAdapter:
    """Read-only SQLite adapter for notes_lifelog_rag."""

    source_namespace = "notes_lifelog_rag"

    def __init__(self, db_path: str | Path | None, *, excerpt_chars: int = 120) -> None:
        self.db_path = str(db_path) if db_path else None
        self.excerpt_chars = excerpt_chars
        self._warnings: list[str] = []

    def warnings(self) -> list[str]:
        return list(self._warnings)

    def search_notes(
        self,
        query: str,
        date_from: str | None = None,
        date_to: str | None = None,
        limit: int = 50,
        mode: str = "private",
    ) -> list[NoteEvidence]:
        conn = self._connect()
        if conn is None:
            return []
        try:
            if not table_exists(conn, "notes"):
                self._warn("上流notes adapter: notes テーブルが見つかりません。")
                return []
            columns = table_columns(conn, "notes")
            id_col = first_existing_column(columns, ("id", "note_id"))
            date_col = first_existing_column(columns, ("note_date", "date", "created_at", "updated_at"))
            title_col = first_existing_column(columns, ("title", "generated_title"))
            body_col = first_existing_column(columns, ("body", "text", "content"))
            summary_col = first_existing_column(columns, ("summary", "generated_summary"))
            if not id_col:
                self._warn("上流notes adapter: notes の必須列が不足しています。")
                return []
            where: list[str] = []
            params: list[object] = []
            where_date_range(where, params, date_col, date_from, date_to)
            where_text_search(where, params, [col for col in (title_col, summary_col, body_col) if col], query)
            selected = _selected_columns(id_col, date_col, title_col, summary_col, body_col)
            rows = select_rows(conn, "notes", selected, where, params, order_by=date_col, limit=limit)
            return [
                NoteEvidence(
                    source_id=str(row[id_col]),
                    date=normalize_date(row_value(row, date_col)),
                    role="supporting",
                    summary=_note_summary(row_value(row, title_col), row_value(row, summary_col), query),
                    excerpt=short_excerpt(row_value(row, summary_col) or row_value(row, body_col) or row_value(row, title_col), mode=mode, max_chars=self.excerpt_chars),
                    confidence=0.5,
                    evidence_strength=0.45,
                    sensitivity="private",
                    source_pointer=self._pointer("notes", row[id_col]),
                    note_id=str(row[id_col]),
                    category="relationship_reflection",
                    metadata=_metadata("notes"),
                )
                for row in rows
            ]
        finally:
            conn.close()

    def search_thoughts(
        self,
        query: str,
        date_from: str | None = None,
        date_to: str | None = None,
        limit: int = 50,
        mode: str = "private",
    ) -> list[ThoughtEvidence]:
        conn = self._connect()
        if conn is None:
            return []
        try:
            if not table_exists(conn, "thoughts"):
                self._warn("上流notes adapter: thoughts テーブルが見つかりません。")
                return []
            columns = table_columns(conn, "thoughts")
            id_col = first_existing_column(columns, ("id", "thought_id"))
            note_col = first_existing_column(columns, ("note_id",))
            date_col = first_existing_column(columns, ("date", "date_label", "created_at"))
            title_col = first_existing_column(columns, ("title",))
            summary_col = first_existing_column(columns, ("summary", "text"))
            type_col = first_existing_column(columns, ("thought_type", "type"))
            themes_col = first_existing_column(columns, ("themes", "emotion_labels", "tags"))
            confidence_col = first_existing_column(columns, ("confidence", "score"))
            if not id_col:
                self._warn("上流notes adapter: thoughts の必須列が不足しています。")
                return []
            where: list[str] = []
            params: list[object] = []
            where_date_range(where, params, date_col, date_from, date_to)
            where_text_search(where, params, [col for col in (title_col, summary_col, themes_col) if col], query)
            selected = _selected_columns(id_col, note_col, date_col, title_col, summary_col, type_col, themes_col, confidence_col)
            rows = select_rows(conn, "thoughts", selected, where, params, order_by=date_col, limit=limit)
            return [
                ThoughtEvidence(
                    source_id=str(row[id_col]),
                    date=normalize_date(row_value(row, date_col)),
                    role="supporting",
                    summary="上流思考メモに自分側の振り返り候補があります。",
                    excerpt=short_excerpt(row_value(row, summary_col) or row_value(row, title_col), mode=mode, max_chars=self.excerpt_chars),
                    confidence=_safe_score(row_value(row, confidence_col), default=0.5),
                    evidence_strength=0.45,
                    sensitivity="private",
                    source_pointer=self._pointer("thoughts", row[id_col]),
                    thought_id=str(row[id_col]),
                    thought_type=str(row[type_col]) if type_col and row[type_col] else "own_thought",
                    metadata=_metadata("thoughts", note_id=row_value(row, note_col), themes=row_value(row, themes_col)),
                )
                for row in rows
            ]
        finally:
            conn.close()

    def get_monthly_reflection(self, month: str, mode: str = "private") -> MonthlyReflection:
        conn = self._connect()
        if conn is None:
            return _missing_reflection(month, self._warnings[-1])
        try:
            if not table_exists(conn, "monthly_reflections"):
                self._warn("上流notes adapter: monthly_reflections テーブルが見つかりません。")
                return _missing_reflection(month, self._warnings[-1])
            columns = table_columns(conn, "monthly_reflections")
            month_col = first_existing_column(columns, ("month", "target_month"))
            summary_col = first_existing_column(columns, ("summary", "summary_json", "reflection", "body"))
            confidence_col = first_existing_column(columns, ("confidence", "importance", "score"))
            if not month_col:
                self._warn("上流notes adapter: monthly_reflections の月列が不足しています。")
                return _missing_reflection(month, self._warnings[-1])
            selected = _selected_columns(month_col, summary_col, confidence_col)
            rows = select_rows(
                conn,
                "monthly_reflections",
                selected,
                [f'"{month_col}" = ?'],
                [month],
                order_by=month_col,
                limit=1,
            )
            if not rows:
                self._warn("上流notes adapter: 指定月の月次振り返りが見つかりません。")
                return _missing_reflection(month, self._warnings[-1])
            row = rows[0]
            confidence = _safe_score(row_value(row, confidence_col), default=0.5)
            return MonthlyReflection(
                source_id=month,
                date=f"{month}-01",
                role="context",
                summary="上流月次振り返りに関係レビュー候補があります。",
                excerpt=short_excerpt(row_value(row, summary_col), mode=mode, max_chars=self.excerpt_chars),
                confidence=confidence,
                evidence_strength=max(0.35, min(confidence, 0.7)),
                sensitivity="private",
                source_pointer=self._pointer("monthly_reflections", month),
                month=month,
                metadata=_metadata("monthly_reflections"),
            )
        finally:
            conn.close()

    def _connect(self) -> sqlite3.Connection | None:
        if not self.db_path:
            self._warn("上流adapter未設定: notes_lifelog_db が未設定です。")
            return None
        try:
            return open_readonly_sqlite(self.db_path)
        except UpstreamReadOnlyError:
            self._warn("上流adapter未設定: notes_lifelog_db を read-only で開けません。")
            return None
        except sqlite3.Error:
            self._warn("上流notes adapter: SQLite read-only 接続に失敗しました。")
            return None

    def _pointer(self, table_name: str, row_id: object) -> str:
        return f"{self.source_namespace}:{table_name}:{row_id}"

    def _warn(self, message: str) -> None:
        if message not in self._warnings:
            self._warnings.append(message)


def _missing_reflection(month: str, warning: str) -> MonthlyReflection:
    return MonthlyReflection(
        source_id=f"upstream-missing-reflection-{month}",
        date=f"{month}-01",
        role="context",
        summary="上流月次振り返りは取得できませんでした。",
        excerpt=None,
        confidence=0.0,
        evidence_strength=0.0,
        sensitivity="private",
        source_pointer=f"notes_lifelog_rag:monthly_reflections:{month}",
        month=month,
        metadata={"warning": warning, "adapter_backend": "upstream_readonly"},
    )


def _selected_columns(*columns: str | None) -> list[str]:
    return list(dict.fromkeys(column for column in columns if column))


def _note_summary(title: object, summary: object, query: str) -> str:
    text = " ".join(str(value) for value in (title, summary, query) if value)
    if any(token in text for token in ("反省", "不安", "感謝", "関係")):
        return "上流ノートに自分側の反省・不安・振り返り候補があります。"
    return "上流ノートに検索条件と一致する候補があります。"


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
