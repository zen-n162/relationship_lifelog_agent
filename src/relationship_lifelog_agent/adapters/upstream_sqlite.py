from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Any, Iterable
from urllib.parse import quote


class UpstreamReadOnlyError(RuntimeError):
    """Raised when a read-only upstream SQLite connection cannot be opened."""


def sqlite_readonly_uri(path: str | Path) -> str:
    db_path = Path(path).expanduser()
    return f"file:{quote(str(db_path), safe='/:')}?mode=ro"


def open_readonly_sqlite(path: str | Path) -> sqlite3.Connection:
    db_path = Path(path).expanduser()
    if not db_path.exists():
        raise UpstreamReadOnlyError("upstream DB path is not configured or does not exist")
    conn = sqlite3.connect(sqlite_readonly_uri(db_path), uri=True)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA query_only = ON")
    return conn


def table_exists(conn: sqlite3.Connection, table_name: str) -> bool:
    row = conn.execute(
        "SELECT 1 FROM sqlite_master WHERE type='table' AND name=?",
        (table_name,),
    ).fetchone()
    return row is not None


def table_columns(conn: sqlite3.Connection, table_name: str) -> set[str]:
    if not table_exists(conn, table_name):
        return set()
    rows = conn.execute(f"PRAGMA table_info({_quote_identifier(table_name)})").fetchall()
    return {str(row["name"]) for row in rows}


def first_existing_column(columns: set[str], candidates: Iterable[str]) -> str | None:
    for candidate in candidates:
        if candidate in columns:
            return candidate
    return None


def normalize_date(value: Any, fallback: str = "1970-01-01") -> str:
    if value is None:
        return fallback
    text = str(value).strip()
    if not text:
        return fallback
    return text[:10]


def short_excerpt(value: Any, *, mode: str = "private", max_chars: int = 120) -> str | None:
    if mode == "public" or value is None:
        return None
    text = " ".join(str(value).split())
    if not text:
        return None
    if len(text) <= max_chars:
        return text
    return f"{text[:max_chars].rstrip()}..."


def safe_like_term(term: str) -> str:
    return f"%{term.replace('%', '').replace('_', '')}%"


def query_terms(query: str) -> list[str]:
    raw = query.strip()
    if not raw:
        return []
    terms = [part for part in raw.replace("　", " ").split(" ") if part]
    for token in (
        "喧嘩",
        "けんか",
        "すれ違い",
        "仲直り",
        "謝罪",
        "予定",
        "外出",
        "場所",
        "反省",
        "不安",
        "感謝",
        "メモ",
    ):
        if token in raw:
            terms.append(token)
    return list(dict.fromkeys(terms))


def where_date_range(
    where: list[str],
    params: list[Any],
    date_column: str | None,
    date_from: str | None,
    date_to: str | None,
) -> None:
    if not date_column:
        return
    quoted = _quote_identifier(date_column)
    if date_from:
        where.append(f"substr({quoted}, 1, 10) >= ?")
        params.append(date_from)
    if date_to:
        where.append(f"substr({quoted}, 1, 10) <= ?")
        params.append(date_to)


def where_text_search(
    where: list[str],
    params: list[Any],
    text_columns: list[str],
    query: str,
) -> None:
    terms = query_terms(query)
    if not terms or not text_columns:
        return
    term_clauses: list[str] = []
    for term in terms:
        column_clauses = [f"{_quote_identifier(column)} LIKE ?" for column in text_columns]
        term_clauses.append("(" + " OR ".join(column_clauses) + ")")
        params.extend([safe_like_term(term)] * len(text_columns))
    where.append("(" + " OR ".join(term_clauses) + ")")


def select_rows(
    conn: sqlite3.Connection,
    table_name: str,
    columns: list[str],
    where: list[str],
    params: list[Any],
    *,
    order_by: str | None = None,
    limit: int = 50,
) -> list[sqlite3.Row]:
    select_list = ", ".join(_quote_identifier(column) for column in columns)
    sql = f"SELECT {select_list} FROM {_quote_identifier(table_name)}"
    if where:
        sql += " WHERE " + " AND ".join(where)
    if order_by:
        sql += f" ORDER BY {_quote_identifier(order_by)} ASC"
    sql += " LIMIT ?"
    return list(conn.execute(sql, (*params, max(1, min(limit, 200)))).fetchall())


def row_value(row: sqlite3.Row, column: str | None) -> Any:
    if not column:
        return None
    return row[column]


def _quote_identifier(value: str) -> str:
    return '"' + value.replace('"', '""') + '"'
