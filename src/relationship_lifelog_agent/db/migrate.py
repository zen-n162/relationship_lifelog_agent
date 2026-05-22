from __future__ import annotations

from importlib import resources
from pathlib import Path
import sqlite3


def initialize_database(db_path: str | Path) -> Path:
    path = Path(db_path).expanduser()
    path.parent.mkdir(parents=True, exist_ok=True)
    schema = resources.files("relationship_lifelog_agent.db").joinpath("schema.sql").read_text(encoding="utf-8")
    with sqlite3.connect(path) as conn:
        conn.execute("PRAGMA foreign_keys = ON")
        table_schema, trigger_schema = _split_trigger_schema(schema)
        conn.executescript(table_schema)
        _ensure_updated_at_columns(conn)
        if trigger_schema:
            conn.executescript(trigger_schema)
    return path


def _split_trigger_schema(schema: str) -> tuple[str, str]:
    marker = "CREATE TRIGGER IF NOT EXISTS"
    if marker not in schema:
        return schema, ""
    table_schema, trigger_schema = schema.split(marker, 1)
    return table_schema, marker + trigger_schema


def _ensure_updated_at_columns(conn: sqlite3.Connection) -> None:
    tables = (
        "relationship_profiles",
        "relationship_events",
        "relationship_event_evidence",
        "interaction_metrics",
        "post_conflict_activities",
        "relationship_review_actions",
    )
    for table in tables:
        columns = {row[1] for row in conn.execute(f"PRAGMA table_info({table})")}
        if "updated_at" not in columns:
            conn.execute(f"ALTER TABLE {table} ADD COLUMN updated_at TEXT")
            conn.execute(
                f"""
                UPDATE {table}
                SET updated_at = COALESCE(created_at, CURRENT_TIMESTAMP)
                WHERE updated_at IS NULL
                """
            )
