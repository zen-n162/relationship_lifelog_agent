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
        _ensure_relationship_profiles_contract(conn)
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


def _ensure_relationship_profiles_contract(conn: sqlite3.Connection) -> None:
    row = conn.execute(
        "SELECT sql FROM sqlite_master WHERE type = 'table' AND name = 'relationship_profiles'"
    ).fetchone()
    if row is None:
        return
    sql = row[0] or ""
    if "label_source = 'user_manual'" in sql and "relationship_label IN" in sql:
        return

    columns = {item[1] for item in conn.execute("PRAGMA table_info(relationship_profiles)")}
    conn.execute("PRAGMA foreign_keys = OFF")
    conn.execute("ALTER TABLE relationship_profiles RENAME TO relationship_profiles_legacy")
    conn.execute(_relationship_profiles_schema())
    select_values = {
        "id": "id" if "id" in columns else "NULL",
        "profile_name": "profile_name" if "profile_name" in columns else "'Manual Profile'",
        "person_source_id": "person_source_id" if "person_source_id" in columns else "NULL",
        "line_speaker_source_id": "line_speaker_source_id" if "line_speaker_source_id" in columns else "NULL",
        "relationship_label": (
            "CASE WHEN relationship_label IN ('partner', 'ex_partner', 'close_person', 'other_private') "
            "THEN relationship_label ELSE NULL END"
            if "relationship_label" in columns
            else "NULL"
        ),
        "valid_from": "valid_from" if "valid_from" in columns else "NULL",
        "valid_to": "valid_to" if "valid_to" in columns else "NULL",
        "visibility": (
            "CASE WHEN visibility IN ('private', 'hidden') THEN visibility ELSE 'private' END"
            if "visibility" in columns
            else "'private'"
        ),
        "notes": "notes" if "notes" in columns else "NULL",
        "created_at": "created_at" if "created_at" in columns else "CURRENT_TIMESTAMP",
        "updated_at": "updated_at" if "updated_at" in columns else "CURRENT_TIMESTAMP",
    }
    conn.execute(
        """
        INSERT INTO relationship_profiles (
          id,
          profile_name,
          person_source_id,
          line_speaker_source_id,
          relationship_label,
          label_source,
          valid_from,
          valid_to,
          visibility,
          notes,
          created_at,
          updated_at
        )
        SELECT
          {id},
          {profile_name},
          {person_source_id},
          {line_speaker_source_id},
          {relationship_label},
          'user_manual',
          {valid_from},
          {valid_to},
          {visibility},
          {notes},
          {created_at},
          {updated_at}
        FROM relationship_profiles_legacy
        """.format(**select_values)
    )
    conn.execute("DROP TABLE relationship_profiles_legacy")
    conn.execute("PRAGMA foreign_keys = ON")


def _relationship_profiles_schema() -> str:
    return """
    CREATE TABLE relationship_profiles (
      id INTEGER PRIMARY KEY AUTOINCREMENT,
      profile_name TEXT NOT NULL,
      person_source_id TEXT,
      line_speaker_source_id TEXT,
      relationship_label TEXT,
      label_source TEXT NOT NULL DEFAULT 'user_manual',
      valid_from TEXT,
      valid_to TEXT,
      visibility TEXT NOT NULL DEFAULT 'private',
      notes TEXT,
      created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
      updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
      CHECK (relationship_label IS NULL OR relationship_label IN ('partner', 'ex_partner', 'close_person', 'other_private')),
      CHECK (label_source = 'user_manual'),
      CHECK (visibility IN ('private', 'hidden'))
    )
    """
