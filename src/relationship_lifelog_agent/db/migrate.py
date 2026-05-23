from __future__ import annotations

from importlib import resources
from pathlib import Path
import sqlite3


RELATIONSHIP_PROFILE_EXTRA_COLUMNS = {
    "line_speaker_group_source_id": "TEXT",
    "self_person_source_id": "TEXT",
    "self_line_speaker_source_id": "TEXT",
    "self_line_speaker_group_source_id": "TEXT",
}


def initialize_database(db_path: str | Path) -> Path:
    path = Path(db_path).expanduser()
    path.parent.mkdir(parents=True, exist_ok=True)
    schema = resources.files("relationship_lifelog_agent.db").joinpath("schema.sql").read_text(encoding="utf-8")
    with sqlite3.connect(path) as conn:
        conn.execute("PRAGMA foreign_keys = ON")
        table_schema, trigger_schema = _split_trigger_schema(schema)
        conn.executescript(table_schema)
        _ensure_relationship_profiles_contract(conn)
        _ensure_relationship_events_contract(conn)
        _ensure_relationship_event_evidence_contract(conn)
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


def _ensure_relationship_events_contract(conn: sqlite3.Connection) -> None:
    row = conn.execute(
        "SELECT sql FROM sqlite_master WHERE type = 'table' AND name = 'relationship_events'"
    ).fetchone()
    if row is None:
        return
    sql = row[0] or ""
    if (
        "status TEXT" in sql
        and "review_status TEXT NOT NULL DEFAULT 'unreviewed'" in sql
        and "needs_reanalysis" in sql
    ):
        return

    columns = {item[1] for item in conn.execute("PRAGMA table_info(relationship_events)")}
    conn.execute("PRAGMA foreign_keys = OFF")
    conn.execute(_relationship_events_schema("relationship_events_new"))
    select_values = {
        "id": "id" if "id" in columns else "NULL",
        "profile_id": "profile_id" if "profile_id" in columns else "NULL",
        "event_type": (
            "CASE WHEN event_type IN ("
            "'conflict','minor_misunderstanding','reconciliation','promise','date_or_outing',"
            "'reply_delay','call_event','emotional_note','affection_note','important_memory',"
            "'anniversary','unresolved_issue','other') THEN event_type ELSE 'other' END"
            if "event_type" in columns
            else "'other'"
        ),
        "event_date": "event_date" if "event_date" in columns else "CURRENT_DATE",
        "summary": "summary" if "summary" in columns else "'relationship event candidate'",
        "status": (
            "CASE WHEN status IN ('candidate','hidden','archived') THEN status ELSE 'candidate' END"
            if "status" in columns
            else "'candidate'"
        ),
        "review_status": (
            "CASE "
            "WHEN review_status IN ('unreviewed','verified','corrected','needs_reanalysis','rejected') THEN review_status "
            "WHEN review_status = 'candidate' THEN 'unreviewed' "
            "ELSE 'unreviewed' END"
            if "review_status" in columns
            else "'unreviewed'"
        ),
        "confidence": (
            "CASE WHEN confidence BETWEEN 0.0 AND 1.0 THEN confidence ELSE 0.5 END"
            if "confidence" in columns
            else "0.5"
        ),
        "evidence_strength": (
            "CASE WHEN evidence_strength BETWEEN 0.0 AND 1.0 THEN evidence_strength ELSE 0.5 END"
            if "evidence_strength" in columns
            else "0.5"
        ),
        "severity": (
            "CASE WHEN severity BETWEEN 0 AND 4 THEN severity ELSE 0 END"
            if "severity" in columns
            else "0"
        ),
        "generated_by_model": "generated_by_model" if "generated_by_model" in columns else "NULL",
        "prompt_version": "prompt_version" if "prompt_version" in columns else "NULL",
        "created_at": "created_at" if "created_at" in columns else "CURRENT_TIMESTAMP",
        "updated_at": "updated_at" if "updated_at" in columns else "CURRENT_TIMESTAMP",
    }
    conn.execute(
        """
        INSERT INTO relationship_events_new (
          id,
          profile_id,
          event_type,
          event_date,
          summary,
          status,
          review_status,
          confidence,
          evidence_strength,
          severity,
          generated_by_model,
          prompt_version,
          created_at,
          updated_at
        )
        SELECT
          {id},
          {profile_id},
          {event_type},
          {event_date},
          {summary},
          {status},
          {review_status},
          {confidence},
          {evidence_strength},
          {severity},
          {generated_by_model},
          {prompt_version},
          {created_at},
          {updated_at}
        FROM relationship_events
        """.format(**select_values)
    )
    conn.execute("DROP TABLE relationship_events")
    conn.execute("ALTER TABLE relationship_events_new RENAME TO relationship_events")
    conn.execute("PRAGMA foreign_keys = ON")


def _ensure_relationship_event_evidence_contract(conn: sqlite3.Connection) -> None:
    columns = {row[1] for row in conn.execute("PRAGMA table_info(relationship_event_evidence)")}
    if not columns:
        return
    if "source_pointer" not in columns:
        conn.execute("ALTER TABLE relationship_event_evidence ADD COLUMN source_pointer TEXT")
        conn.execute(
            """
            UPDATE relationship_event_evidence
            SET source_pointer = source_type || ':' || source_id
            WHERE source_pointer IS NULL
            """
        )
    if "evidence_strength" not in columns:
        conn.execute("ALTER TABLE relationship_event_evidence ADD COLUMN evidence_strength REAL DEFAULT 0.5")
        conn.execute(
            """
            UPDATE relationship_event_evidence
            SET evidence_strength = COALESCE(confidence, 0.5)
            WHERE evidence_strength IS NULL
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
        _ensure_relationship_profile_extra_columns(conn)
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
        "line_speaker_group_source_id": (
            "line_speaker_group_source_id" if "line_speaker_group_source_id" in columns else "NULL"
        ),
        "self_person_source_id": "self_person_source_id" if "self_person_source_id" in columns else "NULL",
        "self_line_speaker_source_id": (
            "self_line_speaker_source_id" if "self_line_speaker_source_id" in columns else "NULL"
        ),
        "self_line_speaker_group_source_id": (
            "self_line_speaker_group_source_id" if "self_line_speaker_group_source_id" in columns else "NULL"
        ),
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
          line_speaker_group_source_id,
          self_person_source_id,
          self_line_speaker_source_id,
          self_line_speaker_group_source_id,
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
          {line_speaker_group_source_id},
          {self_person_source_id},
          {self_line_speaker_source_id},
          {self_line_speaker_group_source_id},
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


def _ensure_relationship_profile_extra_columns(conn: sqlite3.Connection) -> None:
    columns = {item[1] for item in conn.execute("PRAGMA table_info(relationship_profiles)")}
    for column, column_type in RELATIONSHIP_PROFILE_EXTRA_COLUMNS.items():
        if column not in columns:
            conn.execute(f"ALTER TABLE relationship_profiles ADD COLUMN {column} {column_type}")


def _relationship_profiles_schema() -> str:
    return """
    CREATE TABLE relationship_profiles (
      id INTEGER PRIMARY KEY AUTOINCREMENT,
      profile_name TEXT NOT NULL,
      person_source_id TEXT,
      line_speaker_source_id TEXT,
      line_speaker_group_source_id TEXT,
      self_person_source_id TEXT,
      self_line_speaker_source_id TEXT,
      self_line_speaker_group_source_id TEXT,
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


def _relationship_events_schema(table_name: str) -> str:
    return f"""
    CREATE TABLE {table_name} (
      id INTEGER PRIMARY KEY AUTOINCREMENT,
      profile_id INTEGER,
      event_type TEXT NOT NULL,
      event_date TEXT NOT NULL,
      summary TEXT NOT NULL,
      status TEXT NOT NULL DEFAULT 'candidate',
      review_status TEXT NOT NULL DEFAULT 'unreviewed',
      confidence REAL NOT NULL DEFAULT 0.5,
      evidence_strength REAL NOT NULL DEFAULT 0.5,
      severity INTEGER NOT NULL DEFAULT 0,
      generated_by_model TEXT,
      prompt_version TEXT,
      created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
      updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
      FOREIGN KEY (profile_id) REFERENCES relationship_profiles(id) ON DELETE SET NULL,
      CHECK (event_type IN (
        'conflict',
        'minor_misunderstanding',
        'reconciliation',
        'promise',
        'date_or_outing',
        'reply_delay',
        'call_event',
        'emotional_note',
        'affection_note',
        'important_memory',
        'anniversary',
        'unresolved_issue',
        'other'
      )),
      CHECK (status IN ('candidate', 'hidden', 'archived')),
      CHECK (review_status IN ('unreviewed', 'verified', 'corrected', 'needs_reanalysis', 'rejected')),
      CHECK (confidence >= 0.0 AND confidence <= 1.0),
      CHECK (evidence_strength >= 0.0 AND evidence_strength <= 1.0),
      CHECK (severity >= 0 AND severity <= 4)
    )
    """
