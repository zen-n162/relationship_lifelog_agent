from __future__ import annotations

from pathlib import Path
import sqlite3

from relationship_lifelog_agent.cli import main as cli_main
from relationship_lifelog_agent.db.repository import RelationshipRepository
from relationship_lifelog_agent.upstream_smoke import render_smoke_json, render_smoke_markdown, run_upstream_smoke


def test_mock_backend_smoke_counts_without_raw_data(tmp_path) -> None:
    relationship_db = _relationship_db_with_profile(tmp_path)
    config_path = _write_config(tmp_path, relationship_db=relationship_db)

    report = run_upstream_smoke(
        config_path=config_path,
        backend="mock",
        date_from="2025-01-01",
        date_to="2025-01-31",
        profile_id=1,
    )
    rendered = render_smoke_markdown(report)

    assert report.status == "OK"
    assert report.profile.configured is True
    assert report.source_counts["line_evidence"] > 0
    assert report.source_counts["note_evidence"] > 0
    assert report.write_count == 0
    assert "raw_data_included: false" in rendered
    assert "予定の話" not in rendered


def test_upstream_unconfigured_smoke_warns_safely(tmp_path) -> None:
    relationship_db = _relationship_db_with_profile(tmp_path)
    config_path = _write_config(tmp_path, relationship_db=relationship_db)

    report = run_upstream_smoke(
        config_path=config_path,
        backend="upstream_readonly",
        date_from="2025-01-01",
        date_to="2025-01-31",
        profile_id=1,
    )

    assert report.status == "WARN"
    assert report.write_count == 0
    assert report.source_counts == {
        "line_evidence": 0,
        "media_evidence": 0,
        "event_evidence": 0,
        "note_evidence": 0,
        "thought_evidence": 0,
    }
    assert any("未設定" in warning for warning in report.warnings)


def test_upstream_smoke_reads_synthetic_dbs_counts_only(tmp_path) -> None:
    relationship_db = _relationship_db_with_profile(tmp_path)
    personal_db = tmp_path / "personal.sqlite"
    notes_db = tmp_path / "notes.sqlite"
    _create_personal_db(personal_db)
    _create_notes_db(notes_db)
    config_path = _write_config(
        tmp_path,
        relationship_db=relationship_db,
        personal_db=personal_db,
        notes_db=notes_db,
    )

    report = run_upstream_smoke(
        config_path=config_path,
        backend="upstream_readonly",
        date_from="2025-01-01",
        date_to="2025-01-31",
        profile_id=1,
    )
    rendered = render_smoke_json(report)

    assert report.write_count == 0
    assert report.source_counts["line_evidence"] == 1
    assert report.source_counts["media_evidence"] == 1
    assert report.source_counts["event_evidence"] == 1
    assert report.source_counts["note_evidence"] == 1
    assert report.source_counts["thought_evidence"] == 1
    assert report.monthly_reflection_available is True
    assert "private LINE body" not in rendered
    assert "private note body" not in rendered
    assert str(tmp_path) not in rendered


def test_upstream_smoke_cli_writes_report_with_redacted_path(tmp_path, capsys) -> None:
    relationship_db = _relationship_db_with_profile(tmp_path)
    config_path = _write_config(tmp_path, relationship_db=relationship_db)
    output_path = tmp_path / "exports" / "smoke.md"

    cli_main(
        [
            "--config",
            str(config_path),
            "upstream",
            "smoke",
            "--backend",
            "mock",
            "--date-from",
            "2025-01-01",
            "--date-to",
            "2025-01-31",
            "--profile-id",
            "1",
            "--output",
            str(output_path),
        ]
    )
    captured = capsys.readouterr()

    assert output_path.exists()
    assert "Upstream Read-only Smoke Report" in output_path.read_text(encoding="utf-8")
    assert str(output_path) not in captured.out
    assert "[redacted_path]" in captured.out


def _relationship_db_with_profile(tmp_path: Path) -> Path:
    db_path = tmp_path / "relationship.sqlite"
    repo = RelationshipRepository(db_path)
    repo.create_profile(
        "manual profile",
        relationship_label="partner",
        person_source_id="person-source",
        line_speaker_source_id="speaker-source",
    )
    return db_path


def _write_config(
    tmp_path: Path,
    *,
    relationship_db: Path,
    personal_db: Path | None = None,
    notes_db: Path | None = None,
) -> Path:
    config_path = tmp_path / "config.local.yaml"
    config_path.write_text(
        "\n".join(
            [
                "app:",
                "  host: 127.0.0.1",
                "  allow_external_api: false",
                "  allow_model_auto_download: false",
                "  allow_gradio_share: false",
                "adapter:",
                "  backend: upstream_readonly",
                "  upstream_access_mode: readonly",
                "  copy_raw_upstream_data: false",
                "paths:",
                f"  relationship_db: {relationship_db}",
                f"  personal_lifelog_db: {_yaml_nullable(personal_db)}",
                f"  notes_lifelog_db: {_yaml_nullable(notes_db)}",
            ]
        ),
        encoding="utf-8",
    )
    return config_path


def _create_personal_db(path: Path) -> None:
    conn = sqlite3.connect(path)
    conn.execute(
        "CREATE TABLE line_messages (id TEXT PRIMARY KEY, sent_at TEXT, text TEXT, sender TEXT, chat_id TEXT)"
    )
    conn.execute(
        "INSERT INTO line_messages VALUES (?, ?, ?, ?, ?)",
        ("line-1", "2025-01-05", "private LINE body with 喧嘩 and 謝罪", "speaker-source", "chat"),
    )
    conn.execute("CREATE TABLE media_items (id TEXT PRIMARY KEY, captured_at TEXT, caption TEXT, media_type TEXT)")
    conn.execute(
        "INSERT INTO media_items VALUES (?, ?, ?, ?)",
        ("media-1", "2025-01-07", "private photo path must not leak 外出", "photo"),
    )
    conn.execute("CREATE TABLE events (id TEXT PRIMARY KEY, date TEXT, title TEXT, summary TEXT)")
    conn.execute(
        "INSERT INTO events VALUES (?, ?, ?, ?)",
        ("event-1", "2025-01-08", "private event title", "private event summary"),
    )
    conn.commit()
    conn.close()


def _create_notes_db(path: Path) -> None:
    conn = sqlite3.connect(path)
    conn.execute("CREATE TABLE notes (id TEXT PRIMARY KEY, note_date TEXT, title TEXT, body TEXT)")
    conn.execute("INSERT INTO notes VALUES (?, ?, ?, ?)", ("note-1", "2025-01-09", "private title", "private note body 反省"))
    conn.execute("CREATE TABLE thoughts (id TEXT PRIMARY KEY, date TEXT, summary TEXT)")
    conn.execute("INSERT INTO thoughts VALUES (?, ?, ?)", ("thought-1", "2025-01-10", "private thought summary 不安"))
    conn.execute("CREATE TABLE monthly_reflections (month TEXT PRIMARY KEY, summary TEXT)")
    conn.execute("INSERT INTO monthly_reflections VALUES (?, ?)", ("2025-01", "private monthly body"))
    conn.commit()
    conn.close()


def _yaml_nullable(value: Path | None) -> str:
    if value is None:
        return "null"
    return str(value)
