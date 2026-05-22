from __future__ import annotations

from pathlib import Path
import sqlite3

from relationship_lifelog_agent.cli import main as cli_main
from relationship_lifelog_agent.upstream_inspect import (
    render_inspection_json,
    render_inspection_markdown,
    run_upstream_inspection,
)


def test_mock_schema_inspection_reports_supported_coverage() -> None:
    report = run_upstream_inspection(backend="mock")

    assert report.status == "OK"
    assert all(database.connection_status == "mock" for database in report.databases)
    methods = {
        coverage.adapter_method: coverage.status
        for database in report.databases
        for coverage in database.method_coverage
    }
    assert methods["search_line"] == "supported"
    assert methods["search_notes"] == "supported"
    assert "LINE message body" not in render_inspection_markdown(report)


def test_upstream_unconfigured_schema_inspection_warns_safely(tmp_path) -> None:
    config_path = _write_config(tmp_path)

    report = run_upstream_inspection(config_path=config_path, backend="upstream_readonly")

    assert report.status == "WARN"
    assert all(database.status == "WARN" for database in report.databases)
    assert any("not configured" in warning for warning in report.warnings)


def test_schema_inspection_reads_sqlite_schema_counts_only(tmp_path) -> None:
    personal_db = tmp_path / "personal.sqlite"
    notes_db = tmp_path / "notes.sqlite"
    _create_personal_db(personal_db)
    _create_notes_db(notes_db)
    config_path = _write_config(tmp_path, personal_db=personal_db, notes_db=notes_db)

    report = run_upstream_inspection(config_path=config_path, backend="upstream_readonly")
    rendered = render_inspection_json(report)

    assert report.status in {"OK", "WARN"}
    assert '"line_messages":' not in rendered
    assert "private LINE text" not in rendered
    assert "private note body" not in rendered
    assert str(tmp_path) not in rendered

    personal = next(database for database in report.databases if database.app_name == "personal_lifelog_rag")
    line_mapping = next(mapping for mapping in personal.table_mappings if mapping.key == "line_messages")
    assert line_mapping.table_exists is True
    assert line_mapping.row_count == 1
    assert line_mapping.mapping_status == "supported"
    assert "text" in line_mapping.columns


def test_schema_inspection_private_path_redaction_and_no_raw_data(tmp_path) -> None:
    personal_db = tmp_path / "secret" / "personal.sqlite"
    personal_db.parent.mkdir(parents=True)
    _create_personal_db(personal_db)
    config_path = _write_config(tmp_path / "secret", personal_db=personal_db)

    report = run_upstream_inspection(config_path=config_path, backend="upstream_readonly")
    rendered = render_inspection_markdown(report)

    assert str(tmp_path) not in rendered
    assert "private LINE text" not in rendered
    assert "line_messages" in rendered
    assert "row_count: 1" in rendered


def test_upstream_inspect_cli_writes_markdown_output(tmp_path, capsys) -> None:
    output_path = tmp_path / "exports" / "inspection.md"

    cli_main(["upstream", "inspect", "--backend", "mock", "--format", "markdown", "--output", str(output_path)])
    captured = capsys.readouterr()

    assert output_path.exists()
    assert "Upstream Schema Inspection" in output_path.read_text(encoding="utf-8")
    assert str(output_path) not in captured.out
    assert "[redacted_path]" in captured.out


def _write_config(
    tmp_path: Path,
    *,
    personal_db: Path | None = None,
    notes_db: Path | None = None,
) -> Path:
    tmp_path.mkdir(parents=True, exist_ok=True)
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
                "  relationship_db: null",
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
        ("line-1", "2025-01-01", "private LINE text must not leak", "speaker", "chat"),
    )
    conn.execute("CREATE TABLE media_items (id TEXT PRIMARY KEY, captured_at TEXT, caption TEXT, ocr_text TEXT)")
    conn.execute("CREATE TABLE events (id TEXT PRIMARY KEY, date TEXT, title TEXT, summary TEXT)")
    conn.commit()
    conn.close()


def _create_notes_db(path: Path) -> None:
    conn = sqlite3.connect(path)
    conn.execute("CREATE TABLE notes (id TEXT PRIMARY KEY, note_date TEXT, title TEXT, body TEXT)")
    conn.execute("INSERT INTO notes VALUES (?, ?, ?, ?)", ("note-1", "2025-01-01", "title", "private note body"))
    conn.execute("CREATE TABLE thoughts (id TEXT PRIMARY KEY, date TEXT, summary TEXT)")
    conn.execute("CREATE TABLE monthly_reflections (month TEXT PRIMARY KEY, summary TEXT)")
    conn.commit()
    conn.close()


def _yaml_nullable(value: Path | None) -> str:
    if value is None:
        return "null"
    return str(value)
