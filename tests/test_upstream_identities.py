from __future__ import annotations

from pathlib import Path
import sqlite3

from relationship_lifelog_agent.cli import main as cli_main
from relationship_lifelog_agent.upstream_identities import (
    render_identities_json,
    render_identities_markdown,
    run_upstream_identities,
)


def test_mock_backend_identities_are_safe() -> None:
    report = run_upstream_identities(backend="mock", kind="all", privacy_level="redacted")
    rendered = render_identities_markdown(report)

    assert report.status == "OK"
    assert report.people
    assert report.line_speakers
    assert "人物A" in rendered
    assert "話者A" in rendered
    assert "raw_data_included: false" in rendered
    assert "恋人候補" not in rendered


def test_upstream_unconfigured_identities_warn_safely(tmp_path) -> None:
    config_path = _write_config(tmp_path)

    report = run_upstream_identities(
        config_path=config_path,
        backend="upstream_readonly",
        kind="all",
        privacy_level="redacted",
    )
    rendered = render_identities_json(report)

    assert report.status == "WARN"
    assert "not configured" in rendered
    assert str(tmp_path) not in rendered


def test_upstream_people_redacted_hides_manual_names_and_paths(tmp_path) -> None:
    personal_db = tmp_path / "personal.sqlite"
    _create_personal_db(personal_db)
    config_path = _write_config(tmp_path, personal_db=personal_db)

    report = run_upstream_identities(
        config_path=config_path,
        backend="upstream_readonly",
        kind="people",
        privacy_level="redacted",
    )
    rendered = render_identities_markdown(report)

    assert report.status == "OK"
    assert "人物A" in rendered
    assert "Private Person" not in rendered
    assert "secret/photo.jpg" not in rendered
    assert str(tmp_path) not in rendered
    assert "plr:person:1" in rendered
    assert "恋人候補" not in rendered


def test_upstream_line_speakers_redacted_hides_display_names_and_raw_text(tmp_path) -> None:
    personal_db = tmp_path / "personal.sqlite"
    _create_personal_db(personal_db)
    config_path = _write_config(tmp_path, personal_db=personal_db)

    report = run_upstream_identities(
        config_path=config_path,
        backend="upstream_readonly",
        kind="line-speakers",
        privacy_level="redacted",
    )
    rendered = render_identities_json(report)

    assert report.status == "OK"
    assert "話者A" in rendered
    assert "Private Speaker" not in rendered
    assert "raw LINE body must not leak" not in rendered
    assert str(tmp_path) not in rendered
    assert "plr:line_speaker:10" in rendered


def test_upstream_identities_private_names_but_no_raw_body_or_paths(tmp_path) -> None:
    personal_db = tmp_path / "personal.sqlite"
    _create_personal_db(personal_db)
    config_path = _write_config(tmp_path, personal_db=personal_db)

    report = run_upstream_identities(
        config_path=config_path,
        backend="upstream_readonly",
        kind="all",
        privacy_level="private",
    )
    rendered = render_identities_markdown(report)

    assert "Private Person" in rendered
    assert "Private Speaker" in rendered
    assert "raw LINE body must not leak" not in rendered
    assert "secret/photo.jpg" not in rendered
    assert "35.123456" not in rendered
    assert "恋人候補" not in rendered


def test_upstream_identities_cli_writes_report_with_redacted_path(tmp_path, capsys) -> None:
    personal_db = tmp_path / "personal.sqlite"
    _create_personal_db(personal_db)
    config_path = _write_config(tmp_path, personal_db=personal_db)
    output_path = tmp_path / "exports" / "identities.md"

    cli_main(
        [
            "--config",
            str(config_path),
            "upstream",
            "identities",
            "--backend",
            "upstream_readonly",
            "--kind",
            "all",
            "--privacy-level",
            "redacted",
            "--format",
            "markdown",
            "--output",
            str(output_path),
        ]
    )
    captured = capsys.readouterr()

    assert output_path.exists()
    rendered = output_path.read_text(encoding="utf-8")
    assert "Upstream Identity Inventory" in rendered
    assert "Private Person" not in rendered
    assert "Private Speaker" not in rendered
    assert str(output_path) not in captured.out
    assert "[redacted_path]" in captured.out


def _write_config(
    tmp_path: Path,
    *,
    personal_db: Path | None = None,
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
                "  relationship_db: null",
                f"  personal_lifelog_db: {_yaml_nullable(personal_db)}",
                "  notes_lifelog_db: null",
            ]
        ),
        encoding="utf-8",
    )
    return config_path


def _create_personal_db(path: Path) -> None:
    conn = sqlite3.connect(path)
    conn.execute(
        "CREATE TABLE persons (id INTEGER PRIMARY KEY, display_name TEXT, manual_verified INTEGER, hidden INTEGER, deleted_at TEXT)"
    )
    conn.execute("INSERT INTO persons VALUES (?, ?, ?, ?, ?)", (1, "Private Person", 1, 0, None))
    conn.execute(
        "CREATE TABLE media_people (media_id INTEGER, person_id INTEGER, hidden INTEGER)"
    )
    conn.execute("INSERT INTO media_people VALUES (?, ?, ?)", (100, 1, 0))
    conn.execute(
        "CREATE TABLE media_items (id INTEGER PRIMARY KEY, captured_at TEXT, file_path TEXT, gps_lat REAL, gps_lon REAL)"
    )
    conn.execute(
        "INSERT INTO media_items VALUES (?, ?, ?, ?, ?)",
        (100, "2025-01-02", "secret/photo.jpg", 35.123456, 139.123456),
    )
    conn.execute(
        "CREATE TABLE line_speaker_links (id INTEGER PRIMARY KEY, chat_id TEXT, speaker_name TEXT, verified_by_user INTEGER)"
    )
    conn.execute("INSERT INTO line_speaker_links VALUES (?, ?, ?, ?)", (10, "chat-1", "Private Speaker", 1))
    conn.execute(
        "CREATE TABLE line_messages (id INTEGER PRIMARY KEY, chat_id TEXT, sender TEXT, text TEXT, sent_at TEXT)"
    )
    conn.execute(
        "INSERT INTO line_messages VALUES (?, ?, ?, ?, ?)",
        (500, "chat-1", "Private Speaker", "raw LINE body must not leak", "2025-01-03"),
    )
    conn.commit()
    conn.close()


def _yaml_nullable(value: Path | None) -> str:
    if value is None:
        return "null"
    return str(value)
