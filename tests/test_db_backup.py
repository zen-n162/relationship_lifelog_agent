import sqlite3

from relationship_lifelog_agent.cli import main as cli_main
from relationship_lifelog_agent.db.backup import list_relationship_db_backups
from relationship_lifelog_agent.db.repository import RelationshipRepository


def test_db_backup_list_and_restore_cli(tmp_path, capsys) -> None:
    config_path = _write_config(tmp_path)
    db_path = tmp_path / "relationship.sqlite"
    repo = RelationshipRepository(db_path)
    profile_id = repo.create_profile("Aさん", relationship_label="partner")

    cli_main(["--config", str(config_path), "db", "backup"])
    backup_output = capsys.readouterr().out
    backups = list_relationship_db_backups(db_path)
    assert len(backups) == 1
    assert "relationship DB backup created: [redacted_path]" in backup_output
    assert str(backups[0].path) not in backup_output

    repo.create_event(
        profile_id=profile_id,
        event_type="conflict",
        event_date="2025-01-10",
        summary="喧嘩候補: テスト用summary",
    )
    assert _event_count(db_path) == 1

    cli_main(["--config", str(config_path), "db", "backups"])
    list_output = capsys.readouterr().out
    assert backups[0].filename in list_output
    assert str(backups[0].path.parent) not in list_output
    assert "backup directory: [redacted_path]" in list_output

    cli_main(["--config", str(config_path), "db", "restore", "--backup-path", str(backups[0].path)])
    restore_output = capsys.readouterr().out
    assert "relationship DB restored from: [redacted_path]" in restore_output
    assert str(backups[0].path) not in restore_output
    assert _event_count(db_path) == 0


def _write_config(tmp_path) -> str:
    config_path = tmp_path / "config.yaml"
    db_path = tmp_path / "relationship.sqlite"
    config_path.write_text(
        "paths:\n"
        f"  relationship_db: \"{db_path}\"\n",
        encoding="utf-8",
    )
    return str(config_path)


def _event_count(db_path) -> int:
    with sqlite3.connect(db_path) as conn:
        return int(conn.execute("SELECT COUNT(*) FROM relationship_events").fetchone()[0])
