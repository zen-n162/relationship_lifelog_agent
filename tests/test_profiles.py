import pytest

from relationship_lifelog_agent.agent.executor import answer_question
from relationship_lifelog_agent.cli import main as cli_main
from relationship_lifelog_agent.config import PathSettings, RelationshipSettings, Settings
from relationship_lifelog_agent.db.repository import RelationshipRepository


def test_profile_repository_create_list_show_update(tmp_path) -> None:
    repo = RelationshipRepository(tmp_path / "relationship.sqlite")

    profile_id = repo.create_profile(
        profile_name="Aさん",
        person_source_id="plr:person:4",
        line_speaker_source_id="plr:line_speaker:2",
        line_speaker_group_source_id="plr:line_speaker_group:target",
        self_person_source_id="plr:person:self",
        self_line_speaker_source_id="plr:line_speaker:self",
        self_line_speaker_group_source_id="plr:line_speaker_group:self",
        relationship_label="partner",
        valid_from="2024-01-01",
    )
    saved = repo.get_profile(profile_id)

    assert saved is not None
    assert saved["label_source"] == "user_manual"
    assert saved["relationship_label"] == "partner"
    assert saved["line_speaker_group_source_id"] == "plr:line_speaker_group:target"
    assert saved["self_person_source_id"] == "plr:person:self"
    assert saved["self_line_speaker_source_id"] == "plr:line_speaker:self"
    assert saved["self_line_speaker_group_source_id"] == "plr:line_speaker_group:self"
    assert repo.list_profiles()[0]["id"] == profile_id

    assert repo.update_profile(profile_id, valid_to="2025-12-31", self_person_source_id="plr:person:self-updated") == 1
    assert repo.get_profile(profile_id)["valid_to"] == "2025-12-31"
    assert repo.get_profile(profile_id)["self_person_source_id"] == "plr:person:self-updated"


def test_profile_repository_rejects_invalid_relationship_label(tmp_path) -> None:
    repo = RelationshipRepository(tmp_path / "relationship.sqlite")

    with pytest.raises(ValueError):
        repo.create_profile("invalid", relationship_label="girlfriend")

    with pytest.raises(ValueError):
        repo.create_profile("invalid", relationship_label="partner", label_source="ai_generated")


def test_profile_cli_create_list_show_update(tmp_path, capsys) -> None:
    config_path = _write_config(tmp_path)

    cli_main(
        [
            "--config",
            str(config_path),
            "profile",
            "create",
            "--profile-name",
            "Aさん",
            "--person-source-id",
            "plr:person:4",
            "--line-speaker-source-id",
            "plr:line_speaker:2",
            "--line-speaker-group-source-id",
            "plr:line_speaker_group:target",
            "--self-person-source-id",
            "plr:person:self",
            "--self-line-speaker-source-id",
            "plr:line_speaker:self",
            "--self-line-speaker-group-source-id",
            "plr:line_speaker_group:self",
            "--relationship-label",
            "partner",
            "--valid-from",
            "2024-01-01",
        ]
    )
    assert "created profile id=1" in capsys.readouterr().out

    cli_main(["--config", str(config_path), "profile", "list"])
    list_output = capsys.readouterr().out
    assert "Aさん" in list_output
    assert "user_manual" in list_output
    assert "self_line_speaker_group_source_id" in list_output
    assert "plr:line_speaker_group:self" in list_output

    cli_main(["--config", str(config_path), "profile", "show", "--id", "1"])
    show_output = capsys.readouterr().out
    assert "relationship_label: partner" in show_output
    assert "label_source: user_manual" in show_output
    assert "self_person_source_id: plr:person:self" in show_output
    assert "line_speaker_group_source_id: plr:line_speaker_group:target" in show_output

    cli_main(["--config", str(config_path), "profile", "update", "--id", "1", "--valid-to", "2025-12-31"])
    assert "updated profile id=1" in capsys.readouterr().out
    repo = RelationshipRepository(tmp_path / "relationship.sqlite")
    assert repo.get_profile(1)["valid_to"] == "2025-12-31"

    cli_main(
        [
            "--config",
            str(config_path),
            "profile",
            "update",
            "--profile-name",
            "Aさん",
            "--self-person-source-id",
            "plr:person:self-updated",
        ]
    )
    assert "updated profile id=1" in capsys.readouterr().out
    assert repo.get_profile(1)["self_person_source_id"] == "plr:person:self-updated"


def test_profile_missing_returns_safe_guidance(tmp_path) -> None:
    settings = Settings(paths=PathSettings(relationship_db=str(tmp_path / "relationship.sqlite")))

    answer = answer_question("喧嘩はどのくらいしている？", settings=settings)

    assert "relationship profile が手動設定されていない" in answer
    assert "AIは恋人ラベルを推定しません" in answer
    assert "AIはpersonとLINE speakerを自動リンクしません" in answer


def test_public_mode_does_not_expose_relationship_label(tmp_path) -> None:
    db_path = tmp_path / "relationship.sqlite"
    repo = RelationshipRepository(db_path)
    repo.create_profile("Aさん", relationship_label="partner")
    settings = Settings(
        paths=PathSettings(relationship_db=str(db_path)),
        relationship=RelationshipSettings(default_profile_id=1),
    )

    answer = answer_question("喧嘩はどのくらいしている？", mode="public", settings=settings)

    assert "partner" not in answer
    assert "relationship_label" not in answer
    assert "label_source" not in answer


def test_profile_configured_named_question_does_not_stop_at_setup_guidance(tmp_path) -> None:
    db_path = tmp_path / "relationship.sqlite"
    repo = RelationshipRepository(db_path)
    repo.create_profile(
        "いおり",
        relationship_label="partner",
        person_source_id="plr:person:target",
        line_speaker_source_id="plr:line_speaker:target",
        self_person_source_id="plr:person:self",
        self_line_speaker_source_id="plr:line_speaker:self",
    )
    settings = Settings(
        paths=PathSettings(relationship_db=str(db_path)),
        relationship=RelationshipSettings(default_profile_id=1),
    )

    answer = answer_question("いおりとの喧嘩はどれくらいしている？", settings=settings)

    assert "relationship profile が手動設定されていない" not in answer
    assert "profile_configured: True" in answer


def test_reply_delay_warns_when_self_speaker_is_missing(tmp_path) -> None:
    db_path = tmp_path / "relationship.sqlite"
    repo = RelationshipRepository(db_path)
    repo.create_profile(
        "Aさん",
        relationship_label="partner",
        person_source_id="plr:person:target",
        line_speaker_source_id="plr:line_speaker:target",
    )
    settings = Settings(
        paths=PathSettings(relationship_db=str(db_path)),
        relationship=RelationshipSettings(default_profile_id=1),
    )

    answer = answer_question("LINE返せていなかった時期は？", settings=settings)

    assert "self_line_speaker_source_id または self_line_speaker_group_source_id" in answer


def _write_config(tmp_path) -> str:
    config_path = tmp_path / "config.yaml"
    db_path = tmp_path / "relationship.sqlite"
    config_path.write_text(
        "paths:\n"
        f"  relationship_db: \"{db_path}\"\n",
        encoding="utf-8",
    )
    return str(config_path)
