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
        relationship_label="partner",
        valid_from="2024-01-01",
    )
    saved = repo.get_profile(profile_id)

    assert saved is not None
    assert saved["label_source"] == "user_manual"
    assert saved["relationship_label"] == "partner"
    assert repo.list_profiles()[0]["id"] == profile_id

    assert repo.update_profile(profile_id, valid_to="2025-12-31") == 1
    assert repo.get_profile(profile_id)["valid_to"] == "2025-12-31"


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

    cli_main(["--config", str(config_path), "profile", "show", "--id", "1"])
    show_output = capsys.readouterr().out
    assert "relationship_label: partner" in show_output
    assert "label_source: user_manual" in show_output

    cli_main(["--config", str(config_path), "profile", "update", "--id", "1", "--valid-to", "2025-12-31"])
    assert "updated profile id=1" in capsys.readouterr().out
    repo = RelationshipRepository(tmp_path / "relationship.sqlite")
    assert repo.get_profile(1)["valid_to"] == "2025-12-31"


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


def _write_config(tmp_path) -> str:
    config_path = tmp_path / "config.yaml"
    db_path = tmp_path / "relationship.sqlite"
    config_path.write_text(
        "paths:\n"
        f"  relationship_db: \"{db_path}\"\n",
        encoding="utf-8",
    )
    return str(config_path)
