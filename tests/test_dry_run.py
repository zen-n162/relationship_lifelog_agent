import sqlite3

from relationship_lifelog_agent.adapters.types import LineEvidence
from relationship_lifelog_agent.analytics.conflict import build_conflict_candidates
from relationship_lifelog_agent.cli import main as cli_main
from relationship_lifelog_agent.db.repository import RelationshipRepository
from relationship_lifelog_agent.privacy.guard import detect_answer_safety_violations


def test_mock_backend_dry_run_report_is_generated_without_writing_events(tmp_path, capsys) -> None:
    config_path = _write_config(tmp_path)
    repo = RelationshipRepository(tmp_path / "relationship.sqlite")
    profile_id = repo.create_profile("Aさん", relationship_label="partner")
    output_path = tmp_path / "dry_run.md"

    cli_main(
        [
            "--config",
            str(config_path),
            "analyze",
            "dry-run",
            "--profile-id",
            str(profile_id),
            "--date-from",
            "2025-01-01",
            "--date-to",
            "2025-03-31",
            "--backend",
            "mock",
            "--output",
            str(output_path),
        ]
    )

    assert "relationship_events written: 0" in capsys.readouterr().out
    report = output_path.read_text(encoding="utf-8")
    assert "Relationship Event Candidates Dry Run" in report
    assert "relationship DBには候補を書き込んでいません" in report
    assert "conflict candidates" in report
    assert repo.list_events() == []
    _assert_relationship_events_table_empty(tmp_path / "relationship.sqlite")


def test_weak_signal_only_does_not_create_conflict_candidate() -> None:
    weak_line = LineEvidence(
        source_id="weak-delay-only",
        date="2025-01-02",
        role="context",
        summary="返信遅延のみの弱い信号。",
        excerpt="返信が少し空いた。",
        confidence=0.4,
        evidence_strength=0.3,
        metadata={"tags": ("返信遅延", "弱い信号")},
    )

    candidates = build_conflict_candidates(line_evidence=[weak_line], note_evidence=[], event_evidence=[])

    assert candidates == []


def test_dry_run_report_has_no_forbidden_phrases(tmp_path) -> None:
    config_path = _write_config(tmp_path)
    profile_id = RelationshipRepository(tmp_path / "relationship.sqlite").create_profile("Aさん", relationship_label="partner")
    output_path = tmp_path / "dry_run.md"

    cli_main(
        [
            "--config",
            str(config_path),
            "analyze",
            "dry-run",
            "--profile-id",
            str(profile_id),
            "--date-from",
            "2025-01-01",
            "--date-to",
            "2025-03-31",
            "--backend",
            "mock",
            "--output",
            str(output_path),
        ]
    )

    report = output_path.read_text(encoding="utf-8")
    assert detect_answer_safety_violations(report) == []


def test_dry_run_public_mode_redacts_private_profile_and_excerpts(tmp_path) -> None:
    config_path = _write_config(tmp_path)
    profile_id = RelationshipRepository(tmp_path / "relationship.sqlite").create_profile(
        "山田太郎",
        person_source_id="plr:person:4",
        line_speaker_source_id="plr:line_speaker:2",
        relationship_label="partner",
    )
    output_path = tmp_path / "dry_run_public.md"

    cli_main(
        [
            "--config",
            str(config_path),
            "analyze",
            "dry-run",
            "--profile-id",
            str(profile_id),
            "--date-from",
            "2025-01-01",
            "--date-to",
            "2025-03-31",
            "--backend",
            "mock",
            "--mode",
            "public",
            "--output",
            str(output_path),
        ]
    )

    report = output_path.read_text(encoding="utf-8")
    assert "山田太郎" not in report
    assert "partner" not in report
    assert "relationship_label" not in report
    assert "excerpt:" not in report
    assert detect_answer_safety_violations(report, mode="public", person_names=["山田太郎"]) == []


def _write_config(tmp_path) -> str:
    config_path = tmp_path / "config.yaml"
    db_path = tmp_path / "relationship.sqlite"
    config_path.write_text(
        "paths:\n"
        f"  relationship_db: \"{db_path}\"\n",
        encoding="utf-8",
    )
    return str(config_path)


def _assert_relationship_events_table_empty(db_path) -> None:
    with sqlite3.connect(db_path) as conn:
        count = conn.execute("SELECT COUNT(*) FROM relationship_events").fetchone()[0]
    assert count == 0
