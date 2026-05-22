import sqlite3

import pytest

from relationship_lifelog_agent.adapters.types import LineEvidence
from relationship_lifelog_agent.analytics.conflict import build_conflict_candidates
from relationship_lifelog_agent.cli import main as cli_main
from relationship_lifelog_agent.db.backup import list_relationship_db_backups
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


def test_dry_run_counts_only_omits_candidate_body_and_source_pointers(tmp_path) -> None:
    config_path = _write_config(tmp_path)
    profile_id = RelationshipRepository(tmp_path / "relationship.sqlite").create_profile("Aさん", relationship_label="partner")
    output_path = tmp_path / "dry_run_counts.md"

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
            "--privacy-level",
            "counts-only",
            "--output",
            str(output_path),
        ]
    )

    report = output_path.read_text(encoding="utf-8")
    assert "privacy_level: counts-only" in report
    assert "予定変更への不満" not in report
    assert "予定の話が合わず" not in report
    assert "不安もあるが" not in report
    assert "mock-line-" not in report
    assert "[redacted_source_pointer]" not in report
    assert "candidate counts" in report
    assert detect_answer_safety_violations(report) == []


def test_dry_run_redacted_omits_profile_label_paths_and_excerpts(tmp_path, capsys) -> None:
    config_path = _write_config(tmp_path)
    profile_id = RelationshipRepository(tmp_path / "relationship.sqlite").create_profile(
        "山田太郎",
        person_source_id="/home/user/private/person.json",
        line_speaker_source_id="/home/user/private/line.json",
        relationship_label="partner",
    )
    output_path = tmp_path / "dry_run_redacted.md"

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
            "--privacy-level",
            "redacted",
            "--output",
            str(output_path),
        ]
    )

    captured = capsys.readouterr()
    report = output_path.read_text(encoding="utf-8")
    assert "dry-run report written: [redacted_path]" in captured.out
    assert str(output_path) not in captured.out
    assert "山田太郎" not in report
    assert "partner" not in report
    assert "/home/user/private" not in report
    assert "予定の話が合わず" not in report
    assert "[redacted_source_pointer]" in report
    assert detect_answer_safety_violations(report, mode="public", person_names=["山田太郎"]) == []


def test_dry_run_report_separates_reviewed_events_from_new_candidates(tmp_path) -> None:
    config_path = _write_config(tmp_path)
    repo = RelationshipRepository(tmp_path / "relationship.sqlite")
    profile_id = repo.create_profile("Aさん", relationship_label="partner")
    repo.create_event(
        profile_id=profile_id,
        event_type="conflict",
        event_date="2025-01-11",
        summary="人間が確認した保存済みの喧嘩候補。",
        status="candidate",
        review_status="verified",
    )
    repo.create_event(
        profile_id=profile_id,
        event_type="conflict",
        event_date="2025-01-12",
        summary="除外済みの保存候補。",
        status="candidate",
        review_status="rejected",
    )
    output_path = tmp_path / "dry_run_reviewed.md"

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
            "2025-01-31",
            "--backend",
            "mock",
            "--privacy-level",
            "redacted",
            "--output",
            str(output_path),
        ]
    )

    report = output_path.read_text(encoding="utf-8")
    assert "reviewed relationship DB events" in report
    assert "new conflict candidates" in report
    assert "人間確認済み件数: 1" in report
    assert "除外済み件数: 1" in report
    assert "人間が確認した保存済みの喧嘩候補" in report


def test_public_mode_rejects_private_privacy_level(tmp_path) -> None:
    config_path = _write_config(tmp_path)
    profile_id = RelationshipRepository(tmp_path / "relationship.sqlite").create_profile("Aさん", relationship_label="partner")

    with pytest.raises(SystemExit) as exc_info:
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
                "--privacy-level",
                "private",
            ]
        )

    assert "privacy-level private is not allowed in public mode" in str(exc_info.value)


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


def test_dry_run_write_saves_unreviewed_candidate_events(tmp_path, capsys) -> None:
    config_path = _write_config(tmp_path)
    profile_id = RelationshipRepository(tmp_path / "relationship.sqlite").create_profile("Aさん", relationship_label="partner")

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
            "--write",
        ]
    )

    output = capsys.readouterr().out
    backups = list_relationship_db_backups(tmp_path / "relationship.sqlite")
    assert len(backups) == 1
    assert "pre-write backup path: [redacted_path]" in output
    assert backups[0].filename in output
    assert str(backups[0].path) not in output
    assert "write safety candidate events: " in output
    assert "write safety raw leakage issues: 0" in output
    assert "write safety forbidden phrase issues: 0" in output
    assert "post-write raw leakage issues: 0" in output
    assert "post-write forbidden phrase issues: 0" in output
    assert "relationship_events written: " in output
    assert "relationship_events written: 0" not in output

    events = _relationship_event_rows(tmp_path / "relationship.sqlite")
    assert events
    assert {row["status"] for row in events} == {"candidate"}
    assert {row["review_status"] for row in events} == {"unreviewed"}
    assert "conflict" in {row["event_type"] for row in events}
    assert "emotional_note" in {row["event_type"] for row in events}

    evidence = _evidence_rows(tmp_path / "relationship.sqlite")
    assert evidence
    assert all(row["source_pointer"] for row in evidence)
    assert all(0.0 <= row["evidence_strength"] <= 1.0 for row in evidence)
    activities = _post_conflict_activity_rows(tmp_path / "relationship.sqlite")
    assert activities


def test_dry_run_write_prevents_duplicate_events(tmp_path, capsys) -> None:
    config_path = _write_config(tmp_path)
    profile_id = RelationshipRepository(tmp_path / "relationship.sqlite").create_profile("Aさん", relationship_label="partner")
    command = [
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
        "--write",
    ]

    cli_main(command)
    capsys.readouterr()
    first_event_count = len(_relationship_event_rows(tmp_path / "relationship.sqlite"))
    first_activity_count = len(_post_conflict_activity_rows(tmp_path / "relationship.sqlite"))

    cli_main(command)
    second_output = capsys.readouterr().out

    assert len(_relationship_event_rows(tmp_path / "relationship.sqlite")) == first_event_count
    assert len(_post_conflict_activity_rows(tmp_path / "relationship.sqlite")) == first_activity_count
    assert "relationship_events written: 0" in second_output
    assert "duplicates skipped: " in second_output
    assert "write safety preflight duplicate candidates: " in second_output


def test_dry_run_write_does_not_store_raw_full_text_or_long_excerpts(tmp_path) -> None:
    config_path = _write_config(tmp_path)
    profile_id = RelationshipRepository(tmp_path / "relationship.sqlite").create_profile("Aさん", relationship_label="partner")

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
            "--write",
        ]
    )

    texts: list[str] = []
    for row in [*_relationship_event_rows(tmp_path / "relationship.sqlite"), *_evidence_rows(tmp_path / "relationship.sqlite")]:
        texts.extend(str(value or "") for value in dict(row).values())
    joined = "\n".join(texts)
    assert "LINE全文" not in joined
    assert "メモ全文" not in joined
    assert "/home/" not in joined
    assert ".jpg" not in joined
    assert all(len(str(row["excerpt"] or "")) <= 120 for row in _evidence_rows(tmp_path / "relationship.sqlite"))


def test_dry_run_write_counts_only_omits_evidence_excerpts_and_source_summaries(tmp_path) -> None:
    config_path = _write_config(tmp_path)
    profile_id = RelationshipRepository(tmp_path / "relationship.sqlite").create_profile("Aさん", relationship_label="partner")

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
            "--privacy-level",
            "counts-only",
            "--write",
        ]
    )

    evidence = _evidence_rows(tmp_path / "relationship.sqlite")
    assert evidence
    assert all(row["excerpt"] is None for row in evidence)
    stored_text = "\n".join(
        str(value or "")
        for row in [*_relationship_event_rows(tmp_path / "relationship.sqlite"), *evidence]
        for value in dict(row).values()
    )
    assert "予定変更への不満" not in stored_text
    assert "予定の話が合わず" not in stored_text
    assert "不安もあるが" not in stored_text


def test_dry_run_write_private_level_allows_short_excerpts(tmp_path) -> None:
    config_path = _write_config(tmp_path)
    profile_id = RelationshipRepository(tmp_path / "relationship.sqlite").create_profile("Aさん", relationship_label="partner")

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
            "--privacy-level",
            "private",
            "--write",
        ]
    )

    evidence = _evidence_rows(tmp_path / "relationship.sqlite")
    excerpts = [row["excerpt"] for row in evidence if row["excerpt"]]
    assert excerpts
    assert all(len(excerpt) <= 120 for excerpt in excerpts)
    assert all("/home/" not in excerpt for excerpt in excerpts)


def test_dry_run_write_public_mode_omits_evidence_excerpts(tmp_path) -> None:
    config_path = _write_config(tmp_path)
    profile_id = RelationshipRepository(tmp_path / "relationship.sqlite").create_profile("山田太郎", relationship_label="partner")

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
            "--write",
        ]
    )

    evidence = _evidence_rows(tmp_path / "relationship.sqlite")
    assert evidence
    assert all(row["excerpt"] is None for row in evidence)
    stored_text = "\n".join(
        str(value or "")
        for row in [*_relationship_event_rows(tmp_path / "relationship.sqlite"), *evidence]
        for value in dict(row).values()
    )
    assert "山田太郎" not in stored_text
    assert "partner" not in stored_text


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


def _relationship_event_rows(db_path) -> list[sqlite3.Row]:
    return _rows(db_path, "SELECT * FROM relationship_events ORDER BY id")


def _evidence_rows(db_path) -> list[sqlite3.Row]:
    return _rows(db_path, "SELECT * FROM relationship_event_evidence ORDER BY id")


def _post_conflict_activity_rows(db_path) -> list[sqlite3.Row]:
    return _rows(db_path, "SELECT * FROM post_conflict_activities ORDER BY id")


def _rows(db_path, sql: str) -> list[sqlite3.Row]:
    with sqlite3.connect(db_path) as conn:
        conn.row_factory = sqlite3.Row
        return conn.execute(sql).fetchall()
