from __future__ import annotations

from relationship_lifelog_agent.agent.agent_runtime import run_agent
from relationship_lifelog_agent.agent.observation_store import ObservationStore
from relationship_lifelog_agent.config import AnalysisSettings, PathSettings, RelationshipSettings, Settings
from relationship_lifelog_agent.db.repository import RelationshipRepository
from relationship_lifelog_agent.verification.full_answer_verifier import (
    verify_counts_against_python_facts,
    verify_dates_within_scope,
    verify_final_answer,
    verify_no_public_leaks,
    verify_no_unsupported_relationship_assertions,
    verify_source_refs_exist,
)


def test_verifier_detects_missing_source_ref() -> None:
    observations = ObservationStore().add(
        task_id="analyze_single_context",
        observation_type="evidence",
        summary="known source",
        source_refs=("line:known",),
    )

    issues = verify_source_refs_exist("根拠: source_ref: line:missing", observations=observations)

    assert issues
    assert issues[0].code == "missing_source_ref"
    assert issues[0].severity == "major"


def test_verifier_detects_out_of_scope_date() -> None:
    issues = verify_dates_within_scope("2025-02-01 に候補があります。", "2025-01-01", "2025-01-31")

    assert issues
    assert issues[0].code == "date_out_of_scope"


def test_verifier_detects_count_mismatch() -> None:
    issues = verify_counts_against_python_facts(
        "- conflict_candidates: 2\n- minor_misunderstanding_candidates: 1",
        {"conflict_candidates": 1, "minor_misunderstanding_candidates": 1},
    )

    assert len(issues) == 1
    assert issues[0].code == "count_mismatch"
    assert "Python computed 1" in issues[0].message


def test_verifier_detects_public_raw_information() -> None:
    issues = verify_no_public_leaks(
        "LINE全文: これは出してはいけない本文です\nsource_path: /home/private/photo.jpg",
        mode="public",
    )

    codes = {issue.code for issue in issues}
    assert "public_line_full_text" in codes or "raw_data_leak" in codes
    assert "public_private_path" in codes or "raw_data_leak" in codes


def test_verifier_detects_unsupported_relationship_assertions() -> None:
    issues = verify_no_unsupported_relationship_assertions("確実に喧嘩していたし、この人は恋人です。")

    codes = {issue.code for issue in issues}
    assert "unsupported_relationship_assertion" in codes


def test_verification_report_is_reflected_in_corrected_answer() -> None:
    report = verify_final_answer(
        "要約:\n確実に喧嘩していた。\n\n根拠:\nsource_ref: line:missing",
        observations=ObservationStore(),
        evidence_refs=(),
        mode="private",
    )

    assert report.ok is False
    assert report.corrected_answer is not None
    assert "検証:" in report.corrected_answer
    assert "unsupported_relationship_assertion" in report.corrected_answer
    assert "missing_source_ref" in report.corrected_answer or "unverifiable_source_ref" in report.corrected_answer


def test_runtime_final_answer_contains_verification_report(tmp_path) -> None:
    settings = _private_full_settings(tmp_path)

    run = run_agent("いおりとの記録を全体から見て", None, settings)

    assert "検証:" in run.answer_markdown
    assert run.debug_info["verification"]["ok"] is True


def _private_full_settings(tmp_path) -> Settings:
    db_path = tmp_path / "relationship.sqlite"
    repo = RelationshipRepository(db_path)
    profile_id = repo.create_profile(
        "いおり",
        person_source_id="plr:person:target",
        line_speaker_source_id="plr:line_speaker:target",
        self_line_speaker_source_id="plr:line_speaker:self",
        relationship_label="partner",
    )
    return Settings(
        analysis=AnalysisSettings(mode="private_full_range"),
        paths=PathSettings(relationship_db=str(db_path)),
        relationship=RelationshipSettings(default_profile_id=profile_id),
    )
