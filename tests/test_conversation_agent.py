from __future__ import annotations

from relationship_lifelog_agent.agent.answerability import check_answerability
from relationship_lifelog_agent.agent.conversation_state import ConversationState
from relationship_lifelog_agent.agent.evidence_contracts import evidence_contract_for_intent
from relationship_lifelog_agent.agent.profile_resolver import resolve_profile
from relationship_lifelog_agent.agent.query_planner import build_conversation_query_plan
from relationship_lifelog_agent.agent.question_understanding import understand_question
from relationship_lifelog_agent.agent.reasoning_orchestrator import answer_with_reasoning
from relationship_lifelog_agent.config import PathSettings, RelationshipSettings, Settings
from relationship_lifelog_agent.db.repository import RelationshipRepository
from relationship_lifelog_agent.privacy.guard import detect_answer_safety_violations
from relationship_lifelog_agent.profiles import PROFILE_NONE_VALUE
from relationship_lifelog_agent.ui.chat_ui import build_ui_answer


def test_question_understanding_extracts_relationship_frame() -> None:
    frame = understand_question("いおりとの喧嘩はどれくらいしている？")

    assert frame.target_profile_name == "いおり"
    assert "conflict_frequency" in frame.intents
    assert frame.requested_output == "aggregate_with_examples"
    assert frame.needs_evidence is True
    assert frame.risk_level == "relationship_private"


def test_profile_resolver_prefers_default_profile_over_duplicate_names(tmp_path) -> None:
    settings = _settings_with_duplicate_profiles(tmp_path, default_profile_id=1)
    frame = understand_question("いおりとの喧嘩はどれくらいしている？")

    resolution = resolve_profile(settings, frame)

    assert resolution.status == "resolved"
    assert resolution.profile is not None
    assert resolution.profile.id == 1


def test_profile_resolver_avoids_incomplete_profile_when_no_default(tmp_path) -> None:
    settings = _settings_with_duplicate_profiles(tmp_path, default_profile_id=999)
    frame = understand_question("いおりとの喧嘩はどれくらいしている？")

    resolution = resolve_profile(settings, frame)

    assert resolution.status == "resolved"
    assert resolution.profile is not None
    assert resolution.profile.id == 1


def test_missing_profile_answerability_reports_missing_need(tmp_path) -> None:
    settings = Settings(
        paths=PathSettings(relationship_db=str(tmp_path / "relationship.sqlite")),
        relationship=RelationshipSettings(default_profile_id=999),
    )
    frame = understand_question("いおりとの喧嘩はどれくらいしている？")
    resolution = resolve_profile(settings, frame)
    report = check_answerability(frame, resolution, settings)

    assert report.status == "missing_profile"
    assert any(need.name == "manual_profile" for need in report.missing_needs)


def test_self_speaker_missing_blocks_reply_delay_question(tmp_path) -> None:
    db_path = tmp_path / "relationship.sqlite"
    repo = RelationshipRepository(db_path)
    repo.create_profile(
        "いおり",
        person_source_id="plr:person:target",
        line_speaker_source_id="plr:line_speaker:target",
        relationship_label="partner",
    )
    settings = Settings(paths=PathSettings(relationship_db=str(db_path)), relationship=RelationshipSettings(default_profile_id=1))
    frame = understand_question("LINE返せていなかった時期は？")
    resolution = resolve_profile(settings, frame)
    report = check_answerability(frame, resolution, settings)

    assert report.status == "missing_self_speaker"
    assert any(need.name == "self_line_speaker" for need in report.missing_needs)


def test_referenced_day_uses_previous_observed_date() -> None:
    state = ConversationState(last_observed_dates=("2024-12-16",))

    frame = understand_question("その日のLINEを詳しく見せて", state)

    assert frame.referenced_previous_answer is True
    assert frame.date_from == "2024-12-16"
    assert frame.date_to == "2024-12-16"
    assert frame.requested_output == "line_detail_summary"


def test_query_plan_contains_required_conflict_tools(tmp_path) -> None:
    settings = _settings_with_duplicate_profiles(tmp_path, default_profile_id=1)
    frame = understand_question("いおりとの喧嘩はどれくらいしている？")
    resolution = resolve_profile(settings, frame)
    report = check_answerability(frame, resolution, settings)
    plan = build_conversation_query_plan(frame, report)

    tool_names = [step.tool_name for step in plan.plan_steps]
    assert "resolve_profile" in tool_names
    assert "search_line_conflict_signals" in tool_names
    assert "conflict_audit" in tool_names
    assert plan.expected_answer_shape == "aggregate_with_examples"


def test_conflict_evidence_contract_excludes_undated_notes() -> None:
    contract = evidence_contract_for_intent("conflict_frequency")

    assert contract.excludes("undated_note")
    assert contract.excludes("1970-01-01 note")
    assert contract.excludes("unrelated_note")


def test_orchestrator_updates_conversation_state(tmp_path) -> None:
    settings = _settings_with_duplicate_profiles(tmp_path, default_profile_id=1)

    response = answer_with_reasoning("いおりとの喧嘩はどれくらいしている？", settings=settings)

    assert response.conversation_state.active_profile_id == 1
    assert "conflict_frequency" in response.conversation_state.last_intents
    assert response.conversation_state.last_answerability_report["status"] in {
        "answerable",
        "answerable_with_caution",
        "partially_answerable",
    }
    assert "実行した検索方針" in response.answer_markdown


def test_chat_ui_uses_reasoning_orchestrator(tmp_path) -> None:
    settings = _settings_with_duplicate_profiles(tmp_path, default_profile_id=1)

    answer = build_ui_answer(
        "いおりとの喧嘩はどれくらいしている？",
        base_settings=settings,
        selected_backend="mock",
        selected_profile=PROFILE_NONE_VALUE,
        date_from="",
        date_to="",
        post_conflict_window_days=14,
        mode="private",
    )

    assert "実行した検索方針" in answer
    assert "必要だった情報" in answer
    assert "profile_id: 1" in answer


def test_public_mode_orchestrator_omits_excerpts_and_forbidden_phrases(tmp_path) -> None:
    settings = _settings_with_duplicate_profiles(tmp_path, default_profile_id=1)

    answer = build_ui_answer(
        "いおりとの喧嘩はどれくらいしている？",
        base_settings=settings,
        selected_backend="mock",
        selected_profile="1",
        date_from="",
        date_to="",
        post_conflict_window_days=14,
        mode="public",
    )

    assert "excerpt:" not in answer
    assert detect_answer_safety_violations(answer, mode="public", person_names=["いおり"]) == []


def _settings_with_duplicate_profiles(tmp_path, *, default_profile_id: int) -> Settings:
    db_path = tmp_path / "relationship.sqlite"
    repo = RelationshipRepository(db_path)
    repo.create_profile(
        "いおり",
        person_source_id="plr:person:target",
        line_speaker_source_id="plr:line_speaker:target",
        self_line_speaker_source_id="plr:line_speaker:self",
        relationship_label="partner",
    )
    repo.create_profile("いおり", person_source_id="plr:person:target", relationship_label="partner")
    return Settings(paths=PathSettings(relationship_db=str(db_path)), relationship=RelationshipSettings(default_profile_id=default_profile_id))
