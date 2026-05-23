from relationship_lifelog_agent.agent.executor import answer_question
from relationship_lifelog_agent.app import build_arg_parser
from relationship_lifelog_agent.config import AdapterSettings, PathSettings, RelationshipSettings, Settings, gradio_launch_kwargs
from relationship_lifelog_agent.db.repository import RelationshipRepository
from relationship_lifelog_agent.privacy.guard import detect_answer_safety_violations
from relationship_lifelog_agent.profiles import PROFILE_NONE_VALUE
from relationship_lifelog_agent.ui.chat_ui import (
    _latest_user_message,
    build_chat_ui,
    build_ui_answer,
    build_ui_chat_response,
    build_ui_chat_answer,
    save_profile_from_ui,
    save_review_action_from_ui,
)


def test_chat_ui_builds_with_safe_defaults() -> None:
    settings = Settings()
    demo = build_chat_ui(settings)
    kwargs = gradio_launch_kwargs(settings)

    assert demo is not None
    assert settings.ui.show_debug_by_default is False
    assert kwargs["server_name"] == "127.0.0.1"
    assert kwargs["share"] is False


def test_chat_ui_launch_port_can_be_overridden_per_run() -> None:
    settings = Settings()
    kwargs = gradio_launch_kwargs(settings, port=7863)

    assert kwargs["server_name"] == "127.0.0.1"
    assert kwargs["server_port"] == 7863
    assert kwargs["share"] is False


def test_app_parser_accepts_runtime_port() -> None:
    args = build_arg_parser().parse_args(["--port", "7863", "--smoke"])

    assert args.port == 7863
    assert args.smoke is True


def test_chat_ui_extracts_latest_user_message_from_gradio_history_shapes() -> None:
    assert _latest_user_message([["喧嘩はどのくらいしている？", None]]) == "喧嘩はどのくらいしている？"
    assert _latest_user_message([{"role": "user", "content": ["喧嘩の後どこへ行っている？"]}]) == "喧嘩の後どこへ行っている？"


def test_answer_uses_collapsible_markdown_sections() -> None:
    answer = answer_question("喧嘩はどのくらいしている？")

    assert "要約:" in answer
    assert "集計:" in answer
    assert "具体例:" in answer
    assert "根拠:" in answer
    assert "信頼度:" in answer
    assert "注意:" in answer
    assert "<details>" in answer
    assert "<summary>根拠を表示</summary>" in answer
    assert "<summary>注意を表示</summary>" in answer


def test_chat_ui_mock_backend_answer_uses_profile_and_date_range(tmp_path) -> None:
    settings, profile_id = _settings_with_profile(tmp_path)

    answer = build_ui_answer(
        "喧嘩はどのくらいしている？",
        base_settings=settings,
        selected_backend="mock",
        selected_profile=str(profile_id),
        date_from="2025-01-01",
        date_to="2025-01-31",
        post_conflict_window_days=7,
        mode="private",
    )

    assert "mock データ上" in answer
    assert "2025-01-10" in answer
    assert "2025-02-14" not in answer
    assert "profile_configured: True" in answer


def test_chat_ui_answer_coerces_list_question_from_gradio_event(tmp_path) -> None:
    settings, profile_id = _settings_with_profile(tmp_path)

    answer = build_ui_answer(
        ["喧嘩はどのくらいしている？"],
        base_settings=settings,
        selected_backend="mock",
        selected_profile=str(profile_id),
        date_from="2025-01-01",
        date_to="2025-01-31",
        post_conflict_window_days=7,
        mode="private",
    )

    assert "要約:" in answer
    assert "喧嘩候補" in answer


def test_chat_ui_response_returns_gradio_messages_format(tmp_path) -> None:
    import gradio as gr

    settings, profile_id = _settings_with_profile(tmp_path)

    history, chat_answer = build_ui_chat_response(
        "喧嘩はどのくらいしている？",
        [],
        base_settings=settings,
        selected_backend="mock",
        selected_profile=str(profile_id),
        date_from="2025-01-01",
        date_to="2025-01-31",
        post_conflict_window_days=7,
        mode="private",
    )

    assert chat_answer is not None
    assert history[0]["role"] == "user"
    assert history[1]["role"] == "assistant"
    assert "要約:" in history[1]["content"]
    gr.Chatbot().postprocess(history)


def test_chat_ui_upstream_readonly_unconfigured_fails_safely(tmp_path) -> None:
    settings, profile_id = _settings_with_profile(
        tmp_path,
        adapter=AdapterSettings(backend="upstream_readonly"),
    )

    answer = build_ui_answer(
        "喧嘩はどのくらいしている？",
        base_settings=settings,
        selected_backend="upstream_readonly",
        selected_profile=str(profile_id),
        date_from="2025-01-01",
        date_to="2025-01-31",
        post_conflict_window_days=14,
        mode="private",
    )

    assert "上流adapter未設定" in answer
    assert "personal_lifelog_db" in answer
    assert "notes_lifelog_db" in answer


def test_chat_ui_uses_saved_relationship_db_candidate(tmp_path) -> None:
    settings, profile_id = _settings_with_profile(
        tmp_path,
        adapter=AdapterSettings(backend="upstream_readonly"),
    )
    repo = RelationshipRepository(settings.paths.relationship_db)
    event_id = repo.create_event(
        profile_id=profile_id,
        event_type="conflict",
        event_date="2025-03-05",
        summary="保存済みの喧嘩候補。",
        status="candidate",
        review_status="unreviewed",
        confidence=0.61,
        evidence_strength=0.58,
    )
    repo.create_evidence(
        event_id=event_id,
        source_type="manual",
        source_id="saved-candidate",
        source_pointer="manual:saved-candidate",
        summary="保存済みcandidateのsource pointer。",
        confidence=0.6,
        evidence_strength=0.58,
    )

    answer = build_ui_answer(
        "喧嘩はどのくらいしている？",
        base_settings=settings,
        selected_backend="upstream_readonly",
        selected_profile=str(profile_id),
        date_from="2025-03-01",
        date_to="2025-03-31",
        post_conflict_window_days=14,
        mode="private",
    )

    assert "保存済みの喧嘩候補" in answer
    assert "total_candidates: 1" in answer


def test_chat_ui_returns_review_targets_for_saved_candidates(tmp_path) -> None:
    settings, profile_id = _settings_with_profile(
        tmp_path,
        adapter=AdapterSettings(backend="upstream_readonly"),
    )
    repo = RelationshipRepository(settings.paths.relationship_db)
    event_id = repo.create_event(
        profile_id=profile_id,
        event_type="conflict",
        event_date="2025-03-05",
        summary="保存済みの喧嘩候補。",
        status="candidate",
        review_status="unreviewed",
        confidence=0.61,
        evidence_strength=0.58,
    )
    repo.create_evidence(
        event_id=event_id,
        source_type="manual",
        source_id="saved-candidate",
        source_pointer="manual:saved-candidate",
        summary="保存済みcandidateのsource pointer。",
        confidence=0.6,
        evidence_strength=0.58,
    )

    chat_answer = build_ui_chat_answer(
        "喧嘩はどのくらいしている？",
        base_settings=settings,
        selected_backend="upstream_readonly",
        selected_profile=str(profile_id),
        date_from="2025-03-01",
        date_to="2025-03-31",
        post_conflict_window_days=14,
        mode="private",
    )

    assert chat_answer.review_targets
    assert chat_answer.review_targets[0].event_id == event_id
    assert "保存済みの喧嘩候補" in chat_answer.review_targets[0].label


def test_chat_ui_save_review_action_updates_event(tmp_path) -> None:
    settings, profile_id = _settings_with_profile(tmp_path)
    repo = RelationshipRepository(settings.paths.relationship_db)
    event_id = repo.create_event(
        profile_id=profile_id,
        event_type="conflict",
        event_date="2025-03-05",
        summary="保存済みの喧嘩候補。",
    )

    status = save_review_action_from_ui(
        str(event_id),
        "reject",
        "not this one",
        [{"event_id": event_id}],
        settings.paths.relationship_db,
    )

    assert "レビューを保存しました" in status
    assert repo.get_event(event_id)["review_status"] == "rejected"
    assert repo.list_review_actions(event_id=event_id)[0]["action"] == "reject"


def test_chat_ui_public_mode_redacts_saved_candidate_profile_name(tmp_path) -> None:
    settings, profile_id = _settings_with_profile(tmp_path, profile_name="山田太郎")
    repo = RelationshipRepository(settings.paths.relationship_db)
    repo.create_event(
        profile_id=profile_id,
        event_type="conflict",
        event_date="2025-03-05",
        summary="山田太郎との喧嘩候補。",
        status="candidate",
        review_status="unreviewed",
        confidence=0.61,
        evidence_strength=0.58,
    )

    answer = build_ui_answer(
        "喧嘩はどのくらいしている？",
        base_settings=settings,
        selected_backend="upstream_readonly",
        selected_profile=str(profile_id),
        date_from="2025-03-01",
        date_to="2025-03-31",
        post_conflict_window_days=14,
        mode="public",
        show_debug=True,
    )

    assert "山田太郎" not in answer
    assert "excerpt:" not in answer
    assert detect_answer_safety_violations(answer, mode="public", person_names=["山田太郎"]) == []


def test_chat_ui_profile_missing_returns_safe_guidance(tmp_path) -> None:
    settings = Settings(paths=PathSettings(relationship_db=str(tmp_path / "relationship.sqlite")))

    answer = build_ui_answer(
        "喧嘩はどのくらいしている？",
        base_settings=settings,
        selected_backend="mock",
        selected_profile=PROFILE_NONE_VALUE,
        date_from="",
        date_to="",
        post_conflict_window_days=14,
        mode="private",
    )

    assert "relationship profile が手動設定されていない" in answer
    assert "AIは恋人ラベルを推定しません" in answer


def test_chat_ui_can_save_manual_profile_with_self_ids(tmp_path) -> None:
    db_path = tmp_path / "relationship.sqlite"

    status, _dropdown_update = save_profile_from_ui(
        PROFILE_NONE_VALUE,
        "手動profile",
        "partner",
        "plr:person:target",
        "plr:line_speaker:target",
        "plr:line_speaker_group:target",
        "plr:person:self",
        "plr:line_speaker:self",
        "plr:line_speaker_group:self",
        str(db_path),
    )

    repo = RelationshipRepository(db_path)
    profile = repo.get_profile(1)
    assert "manual profile を保存しました" in status
    assert profile["label_source"] == "user_manual"
    assert profile["line_speaker_group_source_id"] == "plr:line_speaker_group:target"
    assert profile["self_person_source_id"] == "plr:person:self"
    assert profile["self_line_speaker_source_id"] == "plr:line_speaker:self"
    assert profile["self_line_speaker_group_source_id"] == "plr:line_speaker_group:self"


def _settings_with_profile(
    tmp_path,
    *,
    profile_name: str = "Aさん",
    adapter: AdapterSettings | None = None,
) -> tuple[Settings, int]:
    db_path = tmp_path / "relationship.sqlite"
    repo = RelationshipRepository(db_path)
    profile_id = repo.create_profile(
        profile_name,
        person_source_id="plr:person:4",
        line_speaker_source_id="plr:line_speaker:2",
        relationship_label="partner",
    )
    settings = Settings(
        adapter=adapter or AdapterSettings(backend="mock"),
        paths=PathSettings(relationship_db=str(db_path)),
        relationship=RelationshipSettings(default_profile_id=profile_id),
    )
    return settings, profile_id
