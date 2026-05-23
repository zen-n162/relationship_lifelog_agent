from __future__ import annotations

from collections.abc import Iterator

import gradio as gr

from relationship_lifelog_agent.agent.reasoning_orchestrator import stream_answer
from relationship_lifelog_agent.config import LlmSettings, PathSettings, RelationshipSettings, Settings, UiSettings
from relationship_lifelog_agent.db.repository import RelationshipRepository
from relationship_lifelog_agent.llm.local_client import LocalLlmClient
from relationship_lifelog_agent.ui.chat_ui import build_full_context_dry_run_preview_text, build_ui_chat_turn_stream


def test_stream_answer_yields_progress_and_final_answer(tmp_path) -> None:
    settings = _settings_with_profile(tmp_path)

    events = list(stream_answer("いおりとの喧嘩はどのくらいしている？", settings=settings))

    assert len(events) > 4
    assert events[-1].kind == "final_answer"
    assert "要約:" in events[-1].message
    assert any(event.kind == "progress" for event in events)


def test_stream_answer_progress_is_safe_for_user(tmp_path) -> None:
    settings = _settings_with_profile(tmp_path)

    events = list(stream_answer("いおりとの喧嘩はどのくらいしている？", settings=settings))
    progress_text = "\n".join(event.message for event in events if event.kind != "final_answer")

    assert "raw LINE" not in progress_text
    assert "raw note" not in progress_text
    assert "/home/" not in progress_text
    assert "37.123" not in progress_text
    assert "face_embedding" not in progress_text


def test_stream_answer_llm_fallback_yields_warning(tmp_path) -> None:
    llm = LlmSettings(
        provider="ollama",
        model="fake",
        base_url="http://127.0.0.1:11434",
        enabled=True,
        use_for_question_understanding=True,
        timeout_seconds=1,
    )
    settings = _settings_with_profile(tmp_path, llm=llm)
    client = LocalLlmClient(settings.llm, http_post=lambda url, payload, timeout: {"message": {"content": "not json"}})

    events = list(stream_answer("いおりとの喧嘩はどのくらいしている？", settings=settings, llm_client=client))

    assert any(event.kind == "llm_start" for event in events)
    assert any(event.kind == "warning" and "rule-based" in event.message for event in events)


def test_stream_answer_has_tool_start_and_done_events(tmp_path) -> None:
    settings = _settings_with_profile(tmp_path)

    events = list(stream_answer("いおりとの喧嘩はどのくらいしている？", settings=settings))

    assert any(event.kind == "tool_start" for event in events)
    assert any(event.kind == "tool_done" for event in events)


def test_chat_ui_stream_handler_is_generator_and_sets_metadata(tmp_path) -> None:
    settings = _settings_with_profile(tmp_path)

    result = build_ui_chat_turn_stream(
        "いおりとの喧嘩はどのくらいしている？",
        [],
        base_settings=settings,
        selected_backend="mock",
        selected_profile="1",
        date_from="",
        date_to="",
        post_conflict_window_days=14,
        mode="private",
        show_debug=False,
        conversation_state={},
    )

    assert isinstance(result, Iterator)
    snapshots = list(result)
    first_history = snapshots[0][1]
    assert first_history[-1]["metadata"]["status"] == "pending"
    assert first_history[-1]["metadata"]["title"] == "処理プロセス"
    final_history = snapshots[-1][1]
    progress_messages = [
        message
        for message in final_history
        if message.get("metadata", {}).get("title") == "処理プロセス"
    ]
    assert len(progress_messages) == 1
    assert progress_messages[0]["metadata"]["status"] == "done"
    assert progress_messages[0]["metadata"]["log"] == "完了"
    assert not [message for message in final_history if message.get("metadata", {}).get("status") == "pending"]
    assert final_history[-1]["metadata"]["title"] == "処理プロセス"
    assert final_history[-2]["role"] == "assistant"
    assert "要約:" in final_history[-2]["content"]
    gr.Chatbot().postprocess(final_history)


def test_completed_progress_block_stays_collapsed_on_next_turn(tmp_path) -> None:
    settings = _settings_with_profile(tmp_path)

    first_snapshots = list(
        build_ui_chat_turn_stream(
            "いおりとの喧嘩はどのくらいしている？",
            [],
            base_settings=settings,
            selected_backend="mock",
            selected_profile="1",
            date_from="",
            date_to="",
            post_conflict_window_days=14,
            mode="private",
            show_debug=False,
            conversation_state={},
        )
    )
    first_history = first_snapshots[-1][1]

    second_snapshots = list(
        build_ui_chat_turn_stream(
            "その日を詳しく",
            first_history,
            base_settings=settings,
            selected_backend="mock",
            selected_profile="1",
            date_from="",
            date_to="",
            post_conflict_window_days=14,
            mode="private",
            show_debug=False,
            conversation_state=first_snapshots[-1][5],
        )
    )
    final_history = second_snapshots[-1][1]
    progress_messages = [
        message
        for message in final_history
        if message.get("metadata", {}).get("title") == "処理プロセス"
    ]

    assert len(progress_messages) == 2
    assert all(message["metadata"]["status"] == "done" for message in progress_messages)
    assert not [message for message in final_history if message.get("metadata", {}).get("status") == "pending"]
    gr.Chatbot().postprocess(final_history)


def test_streaming_config_defaults_hide_raw_llm_thinking() -> None:
    settings = Settings()

    assert settings.ui.stream_progress is True
    assert settings.ui.show_raw_llm_thinking is False


def test_chat_ui_stream_handler_passes_private_full_analysis_mode(tmp_path) -> None:
    settings = _settings_with_profile(tmp_path)

    snapshots = list(
        build_ui_chat_turn_stream(
            "いおりとの記録を全体から見て",
            [],
            base_settings=settings,
            selected_backend="mock",
            analysis_mode="private_full_range",
            date_scope="custom",
            selected_profile="1",
            date_from="2025-01-01",
            date_to="2025-01-31",
            post_conflict_window_days=14,
            include_raw_line_text=True,
            include_raw_note_text=True,
            include_photo_paths=True,
            include_exact_gps=True,
            include_face_crops=True,
            include_face_embeddings=True,
            include_private_file_paths=True,
            include_unverified_person_candidates=True,
            include_unverified_speaker_candidates=True,
            max_runtime=120,
            max_llm_calls=3,
            max_batches=5,
            mode="private",
            show_debug=False,
            conversation_state={},
        )
    )

    final_history = snapshots[-1][1]
    assert "Agent Runtime" in final_history[-2]["content"]
    assert snapshots[-1][5]["last_query_plan"]["runtime"] == "private_full"


def test_private_full_stream_progress_is_safe_and_multi_step(tmp_path) -> None:
    settings = _settings_with_profile(tmp_path)

    snapshots = list(
        build_ui_chat_turn_stream(
            "raw LINE全文とraw note全文を出さずに全体を見て",
            [],
            base_settings=settings,
            selected_backend="mock",
            analysis_mode="private_full_range",
            date_scope="all",
            selected_profile="1",
            date_from="",
            date_to="",
            post_conflict_window_days=14,
            mode="private",
            show_debug=False,
            conversation_state={},
        )
    )

    progress_messages = [
        message
        for message in snapshots[-1][1]
        if message.get("metadata", {}).get("title") == "処理プロセス"
    ]
    assert progress_messages
    progress_text = progress_messages[-1]["content"]
    assert "質問を理解しています" in progress_text
    assert "manifest" in progress_text
    assert "raw LINE全文" not in progress_text
    assert "raw note全文" not in progress_text
    assert "/home/" not in progress_text
    assert progress_messages[-1]["metadata"]["status"] == "done"


def test_full_corpus_ui_dry_run_preview_is_counts_only(tmp_path) -> None:
    settings = _settings_with_profile(tmp_path)

    preview = build_full_context_dry_run_preview_text(
        "mock",
        "private_full_corpus",
        "all",
        "1",
        "",
        "",
        14,
        5,
        base_settings=settings,
    )

    assert "Full-context dry-run preview" in preview
    assert "analysis_mode: `private_full_corpus`" in preview
    assert "date_range: `all`" in preview
    assert "llm_calls: `0`" in preview
    assert "時間がかかる" in preview
    assert "/home/" not in preview
    assert "raw LINE全文" not in preview
    assert "raw note全文" not in preview


def _settings_with_profile(tmp_path, *, llm: LlmSettings | None = None) -> Settings:
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
        paths=PathSettings(relationship_db=str(db_path)),
        relationship=RelationshipSettings(default_profile_id=profile_id),
        ui=UiSettings(stream_progress=True),
        llm=llm or LlmSettings(enabled=False),
    )
