from __future__ import annotations

from collections.abc import Iterator

import gradio as gr

from relationship_lifelog_agent.agent.reasoning_orchestrator import stream_answer
from relationship_lifelog_agent.config import LlmSettings, PathSettings, RelationshipSettings, Settings, UiSettings
from relationship_lifelog_agent.db.repository import RelationshipRepository
from relationship_lifelog_agent.llm.local_client import LocalLlmClient
from relationship_lifelog_agent.ui.chat_ui import build_ui_chat_turn_stream


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
    assert first_history[-1]["metadata"]["status"] in {"pending", "done"}
    assert first_history[-1]["metadata"]["title"]
    final_history = snapshots[-1][1]
    assert final_history[-1]["role"] == "assistant"
    assert "要約:" in final_history[-1]["content"]
    gr.Chatbot().postprocess(final_history)


def test_streaming_config_defaults_hide_raw_llm_thinking() -> None:
    settings = Settings()

    assert settings.ui.stream_progress is True
    assert settings.ui.show_raw_llm_thinking is False


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
