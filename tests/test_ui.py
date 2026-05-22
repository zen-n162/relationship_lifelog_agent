from relationship_lifelog_agent.agent.executor import answer_question
from relationship_lifelog_agent.config import Settings, gradio_launch_kwargs
from relationship_lifelog_agent.ui.chat_ui import build_chat_ui


def test_chat_ui_builds_with_safe_defaults() -> None:
    settings = Settings()
    demo = build_chat_ui(settings)
    kwargs = gradio_launch_kwargs(settings)

    assert demo is not None
    assert settings.ui.show_debug_by_default is False
    assert kwargs["server_name"] == "127.0.0.1"
    assert kwargs["share"] is False


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
