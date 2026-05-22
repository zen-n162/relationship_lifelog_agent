import pytest

from relationship_lifelog_agent.adapters.types import EvidenceItem
from relationship_lifelog_agent.config import AppSettings, Settings, gradio_launch_kwargs, validate_local_safety
from relationship_lifelog_agent.privacy.guard import detect_answer_safety_violations, sanitize_answer
from relationship_lifelog_agent.privacy.redaction import redact_evidence_for_mode, redact_public_text


def test_public_mode_redacts_private_identifiers() -> None:
    text = "山田太郎 は 恋人。GPS は 35.12345, 139.12345。path=/home/user/private/file.txt"
    redacted = redact_public_text(text, person_names=["山田太郎"])
    assert "山田太郎" not in redacted
    assert "恋人" not in redacted
    assert "35.12345" not in redacted
    assert "/home/user/private/file.txt" not in redacted
    assert "人物A" in redacted
    assert "[relationship label redacted]" in redacted
    assert "[exact GPS redacted]" in redacted
    assert "[private path redacted]" in redacted


def test_public_mode_redacts_raw_text_face_photo_and_source_paths() -> None:
    text = (
        "LINE全文: private line message should disappear\n"
        "メモ全文: private note body should disappear\n"
        "source_file_path=/home/user/private/source.json\n"
        "face_crop_path=/home/user/private/face_crop.png\n"
        "real_photo=/home/user/photos/private.jpg\n"
        "GPS=35.123456, 139.123456"
    )
    redacted = redact_public_text(text)

    assert "private line message" not in redacted
    assert "private note body" not in redacted
    assert "/home/user/private/source.json" not in redacted
    assert "/home/user/private/face_crop.png" not in redacted
    assert "/home/user/photos/private.jpg" not in redacted
    assert "35.123456" not in redacted
    assert "[LINE full text redacted]" in redacted
    assert "[note full text redacted]" in redacted
    assert "[source file path redacted]" in redacted
    assert "[face data redacted]" in redacted
    assert "[real photo redacted]" in redacted
    assert "[exact GPS redacted]" in redacted


def test_public_mode_omits_raw_evidence_excerpt() -> None:
    item = EvidenceItem(
        source_type="line_message",
        source_id="mock",
        date="2025-01-01",
        role="primary",
        summary="LINE summary",
        excerpt="full private LINE text",
    )
    redacted = redact_evidence_for_mode(item, mode="public")
    assert redacted.excerpt is None
    assert "full private LINE text" not in repr(redacted)


def test_public_mode_sanitized_answer_has_no_public_violations() -> None:
    unsafe = (
        "山田太郎 は 恋人。LINE全文: hello private text\n"
        "メモ全文: private note\n"
        "source path: /home/user/source.txt\n"
        "face_crop=/home/user/crop.png\n"
        "photo=/home/user/photo.jpg\n"
        "35.123456, 139.123456"
    )
    safe = sanitize_answer(unsafe, mode="public", person_names=["山田太郎"])
    violations = detect_answer_safety_violations(safe, mode="public", person_names=["山田太郎"])

    assert "山田太郎" not in safe
    assert "人物A" in safe
    assert violations == []


def test_gradio_launch_kwargs_never_enable_share() -> None:
    settings = Settings()
    kwargs = gradio_launch_kwargs(settings)
    assert kwargs["server_name"] == "127.0.0.1"
    assert kwargs["share"] is False
    assert kwargs["allowed_paths"] == []
    assert kwargs["footer_links"] == []
    assert kwargs["enable_monitoring"] is False


def test_gradio_launch_port_override_keeps_safe_host_and_share() -> None:
    settings = Settings()
    kwargs = gradio_launch_kwargs(settings, port=7863)
    assert kwargs["server_name"] == "127.0.0.1"
    assert kwargs["server_port"] == 7863
    assert kwargs["share"] is False


def test_unsafe_config_is_rejected() -> None:
    with pytest.raises(ValueError):
        validate_local_safety(Settings(app=AppSettings(allow_gradio_share=True)))
