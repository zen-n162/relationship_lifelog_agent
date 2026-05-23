from __future__ import annotations

import json

from relationship_lifelog_agent.config import (
    AnalysisSettings,
    PrivateFullLlmPayloadSettings,
    Settings,
)
from relationship_lifelog_agent.full_context.manifest import build_full_data_manifest
from relationship_lifelog_agent.full_context.prompt_packer import (
    build_batch_prompt,
    build_single_context_prompt,
    sanitize_or_include_item_by_policy,
)
from relationship_lifelog_agent.full_context.types import (
    FullFaceItem,
    FullLineItem,
    FullMediaItem,
    FullNoteItem,
    FullScanBatch,
)
from relationship_lifelog_agent.privacy.raw_payload_policy import from_config


def _private_policy(*, raw_embeddings: bool = False):
    return from_config(
        Settings(
            analysis=AnalysisSettings(mode="private_full_range"),
            private_full_llm_payload=PrivateFullLlmPayloadSettings(
                allow_raw_line_text=True,
                allow_raw_note_text=True,
                allow_photo_paths=True,
                allow_exact_gps=True,
                allow_face_crops=True,
                allow_face_embeddings=True,
                allow_raw_face_embedding_values=raw_embeddings,
                allow_private_file_paths=True,
                allow_unverified_person_candidates=True,
                allow_unverified_speaker_candidates=True,
                allow_full_prompt_logging=False,
                allow_raw_payload_cache=False,
            ),
        )
    )


def _safe_policy():
    return from_config(Settings(analysis=AnalysisSettings(mode="safe_window")))


def _items():
    line = FullLineItem(
        source_ref="line:1",
        source_id="line-1",
        conversation_id="c1",
        timestamp="2025-01-01T10:00:00",
        speaker_id="speaker-1",
        speaker_label="speaker",
        raw_text="RAW LINE TEXT",
        raw_metadata={"unverified_speaker_candidate": "candidate-a"},
    )
    note = FullNoteItem(
        source_ref="note:1",
        source_id="note-1",
        note_id="n1",
        title="note title",
        created_at="2025-01-01",
        raw_body="RAW NOTE BODY",
        source_path="/private/notes/a.md",
    )
    media = FullMediaItem(
        source_ref="media:1",
        source_id="media-1",
        media_id="m1",
        file_path="/private/photos/a.jpg",
        thumbnail_path="/private/photos/thumb.jpg",
        captured_at="2025-01-01",
        exact_gps={"lat": 35.123, "lon": 139.456},
        face_crop_paths=("/private/faces/f1.jpg",),
        face_embeddings=((0.1, 0.2, 0.3),),
        unverified_person_candidates=({"candidate_id": "p1"},),
    )
    face = FullFaceItem(
        source_ref="face:1",
        source_id="face-1",
        face_id="f1",
        crop_path="/private/faces/f1.jpg",
        embedding=(0.4, 0.5, 0.6),
    )
    return line, note, media, face


def _manifest(line, note, media, face):
    return build_full_data_manifest(
        run_id="run-1",
        profile_id=1,
        date_from="2025-01-01",
        date_to="2025-01-01",
        line_items=(line,),
        note_items=(note,),
        media_items=(media,),
        face_items=(face,),
        location_items=(),
    )


def test_private_full_prompt_includes_raw_line_note_photo_gps_and_face_crop() -> None:
    line, note, media, face = _items()
    manifest = _manifest(line, note, media, face)

    bundle = build_single_context_prompt(
        question="question",
        profile_context={"profile_id": 1},
        manifest=manifest,
        full_data={
            "line_items": (line,),
            "note_items": (note,),
            "media_items": (media,),
            "face_items": (face,),
            "location_items": (),
        },
        raw_payload_policy=_private_policy(),
    )

    prompt = bundle.prompt_text
    assert "RAW LINE TEXT" in prompt
    assert "RAW NOTE BODY" in prompt
    assert "/private/photos/a.jpg" in prompt
    assert '"lat": 35.123' in prompt
    assert "/private/faces/f1.jpg" in prompt
    assert "line-1" in prompt
    assert "note-1" in prompt
    assert "media-1" in prompt
    assert bundle.raw_payload_included is True
    assert bundle.metadata["prompt_logging_allowed"] is False
    assert bundle.metadata["raw_payload_cache_allowed"] is False


def test_face_embedding_values_are_summarized_by_default() -> None:
    _, _, media, face = _items()
    policy = _private_policy(raw_embeddings=False)

    media_payload = sanitize_or_include_item_by_policy(media, policy)
    face_payload = sanitize_or_include_item_by_policy(face, policy)

    assert media_payload["face_embeddings"] == {"embedding_count": 1, "embedding_dim": 3}
    assert face_payload["embedding"] == {"embedding_count": 1, "embedding_dim": 3}
    assert "0.1" not in json.dumps(media_payload)
    assert "0.4" not in json.dumps(face_payload)


def test_face_embedding_values_are_included_when_explicitly_allowed() -> None:
    _, _, media, face = _items()
    policy = _private_policy(raw_embeddings=True)

    media_payload = sanitize_or_include_item_by_policy(media, policy)
    face_payload = sanitize_or_include_item_by_policy(face, policy)

    assert media_payload["face_embeddings"] == ((0.1, 0.2, 0.3),)
    assert face_payload["embedding"] == (0.4, 0.5, 0.6)


def test_safe_window_prompt_excludes_raw_payload_but_keeps_source_ids() -> None:
    line, note, media, face = _items()
    manifest = _manifest(line, note, media, face)

    bundle = build_single_context_prompt(
        question="question",
        profile_context={"profile_id": 1},
        manifest=manifest,
        full_data={
            "line_items": (line,),
            "note_items": (note,),
            "media_items": (media,),
            "face_items": (face,),
            "location_items": (),
        },
        raw_payload_policy=_safe_policy(),
    )

    prompt = bundle.prompt_text
    assert "RAW LINE TEXT" not in prompt
    assert "RAW NOTE BODY" not in prompt
    assert "/private/photos/a.jpg" not in prompt
    assert "/private/faces/f1.jpg" not in prompt
    assert "35.123" not in prompt
    assert "line-1" in prompt
    assert "note-1" in prompt
    assert "media-1" in prompt
    assert "[redacted:raw_line_text]" in prompt
    assert "[redacted:raw_note_body]" in prompt
    assert bundle.raw_payload_included is False


def test_batch_prompt_contains_batch_source_ids() -> None:
    line, note, media, face = _items()
    manifest = _manifest(line, note, media, face)
    batch = FullScanBatch(
        batch_id="batch-1",
        batch_index=0,
        date_from="2025-01-01",
        date_to="2025-01-01",
        source_types=("line", "note", "media", "face"),
        line_items=(line,),
        note_items=(note,),
        media_items=(media,),
        face_items=(face,),
        source_refs=("line:1", "note:1", "media:1", "face:1"),
    )

    bundle = build_batch_prompt(
        question="question",
        profile_context={"profile_id": 1},
        manifest=manifest,
        batch=batch,
        raw_payload_policy=_private_policy(),
    )

    assert "batch-1" in bundle.prompt_text
    assert set(bundle.source_refs) >= {"line:1", "note:1", "media:1", "face:1"}


def test_prompt_logging_and_raw_payload_cache_default_false() -> None:
    policy = _private_policy()

    assert policy.allow_full_prompt_logging is False
    assert policy.allow_raw_payload_cache is False
