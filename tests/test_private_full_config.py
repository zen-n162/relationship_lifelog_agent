from __future__ import annotations

from relationship_lifelog_agent.config import (
    AnalysisSettings,
    PrivateFullLlmPayloadSettings,
    Settings,
    load_config,
)
from relationship_lifelog_agent.full_context.types import (
    FullBatchAnalysis,
    FullDataManifest,
    FullFaceItem,
    FullLineItem,
    FullLocationItem,
    FullMediaItem,
    FullNoteItem,
    FullPromptBundle,
    FullRangeSynthesis,
    FullScanBatch,
)
from relationship_lifelog_agent.privacy.raw_payload_policy import (
    from_config,
    is_private_full_enabled,
    should_include_exact_gps,
    should_include_face_embeddings,
    should_include_photo_paths,
    should_include_raw_line_text,
    should_include_raw_note_text,
)


def test_analysis_mode_is_loaded(tmp_path) -> None:
    config = tmp_path / "config.yaml"
    config.write_text(
        """
analysis:
  mode: private_full_range
  default_scope: ask_llm
  allow_full_corpus: true
  allow_full_range: true
  allow_single_context_full_prompt: true
  allow_iterative_full_scan: true
  full_scan:
    batch_strategy: hybrid
    max_items_per_batch: 123
    max_chars_per_batch: 4567
    overlap_items: 3
    max_batches_per_run: 9
    max_reanalysis_rounds: 2
    summarize_after_each_batch: true
    preserve_raw_refs: true
llm:
  num_ctx: 32768
vision:
  enabled: false
  provider: ollama
  model: null
  pass_images: false
  pass_face_crops: false
  max_images_per_batch: 10
""",
        encoding="utf-8",
    )

    settings = load_config(config)

    assert settings.analysis.mode == "private_full_range"
    assert settings.analysis.full_scan.max_items_per_batch == 123
    assert settings.analysis.full_scan.max_chars_per_batch == 4567
    assert settings.llm.num_ctx == 32768
    assert settings.vision.provider == "ollama"


def test_private_full_llm_payload_is_loaded(tmp_path) -> None:
    config = tmp_path / "config.yaml"
    config.write_text(
        """
analysis:
  mode: private_full_range
private_full_llm_payload:
  allow_raw_line_text: true
  allow_raw_note_text: true
  allow_photo_paths: true
  allow_exact_gps: true
  allow_face_crops: true
  allow_face_embeddings: true
  allow_raw_face_embedding_values: false
  allow_private_file_paths: true
  allow_unverified_person_candidates: true
  allow_unverified_speaker_candidates: true
  allow_full_prompt_logging: false
  allow_raw_payload_cache: false
""",
        encoding="utf-8",
    )

    policy = from_config(load_config(config))

    assert is_private_full_enabled(policy) is True
    assert should_include_raw_line_text(policy) is True
    assert should_include_raw_note_text(policy) is True
    assert should_include_photo_paths(policy) is True
    assert should_include_exact_gps(policy) is True
    assert should_include_face_embeddings(policy) is True
    assert policy.should_include_face_crops() is True
    assert policy.should_include_private_file_paths() is True
    assert policy.should_include_unverified_person_candidates() is True
    assert policy.should_include_unverified_speaker_candidates() is True


def test_safe_window_policy_disables_raw_payload_even_when_flags_are_true() -> None:
    settings = Settings(
        analysis=AnalysisSettings(mode="safe_window"),
        private_full_llm_payload=PrivateFullLlmPayloadSettings(
            allow_raw_line_text=True,
            allow_raw_note_text=True,
            allow_photo_paths=True,
            allow_exact_gps=True,
            allow_face_embeddings=True,
        ),
    )

    policy = from_config(settings)

    assert is_private_full_enabled(policy) is False
    assert should_include_raw_line_text(policy) is False
    assert should_include_raw_note_text(policy) is False
    assert should_include_photo_paths(policy) is False
    assert should_include_exact_gps(policy) is False
    assert should_include_face_embeddings(policy) is False
    assert policy.to_dict()["allow_raw_line_text"] is False


def test_private_full_range_policy_respects_configured_true_flags() -> None:
    settings = Settings(
        analysis=AnalysisSettings(mode="private_full_range"),
        private_full_llm_payload=PrivateFullLlmPayloadSettings(
            allow_raw_line_text=True,
            allow_raw_note_text=True,
            allow_photo_paths=True,
            allow_exact_gps=True,
            allow_face_embeddings=True,
        ),
    )

    policy = from_config(settings)

    assert is_private_full_enabled(policy) is True
    assert should_include_raw_line_text(policy) is True
    assert should_include_raw_note_text(policy) is True
    assert should_include_photo_paths(policy) is True
    assert should_include_exact_gps(policy) is True
    assert should_include_face_embeddings(policy) is True


def test_raw_prompt_logging_and_payload_cache_default_false() -> None:
    settings = Settings()
    policy = from_config(settings)

    assert settings.private_full_llm_payload.allow_full_prompt_logging is False
    assert settings.private_full_llm_payload.allow_raw_payload_cache is False
    assert policy.allow_full_prompt_logging is False
    assert policy.allow_raw_payload_cache is False


def test_full_context_type_scaffolding_can_be_instantiated() -> None:
    line = FullLineItem(source_ref="line:1", source_id="1", conversation_id=None, timestamp="2025-01-01", speaker_id=None, speaker_label=None)
    note = FullNoteItem(source_ref="note:1", source_id="1", note_id="1")
    media = FullMediaItem(source_ref="media:1", source_id="1", media_id="1")
    face = FullFaceItem(source_ref="face:1", source_id="1", face_id="1")
    location = FullLocationItem(source_ref="location:1", source_id="1")
    manifest = FullDataManifest(run_id="run-1", line_count=1, note_count=1, media_count=1, face_count=1, location_count=1)
    prompt = FullPromptBundle(bundle_id="bundle-1", analysis_mode="private_full_range", prompt_text="prompt", source_refs=("line:1",))
    batch = FullScanBatch(
        batch_id="batch-1",
        batch_index=0,
        line_items=(line,),
        note_items=(note,),
        media_items=(media,),
        face_items=(face,),
        location_items=(location,),
        source_refs=("line:1", "note:1", "media:1", "face:1", "location:1"),
    )
    analysis = FullBatchAnalysis(batch_id="batch-1", relevant_evidence_refs=("line:1",))
    synthesis = FullRangeSynthesis(summary="summary", answer="answer", source_refs=("line:1",))

    assert manifest.run_id == "run-1"
    assert prompt.raw_payload_included is False
    assert batch.line_items == (line,)
    assert analysis.batch_id == "batch-1"
    assert synthesis.answer == "answer"
