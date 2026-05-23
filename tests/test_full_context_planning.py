from __future__ import annotations

from relationship_lifelog_agent.cli import main as cli_main
from relationship_lifelog_agent.config import LlmSettings
from relationship_lifelog_agent.db.repository import RelationshipRepository
from relationship_lifelog_agent.full_context.batch_builder import (
    build_chronological_batches,
    build_hybrid_batches,
    build_source_then_time_batches,
    validate_batch_coverage,
)
from relationship_lifelog_agent.full_context.context_budget import decide_context_mode
from relationship_lifelog_agent.full_context.manifest import build_full_data_manifest
from relationship_lifelog_agent.full_context.types import (
    FullDataManifest,
    FullLineItem,
    FullMediaItem,
    FullNoteItem,
)


def test_manifest_counts_and_date_coverage_are_correct() -> None:
    line = FullLineItem(
        source_ref="line:1",
        source_id="1",
        conversation_id="c1",
        timestamp="2025-01-01T12:00:00",
        speaker_id="s1",
        speaker_label="speaker",
        raw_text="hello",
    )
    note = FullNoteItem(
        source_ref="note:1",
        source_id="1",
        note_id="n1",
        created_at="2025-01-02",
        raw_body="note body",
    )
    media = FullMediaItem(
        source_ref="media:1",
        source_id="1",
        media_id="m1",
        captured_at="2025-01-03T09:00:00",
        file_path="/private/photo.jpg",
        exact_gps={"lat": 35.0, "lon": 139.0},
    )

    manifest = build_full_data_manifest(
        run_id="run-1",
        profile_id=1,
        date_from="2025-01-01",
        date_to="2025-01-03",
        line_items=(line,),
        note_items=(note,),
        media_items=(media,),
        face_items=(),
        location_items=(),
    )

    assert manifest.line_count == 1
    assert manifest.note_count == 1
    assert manifest.media_count == 1
    assert manifest.face_count == 0
    assert manifest.location_count == 0
    assert manifest.date_coverage["observed_start"] == "2025-01-01"
    assert manifest.date_coverage["observed_end"] == "2025-01-03"
    assert manifest.date_coverage["days_with_data"] == 3
    assert manifest.date_coverage["requested_days"] == 3
    assert manifest.date_coverage["coverage_ratio"] == 1.0
    assert manifest.source_counts_by_day["2025-01-01"]["line"] == 1
    assert manifest.source_counts_by_day["2025-01-02"]["note"] == 1
    assert manifest.source_counts_by_day["2025-01-03"]["media"] == 1
    assert "raw_line_text" in manifest.available_payload_types
    assert "raw_note_body" in manifest.available_payload_types
    assert "photo_paths" in manifest.available_payload_types
    assert "exact_gps" in manifest.available_payload_types
    assert manifest.estimated_tokens > 0


def test_context_mode_single_context_when_budget_fits() -> None:
    manifest = FullDataManifest(run_id="run-1", estimated_tokens=700)

    assert decide_context_mode(manifest, LlmSettings(num_ctx=1000)) == "single_context"


def test_context_mode_iterative_when_budget_exceeds_threshold() -> None:
    manifest = FullDataManifest(run_id="run-1", estimated_tokens=751)

    assert decide_context_mode(manifest, LlmSettings(num_ctx=1000)) == "iterative_full_scan"


def test_chronological_batches_cover_all_source_refs_and_respect_item_limit() -> None:
    line_items = tuple(
        FullLineItem(
            source_ref=f"line:{index}",
            source_id=str(index),
            conversation_id="c1",
            timestamp=f"2025-01-0{index}",
            speaker_id="s1",
            speaker_label="speaker",
            raw_text=f"message {index}",
        )
        for index in range(1, 6)
    )

    batches = build_chronological_batches(
        run_id="run-1",
        line_items=line_items,
        max_items_per_batch=3,
        max_chars_per_batch=10_000,
        overlap_items=1,
    )
    coverage = validate_batch_coverage(line_items=line_items, batches=batches)

    assert coverage.ok is True
    assert coverage.total_source_refs == 5
    assert coverage.covered_source_refs == 5
    assert not coverage.missing_source_refs
    assert len(batches) == 2
    assert all(len(batch.source_refs) <= 3 for batch in batches)


def test_batch_overlap_is_included() -> None:
    line_items = tuple(
        FullLineItem(
            source_ref=f"line:{index}",
            source_id=str(index),
            conversation_id="c1",
            timestamp=f"2025-01-0{index}",
            speaker_id="s1",
            speaker_label="speaker",
        )
        for index in range(1, 6)
    )

    batches = build_chronological_batches(
        run_id="run-1",
        line_items=line_items,
        max_items_per_batch=3,
        max_chars_per_batch=10_000,
        overlap_items=1,
    )

    assert batches[0].source_refs[-1] == batches[1].source_refs[0]
    assert batches[0].source_refs[-1] in validate_batch_coverage(line_items=line_items, batches=batches).duplicate_source_refs


def test_source_then_time_batches_respect_source_grouping_and_limits() -> None:
    lines = tuple(
        FullLineItem(
            source_ref=f"line:{index}",
            source_id=str(index),
            conversation_id="c1",
            timestamp="2025-01-01",
            speaker_id="s1",
            speaker_label="speaker",
        )
        for index in range(2)
    )
    notes = tuple(
        FullNoteItem(
            source_ref=f"note:{index}",
            source_id=str(index),
            note_id=str(index),
            created_at="2025-01-02",
            raw_body="note",
        )
        for index in range(2)
    )

    batches = build_source_then_time_batches(
        run_id="run-1",
        line_items=lines,
        note_items=notes,
        max_items_per_batch=2,
        max_chars_per_batch=10_000,
        overlap_items=0,
    )

    assert all(len(batch.source_refs) <= 2 for batch in batches)
    assert validate_batch_coverage(line_items=lines, note_items=notes, batches=batches).ok is True


def test_hybrid_batches_cover_media_and_do_not_drop_refs() -> None:
    media_items = tuple(
        FullMediaItem(
            source_ref=f"media:{index}",
            source_id=str(index),
            media_id=str(index),
            captured_at=f"2025-01-0{index + 1}",
            vlm_caption="caption",
        )
        for index in range(3)
    )

    batches = build_hybrid_batches(
        run_id="run-1",
        media_items=media_items,
        max_items_per_batch=2,
        max_chars_per_batch=10_000,
        overlap_items=0,
    )

    coverage = validate_batch_coverage(media_items=media_items, batches=batches)
    assert coverage.ok is True
    assert coverage.covered_source_refs == 3


def test_full_context_plan_cli_outputs_safe_summary(capsys) -> None:
    cli_main(
        [
            "full-context",
            "plan",
            "--profile-id",
            "1",
            "--date-from",
            "2024-12-01",
            "--date-to",
            "2024-12-31",
        ]
    )

    output = capsys.readouterr().out
    assert "Full Context Plan" in output
    assert "recommended_mode: single_context" in output
    assert "batch_count: 0" in output
    assert "[redacted_path]" not in output


def test_full_context_pack_preview_cli_outputs_prompt_metadata(capsys) -> None:
    cli_main(
        [
            "full-context",
            "pack-preview",
            "--profile-id",
            "1",
            "--date-from",
            "2024-12-01",
            "--date-to",
            "2024-12-31",
            "--max-preview-chars",
            "200",
        ]
    )

    output = capsys.readouterr().out
    assert "Full Context Prompt Pack Preview" in output
    assert "prompt_size_chars:" in output
    assert "included_payload_types:" in output
    assert "raw_text_included: false" in output
    assert "prompt_logging_allowed: false" in output
    assert "raw_payload_cache_allowed: false" in output


def test_full_context_analyze_dry_run_does_not_call_llm(capsys) -> None:
    cli_main(
        [
            "full-context",
            "analyze",
            "--profile-id",
            "1",
            "--date-from",
            "2024-12-01",
            "--date-to",
            "2024-12-31",
            "--question",
            "この期間に何があった？",
            "--dry-run",
            "true",
        ]
    )

    output = capsys.readouterr().out
    assert "Full Context Analyze Dry Run" in output
    assert "llm_calls: 0" in output
    assert "local LLM was not called" in output


def test_full_context_runs_list_and_show_cli(tmp_path, capsys) -> None:
    db_path = tmp_path / "relationship.sqlite"
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        f"""
paths:
  relationship_db: "{db_path}"
""",
        encoding="utf-8",
    )
    repo = RelationshipRepository(db_path)
    run_id = repo.create_full_analysis_run(
        question="この期間に何があった？",
        analysis_mode="private_full_range",
        date_from="2025-01-01",
        date_to="2025-01-02",
        manifest={"line_count": 0},
    )

    cli_main(["--config", str(config_path), "full-context", "runs", "list"])
    list_output = capsys.readouterr().out
    assert "id\tstatus\tanalysis_mode" in list_output
    assert str(run_id) in list_output
    assert "private_full_range" in list_output

    cli_main(["--config", str(config_path), "full-context", "runs", "show", "--id", str(run_id)])
    show_output = capsys.readouterr().out
    assert f"id: {run_id}" in show_output
    assert "manifest_keys: line_count" in show_output
    assert "batch_count: 0" in show_output
