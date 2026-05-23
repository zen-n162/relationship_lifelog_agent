from __future__ import annotations

import pytest

from relationship_lifelog_agent.config import AnalysisSettings, PrivateFullLlmPayloadSettings, Settings
from relationship_lifelog_agent.full_context.batch_builder import build_chronological_batches
from relationship_lifelog_agent.full_context.full_range_analyzer import (
    FullRangeAnalysisError,
    FullRangeBudgetExceeded,
    analyze_batch,
    analyze_iterative_batches,
    analyze_single_context,
)
from relationship_lifelog_agent.full_context.manifest import build_full_data_manifest
from relationship_lifelog_agent.full_context.prompt_packer import build_batch_prompt, build_single_context_prompt
from relationship_lifelog_agent.full_context.types import FullLineItem
from relationship_lifelog_agent.llm.full_context_schemas import (
    FullBatchAnalysisSchema,
    FullRangePlanSchema,
    FullRangeSynthesisSchema,
    json_schema_for,
)
from relationship_lifelog_agent.llm.local_client import LlmCallResult
from relationship_lifelog_agent.privacy.raw_payload_policy import from_config


class MockLlmClient:
    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []

    def chat(self, messages, *, schema=None, expect_json=True):
        self.calls.append({"messages": messages, "schema": schema, "expect_json": expect_json})
        if not self.responses:
            return LlmCallResult(ok=False, error="no mock response")
        response = self.responses.pop(0)
        if isinstance(response, LlmCallResult):
            return response
        return LlmCallResult(ok=True, data=response, content="{}")


def _private_policy():
    return from_config(
        Settings(
            analysis=AnalysisSettings(mode="private_full_range"),
            private_full_llm_payload=PrivateFullLlmPayloadSettings(allow_raw_line_text=True),
        )
    )


def _safe_policy():
    return from_config(Settings(analysis=AnalysisSettings(mode="safe_window")))


def _line_items():
    return (
        FullLineItem(
            source_ref="line:1",
            source_id="line-1",
            conversation_id="c1",
            timestamp="2025-01-01T10:00:00",
            speaker_id="s1",
            speaker_label="speaker",
            raw_text="raw line one",
        ),
        FullLineItem(
            source_ref="line:2",
            source_id="line-2",
            conversation_id="c1",
            timestamp="2025-01-02T10:00:00",
            speaker_id="s1",
            speaker_label="speaker",
            raw_text="raw line two",
        ),
    )


def _manifest(lines):
    return build_full_data_manifest(
        run_id="run-1",
        profile_id=1,
        date_from="2025-01-01",
        date_to="2025-01-02",
        line_items=lines,
        note_items=(),
        media_items=(),
        face_items=(),
        location_items=(),
    )


def _single_context_prompt(lines, policy):
    manifest = _manifest(lines)
    return build_single_context_prompt(
        question="この期間に何があった？",
        profile_context={"profile_id": 1},
        manifest=manifest,
        full_data={"line_items": lines},
        raw_payload_policy=policy,
    )


def _batch_response(batch_id="batch-1", refs=None):
    return {
        "batch_id": batch_id,
        "key_findings": ["finding"],
        "candidate_events": [],
        "relevant_evidence_refs": refs or ["line:1"],
        "uncertainties": [],
        "suggested_next_steps": [],
        "confidence": 0.8,
    }


def _synthesis_response(refs=None):
    return {
        "summary": "summary",
        "answer": "answer",
        "timeline": [],
        "evidence": [],
        "uncertainties": [],
        "followup_suggestions": [],
        "source_refs": refs or ["line:1"],
        "confidence": 0.75,
    }


def test_full_context_pydantic_schemas_emit_json_schema() -> None:
    assert json_schema_for(FullRangePlanSchema)["type"] == "object"
    assert json_schema_for(FullBatchAnalysisSchema)["type"] == "object"
    assert json_schema_for(FullRangeSynthesisSchema)["type"] == "object"


def test_mock_llm_single_context_analysis_returns_synthesis() -> None:
    lines = _line_items()
    manifest = _manifest(lines)
    prompt = _single_context_prompt(lines, _private_policy())
    llm = MockLlmClient([_synthesis_response(["line:1"])])

    result = analyze_single_context("question", manifest, prompt, llm)

    assert result.summary == "summary"
    assert result.answer == "answer"
    assert result.source_refs == ("line:1",)
    assert len(llm.calls) == 1
    assert llm.calls[0]["schema"]["type"] == "object"


def test_mock_llm_iterative_batch_analysis_and_synthesis() -> None:
    lines = _line_items()
    manifest = _manifest(lines)
    batches = build_chronological_batches(
        run_id="run-1",
        line_items=lines,
        max_items_per_batch=1,
        max_chars_per_batch=10_000,
        overlap_items=0,
    )
    llm = MockLlmClient(
        [
            _batch_response(batches[0].batch_id, ["line:1"]),
            _batch_response(batches[1].batch_id, ["line:2"]),
            _synthesis_response(["line:1", "line:2"]),
        ]
    )

    result = analyze_iterative_batches(
        "question",
        manifest,
        batches,
        llm,
        raw_payload_policy=_private_policy(),
        max_llm_calls=3,
    )

    assert result.source_refs == ("line:1", "line:2")
    assert result.confidence == 0.75
    assert len(llm.calls) == 3


def test_invalid_json_result_falls_back_for_batch_analysis() -> None:
    lines = _line_items()
    manifest = _manifest(lines)
    batch = build_chronological_batches(run_id="run-1", line_items=lines[:1], overlap_items=0)[0]
    prompt = build_batch_prompt("question", {}, manifest, batch, _private_policy())
    llm = MockLlmClient([LlmCallResult(ok=False, error="invalid json")])

    result = analyze_batch(prompt, FullBatchAnalysisSchema, llm_client=llm, retry_count=0)

    assert result.confidence == 0.0
    assert result.result_json["fallback"] is True
    assert "invalid json" in result.uncertainties[0]


def test_source_refs_are_filtered_to_prompt_refs() -> None:
    lines = _line_items()
    manifest = _manifest(lines)
    prompt = _single_context_prompt(lines, _private_policy())
    llm = MockLlmClient([_synthesis_response(["line:1", "unknown:999"])])

    result = analyze_single_context("question", manifest, prompt, llm)

    assert result.source_refs == ("line:1",)


def test_max_llm_calls_is_not_exceeded() -> None:
    lines = _line_items()
    manifest = _manifest(lines)
    batches = build_chronological_batches(
        run_id="run-1",
        line_items=lines,
        max_items_per_batch=1,
        max_chars_per_batch=10_000,
        overlap_items=0,
    )

    with pytest.raises(FullRangeBudgetExceeded):
        analyze_iterative_batches(
            "question",
            manifest,
            batches,
            MockLlmClient([]),
            raw_payload_policy=_private_policy(),
            max_llm_calls=2,
        )


def test_max_runtime_seconds_is_not_exceeded() -> None:
    lines = _line_items()
    manifest = _manifest(lines)
    prompt = _single_context_prompt(lines, _private_policy())

    with pytest.raises(FullRangeBudgetExceeded):
        analyze_single_context(
            "question",
            manifest,
            prompt,
            MockLlmClient([_synthesis_response()]),
            max_runtime_seconds=0.0,
        )


def test_safe_window_prompt_is_rejected_by_analyzer() -> None:
    lines = _line_items()
    manifest = _manifest(lines)
    prompt = _single_context_prompt(lines, _safe_policy())

    with pytest.raises(FullRangeAnalysisError):
        analyze_single_context("question", manifest, prompt, MockLlmClient([]))


def test_private_full_prompt_can_include_raw_payload() -> None:
    prompt = _single_context_prompt(_line_items(), _private_policy())

    assert "raw line one" in prompt.prompt_text
    assert prompt.raw_payload_included is True
