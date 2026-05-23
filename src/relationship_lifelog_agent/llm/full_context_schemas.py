from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field


class _StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class FullRangePlanStepSchema(_StrictModel):
    step: str
    purpose: str
    required: bool = True
    expected_source_refs: list[str] = Field(default_factory=list)


class FullRangePlanSchema(_StrictModel):
    question: str
    analysis_mode: Literal["private_full_range", "private_full_corpus"]
    recommended_mode: Literal["single_context", "iterative_full_scan"]
    plan_steps: list[FullRangePlanStepSchema]
    max_llm_calls: int = Field(ge=0)
    cautions: list[str] = Field(default_factory=list)


class FullBatchAnalysisSchema(_StrictModel):
    batch_id: str
    key_findings: list[str] = Field(default_factory=list)
    candidate_events: list[dict[str, Any]] = Field(default_factory=list)
    relevant_evidence_refs: list[str] = Field(default_factory=list)
    uncertainties: list[str] = Field(default_factory=list)
    suggested_next_steps: list[str] = Field(default_factory=list)
    confidence: float = Field(default=0.5, ge=0.0, le=1.0)


class FullRangeSynthesisSchema(_StrictModel):
    summary: str
    answer: str
    timeline: list[dict[str, Any]] = Field(default_factory=list)
    evidence: list[dict[str, Any]] = Field(default_factory=list)
    uncertainties: list[str] = Field(default_factory=list)
    followup_suggestions: list[str] = Field(default_factory=list)
    source_refs: list[str] = Field(default_factory=list)
    confidence: float = Field(default=0.5, ge=0.0, le=1.0)


class FullAnswerSchema(_StrictModel):
    answer_markdown: str
    source_refs: list[str] = Field(default_factory=list)
    cautions: list[str] = Field(default_factory=list)
    confidence: float = Field(default=0.5, ge=0.0, le=1.0)


def json_schema_for(model: type[BaseModel]) -> dict[str, Any]:
    return model.model_json_schema()


FULL_RANGE_PLAN_JSON_SCHEMA = json_schema_for(FullRangePlanSchema)
FULL_BATCH_ANALYSIS_JSON_SCHEMA = json_schema_for(FullBatchAnalysisSchema)
FULL_RANGE_SYNTHESIS_JSON_SCHEMA = json_schema_for(FullRangeSynthesisSchema)
FULL_ANSWER_JSON_SCHEMA = json_schema_for(FullAnswerSchema)
