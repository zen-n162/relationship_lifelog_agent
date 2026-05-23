from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable

from pydantic import BaseModel, ValidationError

from relationship_lifelog_agent.full_context.prompt_packer import (
    build_batch_prompt,
    build_synthesis_prompt,
)
from relationship_lifelog_agent.full_context.types import (
    FullBatchAnalysis,
    FullDataManifest,
    FullPromptBundle,
    FullRangeSynthesis,
    FullScanBatch,
)
from relationship_lifelog_agent.llm.full_context_schemas import (
    FullBatchAnalysisSchema,
    FullRangeSynthesisSchema,
    json_schema_for,
)
from relationship_lifelog_agent.llm.local_client import LlmCallResult
from relationship_lifelog_agent.privacy.raw_payload_policy import RawPayloadPolicy


PRIVATE_FULL_MODES = frozenset({"private_full_range", "private_full_corpus"})


class FullRangeAnalysisError(RuntimeError):
    pass


class FullRangeBudgetExceeded(FullRangeAnalysisError):
    pass


@dataclass
class LlmCallBudget:
    max_llm_calls: int = 20
    used_llm_calls: int = 0

    def consume(self) -> None:
        if self.used_llm_calls >= self.max_llm_calls:
            raise FullRangeBudgetExceeded(f"max_llm_calls exceeded: {self.max_llm_calls}")
        self.used_llm_calls += 1


def analyze_single_context(
    question: str,
    manifest: FullDataManifest,
    prompt_bundle: FullPromptBundle,
    llm_client: Any,
    *,
    max_llm_calls: int = 1,
    retry_count: int = 1,
    fallback_on_error: bool = True,
) -> FullRangeSynthesis:
    _ensure_private_full_prompt(prompt_bundle)
    budget = LlmCallBudget(max_llm_calls=max_llm_calls)
    result = _call_structured_llm(
        prompt_bundle=prompt_bundle,
        schema_model=FullRangeSynthesisSchema,
        llm_client=llm_client,
        budget=budget,
        retry_count=retry_count,
    )
    if result.ok and result.validated is not None:
        return _synthesis_from_schema(result.validated, manifest, prompt_bundle.source_refs)
    if fallback_on_error:
        return _fallback_synthesis(question, manifest, (), result.error or "single context analysis failed")
    raise FullRangeAnalysisError(result.error or "single context analysis failed")


def analyze_iterative_batches(
    question: str,
    manifest: FullDataManifest,
    batches: Iterable[FullScanBatch],
    llm_client: Any,
    *,
    raw_payload_policy: RawPayloadPolicy | None = None,
    profile_context: dict[str, Any] | None = None,
    max_llm_calls: int = 20,
    retry_count: int = 1,
    fallback_on_error: bool = True,
) -> FullRangeSynthesis:
    policy = raw_payload_policy or _policy_from_manifest(manifest)
    if policy.analysis_mode not in PRIVATE_FULL_MODES:
        raise FullRangeAnalysisError("full range analyzer requires private_full_range or private_full_corpus")
    batch_tuple = tuple(batches)
    if len(batch_tuple) + 1 > max_llm_calls:
        raise FullRangeBudgetExceeded(f"max_llm_calls={max_llm_calls} is smaller than required calls={len(batch_tuple) + 1}")
    budget = LlmCallBudget(max_llm_calls=max_llm_calls)
    analyses: list[FullBatchAnalysis] = []
    for batch in batch_tuple:
        batch_prompt = build_batch_prompt(
            question=question,
            profile_context=profile_context or {},
            manifest=manifest,
            batch=batch,
            raw_payload_policy=policy,
        )
        analyses.append(
            analyze_batch(
                batch_prompt,
                FullBatchAnalysisSchema,
                llm_client=llm_client,
                budget=budget,
                retry_count=retry_count,
                fallback_on_error=fallback_on_error,
            )
        )
    return synthesize_full_range(
        question=question,
        manifest=manifest,
        batch_analyses=analyses,
        llm_client=llm_client,
        budget=budget,
        retry_count=retry_count,
        fallback_on_error=fallback_on_error,
    )


def analyze_batch(
    batch_prompt: FullPromptBundle,
    schema: type[BaseModel] = FullBatchAnalysisSchema,
    *,
    llm_client: Any,
    budget: LlmCallBudget | None = None,
    retry_count: int = 1,
    fallback_on_error: bool = True,
) -> FullBatchAnalysis:
    _ensure_private_full_prompt(batch_prompt)
    budget = budget or LlmCallBudget()
    result = _call_structured_llm(
        prompt_bundle=batch_prompt,
        schema_model=schema,
        llm_client=llm_client,
        budget=budget,
        retry_count=retry_count,
    )
    if result.ok and result.validated is not None:
        if isinstance(result.validated, FullBatchAnalysisSchema):
            return _batch_analysis_from_schema(result.validated, batch_prompt.source_refs)
        raise FullRangeAnalysisError(f"unsupported batch schema: {schema.__name__}")
    if fallback_on_error:
        return FullBatchAnalysis(
            batch_id=batch_prompt.bundle_id,
            uncertainties=(result.error or "batch analysis failed",),
            relevant_evidence_refs=batch_prompt.source_refs,
            confidence=0.0,
            result_json={"fallback": True, "error": result.error, "llm_calls_used": budget.used_llm_calls},
        )
    raise FullRangeAnalysisError(result.error or "batch analysis failed")


def synthesize_full_range(
    question: str,
    manifest: FullDataManifest,
    batch_analyses: Iterable[FullBatchAnalysis],
    llm_client: Any,
    *,
    budget: LlmCallBudget | None = None,
    max_llm_calls: int = 20,
    retry_count: int = 1,
    fallback_on_error: bool = True,
) -> FullRangeSynthesis:
    analyses = tuple(batch_analyses)
    budget = budget or LlmCallBudget(max_llm_calls=max_llm_calls)
    prompt_bundle = build_synthesis_prompt(question, manifest, analyses)
    result = _call_structured_llm(
        prompt_bundle=prompt_bundle,
        schema_model=FullRangeSynthesisSchema,
        llm_client=llm_client,
        budget=budget,
        retry_count=retry_count,
    )
    if result.ok and result.validated is not None:
        return _synthesis_from_schema(result.validated, manifest, prompt_bundle.source_refs)
    if fallback_on_error:
        return _fallback_synthesis(question, manifest, analyses, result.error or "synthesis failed")
    raise FullRangeAnalysisError(result.error or "synthesis failed")


@dataclass(frozen=True)
class _ValidatedResult:
    ok: bool
    validated: BaseModel | None = None
    error: str | None = None
    raw_data: dict[str, Any] | None = None


def _call_structured_llm(
    *,
    prompt_bundle: FullPromptBundle,
    schema_model: type[BaseModel],
    llm_client: Any,
    budget: LlmCallBudget,
    retry_count: int,
) -> _ValidatedResult:
    last_error: str | None = None
    attempts = max(1, retry_count + 1)
    for _ in range(attempts):
        budget.consume()
        result = llm_client.chat(
            [{"role": "user", "content": prompt_bundle.prompt_text}],
            schema=json_schema_for(schema_model),
            expect_json=True,
        )
        if not getattr(result, "ok", False):
            last_error = getattr(result, "error", None) or "llm call failed"
            continue
        data = getattr(result, "data", None)
        if not isinstance(data, dict):
            last_error = "llm returned no JSON object"
            continue
        try:
            validated = schema_model.model_validate(data)
        except ValidationError as exc:
            last_error = f"schema validation failed: {exc.errors()[0]['msg'] if exc.errors() else exc}"
            continue
        return _ValidatedResult(ok=True, validated=validated, raw_data=data)
    return _ValidatedResult(ok=False, error=last_error or "structured LLM validation failed")


def _ensure_private_full_prompt(prompt_bundle: FullPromptBundle) -> None:
    if prompt_bundle.analysis_mode not in PRIVATE_FULL_MODES:
        raise FullRangeAnalysisError("full range analyzer cannot run in safe_window mode")


def _batch_analysis_from_schema(schema: FullBatchAnalysisSchema, allowed_source_refs: tuple[str, ...]) -> FullBatchAnalysis:
    refs = _filter_source_refs(schema.relevant_evidence_refs, allowed_source_refs)
    return FullBatchAnalysis(
        batch_id=schema.batch_id,
        key_findings=tuple(schema.key_findings),
        candidate_events=tuple(schema.candidate_events),
        relevant_evidence_refs=refs,
        uncertainties=tuple(schema.uncertainties),
        suggested_next_steps=tuple(schema.suggested_next_steps),
        confidence=schema.confidence,
        result_json=schema.model_dump(),
    )


def _synthesis_from_schema(
    schema: FullRangeSynthesisSchema | BaseModel,
    manifest: FullDataManifest,
    allowed_source_refs: tuple[str, ...],
) -> FullRangeSynthesis:
    if not isinstance(schema, FullRangeSynthesisSchema):
        schema = FullRangeSynthesisSchema.model_validate(schema.model_dump())
    refs = _filter_source_refs(schema.source_refs, allowed_source_refs)
    return FullRangeSynthesis(
        summary=schema.summary,
        answer=schema.answer,
        timeline=tuple(schema.timeline),
        evidence=tuple(schema.evidence),
        uncertainties=tuple(schema.uncertainties),
        followup_suggestions=tuple(schema.followup_suggestions),
        source_refs=refs,
        confidence=schema.confidence,
        result_json={**schema.model_dump(), "manifest_run_id": manifest.run_id},
    )


def _fallback_synthesis(
    question: str,
    manifest: FullDataManifest,
    batch_analyses: tuple[FullBatchAnalysis, ...],
    error: str,
) -> FullRangeSynthesis:
    refs = tuple(dict.fromkeys(ref for analysis in batch_analyses for ref in analysis.relevant_evidence_refs))
    findings = [finding for analysis in batch_analyses for finding in analysis.key_findings]
    summary = "LLM structured analysis fallback was used."
    answer = (
        f"The private full range could not be fully analyzed by the local LLM for question: {question}. "
        "No unverified facts were added."
    )
    if findings:
        answer += " Structured batch findings are available for synthesis."
    return FullRangeSynthesis(
        summary=summary,
        answer=answer,
        uncertainties=(error,),
        source_refs=refs,
        confidence=0.0,
        result_json={
            "fallback": True,
            "error": error,
            "manifest_run_id": manifest.run_id,
            "batch_count": len(batch_analyses),
        },
    )


def _filter_source_refs(refs: Iterable[str], allowed_refs: tuple[str, ...]) -> tuple[str, ...]:
    allowed = set(allowed_refs)
    return tuple(dict.fromkeys(ref for ref in refs if ref in allowed))


def _policy_from_manifest(manifest: FullDataManifest) -> RawPayloadPolicy:
    data = dict(manifest.raw_payload_policy or {})
    if "analysis_mode" not in data:
        data["analysis_mode"] = "private_full_range"
    allowed = {
        "analysis_mode",
        "allow_raw_line_text",
        "allow_raw_note_text",
        "allow_photo_paths",
        "allow_exact_gps",
        "allow_face_crops",
        "allow_face_embeddings",
        "allow_raw_face_embedding_values",
        "allow_private_file_paths",
        "allow_unverified_person_candidates",
        "allow_unverified_speaker_candidates",
        "allow_full_prompt_logging",
        "allow_raw_payload_cache",
    }
    return RawPayloadPolicy(**{key: value for key, value in data.items() if key in allowed})
