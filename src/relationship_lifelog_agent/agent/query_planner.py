from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

from relationship_lifelog_agent.agent.answerability import AnswerabilityReport
from relationship_lifelog_agent.agent.evidence_contracts import EvidenceContract, evidence_contract_for_intent
from relationship_lifelog_agent.agent.question_understanding import QuestionFrame
from relationship_lifelog_agent.agent.tool_registry import list_tools
from relationship_lifelog_agent.config import Settings
from relationship_lifelog_agent.llm.local_client import LocalLlmClient
from relationship_lifelog_agent.llm.prompts import SYSTEM_POLICY
from relationship_lifelog_agent.llm.schemas import QUERY_PLAN_SCHEMA
from relationship_lifelog_agent.llm.usage import LlmUsageTrace


@dataclass(frozen=True)
class PlanStep:
    tool_name: str
    purpose: str
    required: bool
    params: dict[str, Any]
    output_key: str


@dataclass(frozen=True)
class ConversationQueryPlan:
    question_frame: QuestionFrame
    answerability_report: AnswerabilityReport
    plan_steps: tuple[PlanStep, ...]
    evidence_contract: EvidenceContract
    fallback_response: str | None
    expected_answer_shape: str

    def summary_lines(self) -> list[str]:
        lines = [f"- answerability: {self.answerability_report.status}", f"- expected_answer_shape: {self.expected_answer_shape}"]
        lines.extend(f"- {step.tool_name}: {step.purpose}" for step in self.plan_steps)
        if self.fallback_response:
            lines.append(f"- fallback: {self.fallback_response}")
        return lines

    def to_debug_dict(self) -> dict[str, Any]:
        return {
            "question_frame": asdict(self.question_frame),
            "answerability_report": self.answerability_report.to_dict(),
            "plan_steps": [asdict(step) for step in self.plan_steps],
            "evidence_contract": asdict(self.evidence_contract),
            "fallback_response": self.fallback_response,
            "expected_answer_shape": self.expected_answer_shape,
        }


def build_conversation_query_plan(
    frame: QuestionFrame,
    answerability_report: AnswerabilityReport,
    *,
    settings: Settings | None = None,
    llm_client: LocalLlmClient | None = None,
    usage: LlmUsageTrace | None = None,
) -> ConversationQueryPlan:
    contract = evidence_contract_for_intent(frame.primary_intent)
    if not answerability_report.can_execute:
        return ConversationQueryPlan(
            question_frame=frame,
            answerability_report=answerability_report,
            plan_steps=(PlanStep("explain_missing_information", "不足情報を説明して次の手順を案内します。", True, {}, "fallback_answer"),),
            evidence_contract=contract,
            fallback_response=answerability_report.status,
            expected_answer_shape="missing_information_guidance",
        )
    llm_plan = _build_llm_plan(frame, answerability_report, contract, settings, llm_client, usage)
    if llm_plan is not None:
        return llm_plan
    steps = [PlanStep("resolve_profile", "手動profileを確定します。", True, {}, "profile")]
    if frame.primary_intent == "conflict_dates_with_surrounding_media":
        days_before = getattr(settings.relationship, "surrounding_media_days_before", 1) if settings else 1
        days_after = getattr(settings.relationship, "surrounding_media_days_after", 1) if settings else 1
        steps.extend(
            [
                PlanStep("find_conflict_candidate_dates", "喧嘩・軽いすれ違い候補日をPythonで特定します。", True, {}, "conflict_dates"),
                PlanStep("get_line_window_around_date", "候補日の小さなLINE windowだけをread-onlyで取得します。", False, {"days_before": 1, "days_after": 1}, "line_windows"),
                PlanStep("analyze_line_window_with_llm", "小さなLINE windowをlocal LLMで候補性・冗談可能性・severityに整理します。", False, {}, "line_window_analysis"),
                PlanStep("get_media_by_exact_date_range", "候補日の前後だけに限定してmediaを取得します。", True, {"days_before": days_before, "days_after": days_after}, "media_by_day"),
                PlanStep("summarize_media_day_with_llm", "日別media metadataをlocal LLMで要約します。", False, {}, "media_day_summaries"),
                PlanStep("analyze_note_window_with_llm", "日付近傍noteだけを補助根拠として分析します。", False, {}, "note_window_analysis"),
                PlanStep("summarize_conflict_with_surrounding_media", "喧嘩候補日と前後日の写真summaryを統合します。", True, {}, "analysis_result"),
            ]
        )
    elif frame.primary_intent in {"conflict_frequency", "conflict_timeline", "conflict_date_lookup"}:
        steps.extend(
            [
                PlanStep("search_relationship_events", "保存済みreview済み候補を読みます。", False, {}, "relationship_events"),
                PlanStep("search_line_conflict_signals", "LINE上の喧嘩候補・すれ違い候補をread-onlyで探します。", True, {}, "line_evidence"),
                PlanStep("search_date_near_notes", "イベント近傍かつ関係性のあるnoteだけを補助根拠にします。", False, {}, "note_evidence"),
                PlanStep("conflict_audit", "件数・日付・severityをPythonで集計します。", True, {}, "analysis_result"),
            ]
        )
    elif frame.primary_intent == "post_conflict_activity":
        steps.extend(
            [
                PlanStep("search_relationship_events", "喧嘩候補と外出候補を確認します。", True, {}, "relationship_events"),
                PlanStep("search_media_events", "外出・場所候補をread-onlyで探します。", False, {}, "media_evidence"),
                PlanStep("search_post_conflict_activities", "喧嘩候補後の活動を日数で集計します。", True, {}, "post_conflict_activity"),
            ]
        )
    elif frame.primary_intent == "reply_delay_analysis":
        steps.append(
            PlanStep("calculate_reply_delay", "self/target speakerを使い返信間隔をPython/SQLで計算します。", True, {}, "reply_delay_metrics")
        )
    elif frame.primary_intent == "monthly_relationship_review":
        steps.append(PlanStep("get_monthly_reflection", "月次振り返り候補を取得します。", False, {}, "monthly_reflection"))
    else:
        steps.append(PlanStep("compose_answer", "対応可能範囲を説明します。", True, {}, "answer"))
    steps.append(PlanStep("compose_answer", "事実と推定を分けた安全な日本語回答を作ります。", True, {}, "answer"))
    return ConversationQueryPlan(
        question_frame=frame,
        answerability_report=answerability_report,
        plan_steps=tuple(steps),
        evidence_contract=contract,
        fallback_response=None,
        expected_answer_shape=frame.requested_output,
    )


def _build_llm_plan(
    frame: QuestionFrame,
    answerability_report: AnswerabilityReport,
    contract: EvidenceContract,
    settings: Settings | None,
    llm_client: LocalLlmClient | None,
    usage: LlmUsageTrace | None,
) -> ConversationQueryPlan | None:
    if not (
        settings
        and settings.llm.enabled
        and settings.llm.model
        and settings.llm.use_for_query_planning
        and llm_client
        and llm_client.is_configured()
    ):
        return None
    allowed_tools = tuple(tool.name for tool in list_tools())
    prompt = {
        "task": "Create a safe tool plan for this local relationship evidence review question.",
        "question_frame": asdict(frame),
        "answerability_status": answerability_report.status,
        "allowed_tools": allowed_tools,
        "rules": [
            "Use only allowed_tools.",
            "Do not include external API, web, file write, model download, or identity linking tools.",
            "Counts, dates, profile resolution, and privacy redaction are done by Python.",
        ],
    }
    result = llm_client.chat(
        [
            {"role": "system", "content": SYSTEM_POLICY},
            {"role": "user", "content": str(prompt)},
        ],
        schema=QUERY_PLAN_SCHEMA,
        expect_json=True,
    )
    if not result.ok or not isinstance(result.data, dict):
        _record_plan_fallback(usage, result.latency_ms, result.structured_output_used, result.error or "empty llm plan")
        return None
    try:
        steps = _plan_steps_from_llm(result.data, allowed_tools)
    except ValueError as exc:
        _record_plan_fallback(usage, result.latency_ms, result.structured_output_used, str(exc))
        return None
    if usage:
        usage.record(
            stage="query_planning",
            backend="llm",
            success=True,
            latency_ms=result.latency_ms,
            structured_output_used=result.structured_output_used,
            schema_validation_success=True,
        )
    return ConversationQueryPlan(
        question_frame=frame,
        answerability_report=answerability_report,
        plan_steps=steps,
        evidence_contract=contract,
        fallback_response=None,
        expected_answer_shape=str(result.data.get("expected_answer_shape") or frame.requested_output),
    )


def _plan_steps_from_llm(data: dict[str, Any], allowed_tools: tuple[str, ...]) -> tuple[PlanStep, ...]:
    raw_steps = data.get("plan_steps")
    if not isinstance(raw_steps, list):
        raw_steps = data.get("plan")
    if not isinstance(raw_steps, list):
        raw_steps = data.get("steps")
    if not isinstance(raw_steps, list):
        raw_steps = data.get("tools")
    if not isinstance(raw_steps, list) or not raw_steps:
        raise ValueError("llm returned no plan steps")
    steps: list[PlanStep] = []
    for raw in raw_steps:
        if not isinstance(raw, dict):
            raise ValueError("llm returned a malformed plan step")
        tool_name = _normalize_tool_name(str(raw.get("tool_name") or raw.get("tool") or ""))
        if tool_name not in allowed_tools:
            raise ValueError(f"llm returned disallowed tool: {tool_name}")
        params = raw.get("params")
        if not isinstance(params, dict):
            params = raw.get("inputs")
        steps.append(
            PlanStep(
                tool_name=tool_name,
                purpose=str(raw.get("purpose") or raw.get("description") or "LLM suggested safe local step.")[:240],
                required=bool(raw.get("required", True)),
                params=params if isinstance(params, dict) else {},
                output_key=str(raw.get("output_key") or tool_name)[:80],
            )
        )
    if "compose_answer" not in {step.tool_name for step in steps}:
        steps.append(PlanStep("compose_answer", "事実と推定を分けた安全な日本語回答を作ります。", True, {}, "answer"))
    return tuple(steps)


def _normalize_tool_name(value: str) -> str:
    aliases = {
        "find_conflict_dates": "find_conflict_candidate_dates",
        "search_conflict_candidate_dates": "find_conflict_candidate_dates",
        "get_media_window": "get_media_by_exact_date_range",
        "search_media_by_exact_date_range": "get_media_by_exact_date_range",
        "summarize_media_by_day": "summarize_media_day_with_llm",
        "media_day_summary": "summarize_media_day_with_llm",
        "line_window_analysis": "analyze_line_window_with_llm",
        "note_window_analysis": "analyze_note_window_with_llm",
        "compose_compound_answer": "summarize_conflict_with_surrounding_media",
    }
    return aliases.get(value.strip(), value.strip())


def _record_plan_fallback(
    usage: LlmUsageTrace | None,
    latency_ms: float,
    structured_output_used: bool,
    reason: str,
) -> None:
    if usage:
        usage.record(
            stage="query_planning",
            backend="rule",
            success=False,
            latency_ms=latency_ms,
            structured_output_used=structured_output_used,
            fallback_reason=reason,
        )
