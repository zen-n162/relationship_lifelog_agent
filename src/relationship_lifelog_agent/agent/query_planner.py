from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

from relationship_lifelog_agent.agent.answerability import AnswerabilityReport
from relationship_lifelog_agent.agent.evidence_contracts import EvidenceContract, evidence_contract_for_intent
from relationship_lifelog_agent.agent.question_understanding import QuestionFrame


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
    steps = [PlanStep("resolve_profile", "手動profileを確定します。", True, {}, "profile")]
    if frame.primary_intent in {"conflict_frequency", "conflict_timeline"}:
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
