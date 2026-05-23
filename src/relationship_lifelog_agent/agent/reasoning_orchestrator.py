from __future__ import annotations

from dataclasses import asdict, dataclass, replace
import re
from typing import Any

from relationship_lifelog_agent.adapters.types import EvidenceItem
from relationship_lifelog_agent.agent.answerability import AnswerabilityReport, check_answerability
from relationship_lifelog_agent.agent.answer_composer import compose_answer_with_llm
from relationship_lifelog_agent.agent.conversation_state import ConversationState
from relationship_lifelog_agent.agent.executor import ReviewTarget, answer_chat
from relationship_lifelog_agent.agent.information_needs import InformationNeed
from relationship_lifelog_agent.agent.profile_resolver import ProfileResolution, resolve_profile
from relationship_lifelog_agent.agent.query_planner import ConversationQueryPlan, build_conversation_query_plan
from relationship_lifelog_agent.agent.question_understanding import QuestionFrame, understand_question
from relationship_lifelog_agent.config import Settings
from relationship_lifelog_agent.llm.local_client import LocalLlmClient
from relationship_lifelog_agent.llm.usage import LlmUsageTrace
from relationship_lifelog_agent.privacy.guard import sanitize_answer


@dataclass(frozen=True)
class ChatResponse:
    answer_markdown: str
    reasoning_summary: str
    information_needs: tuple[InformationNeed, ...]
    query_plan_summary: str
    evidence_items: tuple[EvidenceItem, ...]
    cautions: tuple[str, ...]
    conversation_state: ConversationState
    review_targets: tuple[ReviewTarget, ...] = ()
    debug_info: dict[str, Any] | None = None


def answer_with_reasoning(
    question: str,
    *,
    settings: Settings,
    mode: str = "private",
    selected_profile_id: int | None = None,
    date_from: str | None = None,
    date_to: str | None = None,
    post_conflict_window_days: int | float | None = None,
    state: ConversationState | dict[str, Any] | None = None,
    memory: object | None = None,
    show_debug: bool = False,
    llm_client: LocalLlmClient | None = None,
) -> ChatResponse:
    current_state = ConversationState.from_value(state)
    usage = LlmUsageTrace.from_settings(settings.llm)
    client = llm_client or LocalLlmClient(settings.llm)
    frame = understand_question(question, current_state, settings=settings, llm_client=client, usage=usage)
    frame = _override_frame_dates(frame, date_from=date_from, date_to=date_to)
    profile_resolution = resolve_profile(settings, frame, selected_profile_id=selected_profile_id)
    answerability = check_answerability(frame, profile_resolution, settings, llm_client=client, usage=usage)
    plan = build_conversation_query_plan(frame, answerability, settings=settings, llm_client=client, usage=usage)
    if not answerability.can_execute:
        answer = _missing_information_answer(frame, profile_resolution, answerability, mode=mode)
        answer = _append_llm_usage_summary(answer, usage)
        response_state = _updated_state(
            current_state,
            frame=frame,
            profile_resolution=profile_resolution,
            answerability=answerability,
            plan=plan,
            answer_markdown=answer,
            review_targets=(),
        )
        return _response(answer, frame, answerability, plan, response_state, (), mode=mode, show_debug=show_debug, usage=usage)

    chat_answer = answer_chat(
        frame.raw_question,
        mode=mode,
        memory=memory,
        settings=settings,
        profile_id=profile_resolution.profile_id,
        date_from=frame.date_from,
        date_to=frame.date_to,
        post_conflict_window_days=post_conflict_window_days,
    )
    person_names = [profile_resolution.profile.profile_name] if profile_resolution.profile else None
    composed = compose_answer_with_llm(
        chat_answer.text,
        question=frame.raw_question,
        settings=settings,
        llm_client=client,
        usage=usage,
        mode=mode,
        person_names=person_names,
    )
    answer = _append_reasoning_sections(composed, answerability, plan, usage=usage, mode=mode, show_debug=show_debug)
    response_state = _updated_state(
        current_state,
        frame=frame,
        profile_resolution=profile_resolution,
        answerability=answerability,
        plan=plan,
        answer_markdown=answer,
        review_targets=chat_answer.review_targets,
    )
    return _response(
        answer,
        frame,
        answerability,
        plan,
        response_state,
        chat_answer.review_targets,
        mode=mode,
        show_debug=show_debug,
        usage=usage,
    )


def _override_frame_dates(frame: QuestionFrame, *, date_from: str | None, date_to: str | None) -> QuestionFrame:
    clean_from = _clean_text(date_from)
    clean_to = _clean_text(date_to)
    if not clean_from and not clean_to:
        return frame
    return replace(frame, date_from=clean_from or frame.date_from, date_to=clean_to or frame.date_to)


def _missing_information_answer(
    frame: QuestionFrame,
    profile_resolution: ProfileResolution,
    answerability: AnswerabilityReport,
    *,
    mode: str,
) -> str:
    lines = ["要約:"]
    if answerability.status == "missing_profile":
        target = frame.target_profile_name or "対象人物"
        lines.append(
            f"relationship profile が手動設定されていないため、この質問に答えるには、{target} の手動profile設定が必要です。"
        )
    elif answerability.status == "ambiguous_profile":
        ids = ", ".join(f"id={row['id']}" for row in profile_resolution.candidates)
        lines.append(f"{frame.target_profile_name or '指定名'} というprofileが複数あります。{ids} のどれを使うか選んでください。")
    elif answerability.status == "missing_self_speaker":
        lines.append("この質問は一部回答できません。返信方向を判定するには、あなた側のLINE speaker IDが必要です。")
    elif answerability.status == "missing_target_speaker":
        lines.append("対象側のLINE speaker IDが未設定のため、LINE根拠を安全に絞り込めません。")
    elif answerability.status == "missing_upstream_adapter":
        lines.append("upstream_readonly adapterのDB pathが未設定または未接続です。")
    else:
        lines.append("この質問は現在の安全ルールでは回答できません。")
    lines.extend(
        [
            "",
            "集計:\n- answerability: " + answerability.status,
            "具体例:\n" + _needs_markdown(answerability.missing_needs),
            "根拠:\n<details>\n<summary>確認した情報</summary>\n\n" + _needs_markdown(answerability.available_needs) + "\n\n</details>",
            "信頼度:\nvery weak (confidence=0.00, evidence_strength=0.00)",
            "注意:\n<details>\n<summary>注意を表示</summary>\n\n- AIは恋人ラベルを推定しません。\n- AIはpersonとLINE speakerの対応を推定しません。\n- 上流データはread-onlyで扱います。\n\n</details>",
        ]
    )
    return sanitize_answer("\n".join(lines), mode=mode, person_names=[frame.target_profile_name] if frame.target_profile_name else None)


def _append_reasoning_sections(
    answer: str,
    answerability: AnswerabilityReport,
    plan: ConversationQueryPlan,
    *,
    usage: LlmUsageTrace,
    mode: str,
    show_debug: bool,
) -> str:
    sections = [
        "LLM利用:\n- " + usage.user_summary(),
        "<details>\n<summary>確認した情報</summary>\n\n" + _needs_markdown(answerability.available_needs) + "\n\n</details>",
        "<details>\n<summary>必要だった情報</summary>\n\n" + _needs_markdown(answerability.needs) + "\n\n</details>",
        "<details>\n<summary>実行した検索方針</summary>\n\n" + "\n".join(plan.summary_lines()) + "\n\n</details>",
    ]
    if answerability.missing_needs:
        sections.append("<details>\n<summary>足りない情報</summary>\n\n" + _needs_markdown(answerability.missing_needs) + "\n\n</details>")
    if show_debug:
        debug_payload = {**plan.to_debug_dict(), "llm_usage": usage.to_dict()}
        sections.append("<details>\n<summary>Debug reasoning</summary>\n\n```text\n" + str(debug_payload) + "\n```\n\n</details>")
    return sanitize_answer(answer + "\n\n" + "\n\n".join(sections), mode=mode)


def _append_llm_usage_summary(answer: str, usage: LlmUsageTrace) -> str:
    return answer + "\n\nLLM利用:\n- " + usage.user_summary()


def _response(
    answer_markdown: str,
    frame: QuestionFrame,
    answerability: AnswerabilityReport,
    plan: ConversationQueryPlan,
    state: ConversationState,
    review_targets: tuple[ReviewTarget, ...],
    *,
    mode: str,
    show_debug: bool,
    usage: LlmUsageTrace,
) -> ChatResponse:
    debug_info = {**plan.to_debug_dict(), "llm_usage": usage.to_dict()} if show_debug else None
    return ChatResponse(
        answer_markdown=sanitize_answer(answer_markdown, mode=mode),
        reasoning_summary=f"{frame.primary_intent}: {answerability.status}",
        information_needs=answerability.needs,
        query_plan_summary="\n".join(plan.summary_lines()),
        evidence_items=(),
        cautions=tuple(answerability.warnings),
        conversation_state=state,
        review_targets=review_targets,
        debug_info=debug_info,
    )


def _updated_state(
    state: ConversationState,
    *,
    frame: QuestionFrame,
    profile_resolution: ProfileResolution,
    answerability: AnswerabilityReport,
    plan: ConversationQueryPlan,
    answer_markdown: str,
    review_targets: tuple[ReviewTarget, ...],
) -> ConversationState:
    observed_dates = _extract_observed_dates(answer_markdown) or tuple(
        date for date in (frame.date_from, frame.date_to) if date
    )
    return ConversationState(
        active_profile_id=profile_resolution.profile_id or state.active_profile_id,
        active_profile_name=profile_resolution.profile.profile_name if profile_resolution.profile else state.active_profile_name,
        last_question=frame.raw_question,
        last_intents=frame.intents,
        last_date_range=(frame.date_from, frame.date_to),
        last_observed_dates=observed_dates,
        last_event_ids=tuple(target.event_id for target in review_targets),
        last_evidence_ids=state.last_evidence_ids,
        last_answer_summary=_extract_summary(answer_markdown),
        pending_setup_requirements=tuple(item.name for item in answerability.missing_needs),
        pending_review_targets=tuple(target.event_id for target in review_targets),
        last_query_plan=plan.to_debug_dict(),
        last_answerability_report=answerability.to_dict(),
    )


def _needs_markdown(needs: tuple[InformationNeed, ...]) -> str:
    if not needs:
        return "- なし"
    return "\n".join(
        f"- {need.name}: {need.status} / {need.reason}"
        + (f" / next: {need.how_to_get}" if need.how_to_get and need.status in {"missing", "ambiguous"} else "")
        for need in needs
    )


def _extract_observed_dates(text: str) -> tuple[str, ...]:
    dates = re.findall(r"\b20\d{2}-\d{2}-\d{2}\b", text)
    return tuple(dict.fromkeys(dates))


def _extract_summary(text: str) -> str | None:
    if "要約:" not in text:
        return None
    body = text.split("要約:", 1)[1].split("\n\n", 1)[0]
    clean = " ".join(body.split())
    return clean[:240] or None


def _clean_text(value: str | None) -> str | None:
    if value is None:
        return None
    stripped = value.strip()
    return stripped or None
