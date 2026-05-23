from __future__ import annotations

from dataclasses import asdict, dataclass, replace
import re
from typing import Any

from relationship_lifelog_agent.adapters.types import EvidenceItem
from relationship_lifelog_agent.agent.answerability import AnswerabilityReport, check_answerability
from relationship_lifelog_agent.agent.answer_composer import compose_answer_with_llm
from relationship_lifelog_agent.agent.agent_runtime import PRIVATE_FULL_MODES, run_agent, stream_run_agent
from relationship_lifelog_agent.agent.conversation_state import ConversationState
from relationship_lifelog_agent.agent.executor import ReviewTarget, answer_chat
from relationship_lifelog_agent.agent.information_needs import InformationNeed
from relationship_lifelog_agent.agent.profile_resolver import ProfileResolution, resolve_profile
from relationship_lifelog_agent.agent.query_planner import ConversationQueryPlan, build_conversation_query_plan
from relationship_lifelog_agent.agent.question_understanding import QuestionFrame, understand_question
from relationship_lifelog_agent.agent.streaming import AgentStreamEvent
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


def stream_answer(
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
):
    current_state = ConversationState.from_value(state)
    if settings.analysis.mode in PRIVATE_FULL_MODES:
        yield from stream_run_agent(
            question,
            current_state,
            settings,
            selected_profile_id=selected_profile_id,
            date_from=date_from,
            date_to=date_to,
            llm_client=llm_client,
        )
        return
    usage = LlmUsageTrace.from_settings(settings.llm)
    client = llm_client or LocalLlmClient(settings.llm)

    yield _event("progress", "質問を整理しています", "質問のintent、対象profile、日付範囲、省略参照を安全に整理します。")
    if _stage_llm_enabled(settings, "question_understanding", client):
        yield _event("llm_start", "local LLMで質問を理解しています", _llm_message(settings, "質問のintentと必要情報をJSON形式で整理します。"))
    frame = understand_question(question, current_state, settings=settings, llm_client=client, usage=usage)
    frame = _override_frame_dates(frame, date_from=date_from, date_to=date_to)
    yield _event(
        "progress",
        "質問を整理しました",
        f"intent={frame.primary_intent}, requested_output={frame.requested_output} として扱います。",
        status="done",
    )
    yield from _llm_stage_events(usage, "question_understanding", "質問理解")

    yield _event("progress", "会話文脈を確認しています", "直前の観測日や選択中profileを確認します。")
    context_message = "直前回答の観測日を参照できます。" if current_state.last_observed_dates else "直前回答の観測日はまだありません。"
    yield _event("progress", "会話文脈を確認しました", context_message, status="done")

    yield _event("progress", "手動profileを確認しています", "AI推定ではなく、保存済みの手動profileだけを使います。")
    profile_resolution = resolve_profile(settings, frame, selected_profile_id=selected_profile_id)
    profile_message = (
        f"profile_id={profile_resolution.profile_id} を使います。"
        if profile_resolution.profile_id
        else f"profile status={profile_resolution.status} です。"
    )
    yield _event("progress", "手動profileを確認しました", profile_message, status="done")

    yield _event("progress", "回答に必要な情報を確認しています", "profile、speaker、adapter、日付範囲、根拠の有無を確認します。")
    if _stage_llm_enabled(settings, "information_needs", client):
        yield _event("llm_start", "local LLMで必要情報を整理しています", _llm_message(settings, "必要情報の候補だけをJSON形式で整理します。"))
    answerability = check_answerability(frame, profile_resolution, settings, llm_client=client, usage=usage)
    yield _event(
        "progress",
        "回答に必要な情報を確認しました",
        f"answerability={answerability.status}, missing_needs={len(answerability.missing_needs)}",
        status="done",
    )
    yield from _llm_stage_events(usage, "information_needs", "必要情報整理")

    yield _event("progress", "検索方針を作成しています", "使えるtoolとEvidence Contractを確認します。")
    if _stage_llm_enabled(settings, "query_planning", client):
        yield _event("llm_start", "local LLMで検索計画を作成しています", _llm_message(settings, "許可済みtoolだけを使う検索計画をJSON形式で作ります。"))
    plan = build_conversation_query_plan(frame, answerability, settings=settings, llm_client=client, usage=usage)
    yield _event("progress", "実行できるtoolか検証しました", f"plan_steps={len(plan.plan_steps)}", status="done")
    yield from _llm_stage_events(usage, "query_planning", "検索計画")

    if not answerability.can_execute:
        yield _event("warning", "不足情報があります", "検索は実行せず、必要な手動設定を案内します。", status="done")
        answer = _append_llm_usage_summary(_missing_information_answer(frame, profile_resolution, answerability, mode=mode), usage)
        response_state = _updated_state(
            current_state,
            frame=frame,
            profile_resolution=profile_resolution,
            answerability=answerability,
            plan=plan,
            answer_markdown=answer,
            review_targets=(),
        )
        response = _response(answer, frame, answerability, plan, response_state, (), mode=mode, show_debug=show_debug, usage=usage)
        yield _final_event(response)
        return

    if frame.primary_intent == "conflict_dates_with_surrounding_media":
        before_days = getattr(settings.relationship, "surrounding_media_days_before", 1)
        after_days = getattr(settings.relationship, "surrounding_media_days_after", 1)
        yield _event("progress", "複合質問として整理しています", "喧嘩候補日を探し、その前後日の写真・メディアsummaryを別々に確認します。")
        yield _event("tool_start", "喧嘩候補日を探しています", "LINE・保存済みeventから喧嘩候補と軽いすれ違い候補の日付をPythonで特定します。")
        yield _event(
            "tool_start",
            f"候補日の前後{before_days}日/{after_days}日を写真から確認しています",
            "post-conflict windowではなく、候補日ごとの厳密な日付範囲だけをread-onlyで確認します。",
        )
        yield _event("tool_start", "日別media summaryを作成しています", "写真パス、正確GPS、顔crop、embeddingは使わず、短いmetadataだけを要約します。")
    else:
        yield _event("tool_start", "LINE記録をread-onlyで検索しています", "対象speakerとself speakerを使い、喧嘩・すれ違い候補を安全に探します。")
        yield _event("tool_start", "日付近傍のメモを確認しています", "日付不明・関係性が低いnoteは通常根拠から除外します。")
    yield _event("progress", "件数を集計しています", "中程度以上の喧嘩候補と軽いすれ違い候補を分けてPythonで集計します。")
    chat_answer = answer_chat(
        frame.raw_question,
        mode=mode,
        memory=memory,
        settings=settings,
        profile_id=profile_resolution.profile_id,
        date_from=frame.date_from,
        date_to=frame.date_to,
        post_conflict_window_days=post_conflict_window_days,
        llm_client=client,
        usage=usage,
    )
    if frame.primary_intent == "conflict_dates_with_surrounding_media":
        observed_dates = _extract_observed_dates(chat_answer.text)
        for observed_date in observed_dates[:3]:
            yield _event("tool_done", f"{observed_date} の前後日を確認しました", "media summaryは候補日の前日・当日・翌日に限定しています。", status="done")
        yield _event("tool_done", "LINEと写真の根拠を統合しました", _candidate_summary(chat_answer.text), status="done")
    else:
        yield _event("tool_done", "LINE記録の検索が完了しました", _candidate_summary(chat_answer.text), status="done")
        yield _event("tool_done", "補助根拠の確認が完了しました", "関係性・日付近傍・根拠強度で通常表示の根拠を絞りました。", status="done")
    yield _event("progress", "根拠の強さを確認しています", "Evidence Contractに合わない根拠や弱すぎる根拠を通常回答から外します。", status="done")

    if _stage_llm_enabled(settings, "answer_composition", client):
        yield _event("llm_start", "local LLMで回答文を整えています", _llm_message(settings, "Pythonで確定した件数・日付を変えずに回答文だけを整えます。"))
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
    yield from _llm_stage_events(usage, "answer_composition", "回答文整形")
    yield _event("progress", "安全チェックをしています", "禁止表現、public mode redaction、過度な断定がないか確認します。")
    answer = _append_reasoning_sections(composed, answerability, plan, usage=usage, mode=mode, show_debug=show_debug)
    yield _event("progress", "安全チェックが完了しました", "最終回答はSafety/Privacy Guardを通過しました。", status="done")
    response_state = _updated_state(
        current_state,
        frame=frame,
        profile_resolution=profile_resolution,
        answerability=answerability,
        plan=plan,
        answer_markdown=answer,
        review_targets=chat_answer.review_targets,
    )
    response = _response(
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
    yield _final_event(response)


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
    if settings.analysis.mode in PRIVATE_FULL_MODES:
        runtime_run = run_agent(
            question,
            current_state,
            settings,
            selected_profile_id=selected_profile_id,
            date_from=date_from,
            date_to=date_to,
            llm_client=llm_client,
        )
        return _runtime_response(runtime_run, mode=mode, show_debug=show_debug)
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
        llm_client=client,
        usage=usage,
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


def _runtime_response(runtime_run: Any, *, mode: str, show_debug: bool) -> ChatResponse:
    debug_info = runtime_run.debug_info if show_debug else None
    return ChatResponse(
        answer_markdown=sanitize_answer(runtime_run.answer_markdown, mode=mode),
        reasoning_summary=f"private_full_runtime: {runtime_run.status}",
        information_needs=(),
        query_plan_summary="\n".join(f"- {task_id}" for task_id in runtime_run.completed_task_order),
        evidence_items=(),
        cautions=tuple([runtime_run.stop_reason] if runtime_run.stop_reason else ()),
        conversation_state=runtime_run.conversation_state,
        review_targets=(),
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


def _event(
    kind: str,
    title: str,
    message: str,
    *,
    status: str = "pending",
    debug_only: bool = False,
    metadata: dict[str, Any] | None = None,
) -> AgentStreamEvent:
    return AgentStreamEvent(
        kind=kind,  # type: ignore[arg-type]
        title=title,
        message=sanitize_answer(message),
        status=status,  # type: ignore[arg-type]
        safe_for_user=True,
        debug_only=debug_only,
        metadata=metadata or {},
    )


def _stage_llm_enabled(settings: Settings, stage: str, client: LocalLlmClient) -> bool:
    if not (settings.llm.enabled and settings.llm.model and client.is_configured()):
        return False
    return bool(getattr(settings.llm, f"use_for_{stage}", False))


def _llm_message(settings: Settings, message: str) -> str:
    return f"{settings.llm.model or 'local model'} を使って、{message} 件数や日付はPythonで確認します。"


def _llm_stage_events(usage: LlmUsageTrace, stage: str, label: str):
    backend = getattr(usage, f"{stage}_backend", "rule")
    reason_prefix = stage + ": "
    stage_reasons = [reason.removeprefix(reason_prefix) for reason in usage.fallback_reasons if reason.startswith(reason_prefix)]
    if backend == "llm":
        yield _event("llm_done", f"local LLMの{label}が完了しました", f"{label}に成功しました。確定処理はPython側で続けます。", status="done")
    elif stage_reasons:
        yield _event("warning", f"{label}はrule-basedに切り替えました", "LLM出力の検証に失敗したため、安全なrule-based fallbackで続行します。", status="done")


def _candidate_summary(answer_markdown: str) -> str:
    fields = ("conflict_candidates", "minor_misunderstanding_candidates", "total_candidates")
    values: list[str] = []
    for field in fields:
        match = re.search(rf"- {re.escape(field)}: ([^\n]+)", answer_markdown)
        if match:
            values.append(f"{field}={match.group(1).strip()}")
    if values:
        return " / ".join(values)
    return "候補件数を集計しました。"


def _final_event(response: ChatResponse) -> AgentStreamEvent:
    review_targets = [
        {
            "event_id": target.event_id,
            "label": target.label,
            "event_type": target.event_type,
            "review_status": target.review_status,
        }
        for target in response.review_targets
    ]
    return AgentStreamEvent(
        kind="final_answer",
        title="回答を作成しました",
        message=response.answer_markdown,
        status="done",
        safe_for_user=True,
        metadata={
            "conversation_state": response.conversation_state.to_dict(),
            "review_targets": review_targets,
            "debug_info": response.debug_info,
        },
    )
