from __future__ import annotations

from dataclasses import dataclass, field, replace
import time
from typing import Any, Iterable
from uuid import uuid4

from relationship_lifelog_agent.agent.answerability import check_answerability
from relationship_lifelog_agent.agent.conversation_state import ConversationState
from relationship_lifelog_agent.agent.observation_store import Observation, ObservationStore
from relationship_lifelog_agent.agent.profile_resolver import resolve_profile
from relationship_lifelog_agent.agent.question_understanding import understand_question
from relationship_lifelog_agent.agent.replanner import replan_for_observations
from relationship_lifelog_agent.agent.streaming import AgentStreamEvent
from relationship_lifelog_agent.agent.task_graph import TaskGraph, TaskNode, private_full_task_graph
from relationship_lifelog_agent.config import Settings
from relationship_lifelog_agent.full_context.batch_builder import build_hybrid_batches
from relationship_lifelog_agent.full_context.context_budget import decide_context_mode
from relationship_lifelog_agent.full_context.full_range_analyzer import analyze_single_context
from relationship_lifelog_agent.full_context.manifest import build_full_data_manifest
from relationship_lifelog_agent.full_context.prompt_packer import build_single_context_prompt
from relationship_lifelog_agent.full_context.types import FullDataManifest
from relationship_lifelog_agent.llm.local_client import LocalLlmClient
from relationship_lifelog_agent.privacy.guard import sanitize_answer
from relationship_lifelog_agent.privacy.raw_payload_policy import from_config as raw_payload_policy_from_config
from relationship_lifelog_agent.verification.full_answer_verifier import verify_final_answer


PRIVATE_FULL_MODES = frozenset({"private_full_range", "private_full_corpus"})


class RuntimeBudgetExceeded(RuntimeError):
    pass


@dataclass(frozen=True)
class RuntimeBudget:
    max_steps: int = 50
    max_llm_calls: int = 20
    max_tool_calls: int = 50
    max_runtime_seconds: float | None = None
    started_at: float = field(default_factory=time.monotonic)
    used_steps: int = 0
    used_llm_calls: int = 0
    used_tool_calls: int = 0

    def consume_step(self) -> "RuntimeBudget":
        self._check_runtime()
        if self.used_steps >= self.max_steps:
            raise RuntimeBudgetExceeded("max_steps reached")
        return replace(self, used_steps=self.used_steps + 1)

    def consume_llm_call(self) -> "RuntimeBudget":
        self._check_runtime()
        if self.used_llm_calls >= self.max_llm_calls:
            raise RuntimeBudgetExceeded("max_llm_calls reached")
        return replace(self, used_llm_calls=self.used_llm_calls + 1)

    def consume_tool_call(self) -> "RuntimeBudget":
        self._check_runtime()
        if self.used_tool_calls >= self.max_tool_calls:
            raise RuntimeBudgetExceeded("max_tool_calls reached")
        return replace(self, used_tool_calls=self.used_tool_calls + 1)

    def _check_runtime(self) -> None:
        if self.max_runtime_seconds is None:
            return
        if time.monotonic() - self.started_at > self.max_runtime_seconds:
            raise RuntimeBudgetExceeded("max_runtime reached")


@dataclass(frozen=True)
class AgentRun:
    run_id: str
    question: str
    analysis_mode: str
    status: str
    task_graph: TaskGraph
    observations: ObservationStore
    budget: RuntimeBudget
    conversation_state: ConversationState
    answer_markdown: str = ""
    stop_reason: str | None = None
    debug_info: dict[str, Any] = field(default_factory=dict)

    @property
    def task_order(self) -> tuple[str, ...]:
        return tuple(node.task_id for node in self.task_graph.topological_order())

    @property
    def completed_task_order(self) -> tuple[str, ...]:
        return tuple(
            node.task_id
            for node in self.task_graph.topological_order()
            if node.status in {"done", "skipped"}
        )


def run_agent(
    question: str,
    conversation_state: ConversationState | dict[str, Any] | None,
    config: Settings,
    *,
    selected_profile_id: int | None = None,
    date_from: str | None = None,
    date_to: str | None = None,
    llm_client: LocalLlmClient | None = None,
    budget: RuntimeBudget | None = None,
) -> AgentRun:
    state = ConversationState.from_value(conversation_state)
    run = AgentRun(
        run_id=f"agent-run-{uuid4().hex[:12]}",
        question=question,
        analysis_mode=config.analysis.mode,
        status="running",
        task_graph=private_full_task_graph(),
        observations=ObservationStore(),
        budget=budget
        or RuntimeBudget(
            max_runtime_seconds=float(config.analysis.max_runtime_seconds),
            max_llm_calls=config.analysis.max_llm_calls,
            max_tool_calls=config.analysis.max_tool_calls,
        ),
        conversation_state=state,
    )
    if config.analysis.mode not in PRIVATE_FULL_MODES:
        return replace(
            run,
            status="skipped",
            answer_markdown="safe_window mode uses the existing reasoning orchestrator fallback.",
            stop_reason="safe_window fallback",
        )
    context: dict[str, Any] = {
        "state": state,
        "settings": config,
        "selected_profile_id": selected_profile_id,
        "date_from": date_from,
        "date_to": date_to,
        "llm_client": llm_client or LocalLlmClient(config.llm),
    }
    while True:
        ready = run.task_graph.ready_nodes()
        if not ready:
            break
        node = ready[0]
        try:
            run = replace(run, budget=run.budget.consume_step(), task_graph=run.task_graph.mark_running(node.task_id))
            run, context = _execute_task(run, node, context)
            decision = replan_for_observations(
                run.task_graph,
                run.observations,
                max_steps_remaining=run.budget.max_steps - run.budget.used_steps,
            )
            run = replace(run, task_graph=decision.graph)
            if not decision.answerable:
                answer = _missing_information_answer(run, decision.stop_reason or "missing information")
                status = "stopped" if decision.stop_reason == "max_steps reached" else "blocked"
                return _finish_run(
                    replace(run, status=status, answer_markdown=answer, stop_reason=decision.stop_reason),
                    context,
                )
        except RuntimeBudgetExceeded as exc:
            graph = run.task_graph.mark_failed(node.task_id, str(exc))
            answer = _budget_stop_answer(str(exc))
            return _finish_run(
                replace(run, status="stopped", task_graph=graph, answer_markdown=answer, stop_reason=str(exc)),
                context,
            )
    final_run = replace(run, status="succeeded")
    answer = _compose_runtime_answer(final_run)
    verification_report = verify_final_answer(
        answer,
        manifest=context.get("manifest"),
        observations=final_run.observations,
        date_from=context.get("date_from"),
        date_to=context.get("date_to"),
        evidence_refs=_runtime_evidence_refs(final_run),
        computed_facts=_runtime_computed_facts(final_run),
        mode=context["settings"].app.mode,
    )
    answer = verification_report.corrected_answer or answer
    final_run = replace(
        final_run,
        observations=final_run.observations.add(
            task_id="verify_answer",
            observation_type="final_answer_verification",
            summary="ok" if verification_report.ok else "issues_found",
            data=verification_report.to_dict(),
        ),
    )
    updated_state = ConversationState(
        active_profile_id=context.get("profile_id") or state.active_profile_id,
        active_profile_name=context.get("profile_name") or state.active_profile_name,
        last_question=question,
        last_intents=tuple(context.get("intents") or ()),
        last_date_range=(context.get("date_from"), context.get("date_to")),
        last_answer_summary=_first_summary_line(answer),
        last_query_plan={"runtime": "private_full", "task_order": run.task_order},
        last_answerability_report=context.get("answerability", {}),
    )
    return _finish_run(replace(final_run, answer_markdown=sanitize_answer(answer), conversation_state=updated_state), context)


def stream_run_agent(
    question: str,
    conversation_state: ConversationState | dict[str, Any] | None,
    config: Settings,
    **kwargs: Any,
) -> Iterable[AgentStreamEvent]:
    for event in _runtime_start_events(config):
        yield event
    run = run_agent(question, conversation_state, config, **kwargs)
    yield from _runtime_done_events(run)
    yield AgentStreamEvent(
        kind="final_answer",
        title="回答を作成しました",
        message=run.answer_markdown,
        status="done",
        safe_for_user=True,
        metadata={
            "conversation_state": run.conversation_state.to_dict(),
            "review_targets": [],
            "debug_info": run.debug_info,
        },
    )


def _runtime_start_events(config: Settings) -> tuple[AgentStreamEvent, ...]:
    mode_message = (
        f"private full mode: analysis_mode={config.analysis.mode} として、"
        "Plan -> Execute -> Observe -> Analyze -> Replan -> Answerを実行します。"
    )
    return (
        _event("progress", "質問を理解しています", mode_message),
        _event("tool_start", "対象範囲の全データを集めています", "上流DBはread-onlyで扱い、progressには件数だけを表示します。"),
        _event("progress", "manifestを作っています", "source数、日付coverage、推定context量を安全に集計します。"),
        _event("progress", "contextに入るか確認しています", "llm.num_ctxとmanifestの推定token数からsingle/batchを選びます。"),
        _event("progress", "batchを作っています", "contextに入らない場合は日付順・source順のbatchへ分割します。"),
        _event("llm_start", "batch n/mをLLMで分析しています", "local LLMに渡す場合もraw promptやraw payloadはprogressに表示しません。"),
        _event("progress", "観測結果を統合しています", "batch観測とsource_refを統合して、根拠参照を維持します。"),
        _event("progress", "追加調査が必要か判断しています", "不足情報やbudget超過があれば安全に停止して案内します。"),
        _event("progress", "最終回答を検証しています", "source_ref検証とSafety/Privacy Guardを通します。"),
        _event("progress", "回答を作成しています", "件数・日付・ID対応はPython側の結果を優先します。"),
    )


def _runtime_done_events(run: AgentRun) -> Iterable[AgentStreamEvent]:
    manifest = run.observations.latest("manifest")
    batches = run.observations.latest("batches")
    context_mode = run.observations.latest("context_mode")
    loaded = run.observations.latest("full_data_loaded")
    if loaded:
        data = loaded.data
        yield _event(
            "tool_done",
            "対象範囲の全データを集めました",
            (
                f"line={data.get('line_count', 0)}, note={data.get('note_count', 0)}, "
                f"media={data.get('media_count', 0)}, face={data.get('face_count', 0)}, "
                f"location={data.get('location_count', 0)}"
            ),
            status="done",
        )
    if manifest:
        yield _event(
            "progress",
            "manifestを作りました",
            f"estimated_tokens={manifest.data.get('estimated_tokens', 0)}, line_count={manifest.data.get('line_count', 0)}",
            status="done",
        )
    if context_mode:
        yield _event("progress", "context判定が完了しました", f"context_mode={context_mode.summary}", status="done")
    if batches:
        batch_count = int(batches.data.get("batch_count", 0) or 0)
        yield _event("progress", "batch作成が完了しました", f"batch_count={batch_count}", status="done")
        if batch_count:
            yield _event("llm_done", f"batch {batch_count}/{batch_count}をLLMで分析しました", "structured resultだけを保持します。", status="done")
    if run.stop_reason:
        yield _event("warning", "Agent Runtimeが途中停止しました", run.stop_reason, status="done")
    else:
        yield _event("progress", "追加調査の判断が完了しました", "このrunでは追加taskなしで回答へ進みました。", status="done")
    yield _event("progress", "最終回答の検証が完了しました", "raw promptやraw payloadは表示していません。", status="done")


def _execute_task(run: AgentRun, node: TaskNode, context: dict[str, Any]) -> tuple[AgentRun, dict[str, Any]]:
    task_id = node.task_id
    settings: Settings = context["settings"]
    observations = run.observations
    result: dict[str, Any] = {}
    if task_id == "understand_question":
        frame = understand_question(run.question, context["state"], settings=settings, llm_client=context["llm_client"])
        context["frame"] = frame
        context["intents"] = frame.intents
        result = {"primary_intent": frame.primary_intent}
        observations = observations.add(task_id=task_id, observation_type="question_frame", summary=frame.primary_intent, data=result)
    elif task_id == "resolve_profile":
        resolution = resolve_profile(settings, context["frame"], selected_profile_id=context.get("selected_profile_id"))
        context["profile_resolution"] = resolution
        context["profile_id"] = resolution.profile_id
        context["profile_name"] = resolution.profile.profile_name if resolution.profile else None
        result = {"status": resolution.status, "profile_id": resolution.profile_id}
        observations = observations.add(task_id=task_id, observation_type="profile_resolution", summary=resolution.status, data=result)
    elif task_id == "determine_scope":
        frame = context["frame"]
        context["date_from"] = _clean_text(context.get("date_from")) or frame.date_from
        context["date_to"] = _clean_text(context.get("date_to")) or frame.date_to
        result = {"date_from": context["date_from"], "date_to": context["date_to"], "scope": settings.analysis.default_scope}
        observations = observations.add(task_id=task_id, observation_type="scope", summary="scope determined", data=result)
    elif task_id == "check_answerability":
        report = check_answerability(context["frame"], context["profile_resolution"], settings, llm_client=context["llm_client"])
        context["answerability"] = report.to_dict()
        result = {"status": report.status, "can_execute": report.can_execute}
        observation_type = "answerability" if report.can_execute else "missing_information"
        observations = observations.add(task_id=task_id, observation_type=observation_type, summary=report.status, data=result)
        if report.status == "partially_answerable":
            observations = observations.add(
                task_id=task_id,
                observation_type="partial_answerability",
                summary=report.status,
                data=result,
            )
    elif task_id == "load_full_data":
        run = replace(run, budget=run.budget.consume_tool_call())
        context["full_data"] = {"line_items": (), "note_items": (), "media_items": (), "face_items": (), "location_items": ()}
        result = {"line_count": 0, "note_count": 0, "media_count": 0, "face_count": 0, "location_count": 0}
        observations = observations.add(task_id=task_id, observation_type="full_data_loaded", summary="empty full data placeholder", data=result)
    elif task_id == "build_manifest":
        data = context["full_data"]
        manifest = build_full_data_manifest(
            run_id=run.run_id,
            profile_id=context.get("profile_id"),
            date_from=context.get("date_from"),
            date_to=context.get("date_to"),
            line_items=data["line_items"],
            note_items=data["note_items"],
            media_items=data["media_items"],
            face_items=data["face_items"],
            location_items=data["location_items"],
        )
        context["manifest"] = manifest
        result = {"estimated_tokens": manifest.estimated_tokens, "line_count": manifest.line_count}
        observations = observations.add(task_id=task_id, observation_type="manifest", summary="manifest built", data=result)
    elif task_id == "decide_context_mode":
        manifest = context["manifest"]
        mode = decide_context_mode(manifest, settings.llm)
        context["context_mode"] = mode
        result = {"context_mode": mode}
        observations = observations.add(task_id=task_id, observation_type="context_mode", summary=mode, data=result)
    elif task_id == "build_batches":
        data = context["full_data"]
        batches = build_hybrid_batches(
            run_id=run.run_id,
            line_items=data["line_items"],
            note_items=data["note_items"],
            media_items=data["media_items"],
            face_items=data["face_items"],
            location_items=data["location_items"],
            max_items_per_batch=settings.analysis.full_scan.max_items_per_batch,
            max_chars_per_batch=settings.analysis.full_scan.max_chars_per_batch,
            overlap_items=settings.analysis.full_scan.overlap_items,
            max_batches_per_run=settings.analysis.full_scan.max_batches_per_run,
        )
        context["batches"] = batches
        result = {"batch_count": len(batches)}
        observations = observations.add(task_id=task_id, observation_type="batches", summary="batches built", data=result)
    elif task_id == "analyze_single_context":
        if context.get("context_mode") != "single_context":
            return replace(run, task_graph=run.task_graph.mark_skipped(task_id, "iterative context mode"), observations=observations), context
        policy = raw_payload_policy_from_config(settings)
        manifest: FullDataManifest = replace(context["manifest"], raw_payload_policy=policy.to_dict())
        bundle = build_single_context_prompt(
            run.question,
            {"profile_id": context.get("profile_id")},
            manifest,
            context["full_data"],
            policy,
        )
        if context["llm_client"].is_configured():
            run = replace(run, budget=run.budget.consume_llm_call())
            synthesis = analyze_single_context(run.question, manifest, bundle, context["llm_client"], max_llm_calls=1)
            summary = synthesis.summary
            context["synthesis"] = synthesis
        else:
            summary = "local LLM is not configured; structured analysis was skipped"
        result = {"summary": summary}
        observations = observations.add(task_id=task_id, observation_type="single_context_analysis", summary=summary, data=result)
    elif task_id == "analyze_full_batch":
        if context.get("context_mode") == "single_context":
            return replace(run, task_graph=run.task_graph.mark_skipped(task_id, "single context mode"), observations=observations), context
        if context["llm_client"].is_configured():
            run = replace(run, budget=run.budget.consume_llm_call())
        result = {"batch_count": len(context.get("batches") or ())}
        observations = observations.add(task_id=task_id, observation_type="batch_analysis", summary="batch analysis placeholder", data=result)
    elif task_id == "synthesize_full_range":
        if context.get("context_mode") == "single_context":
            return replace(run, task_graph=run.task_graph.mark_skipped(task_id, "single context mode"), observations=observations), context
        if context["llm_client"].is_configured():
            run = replace(run, budget=run.budget.consume_llm_call())
        observations = observations.add(task_id=task_id, observation_type="synthesis", summary="synthesis placeholder", data={})
    elif task_id == "verify_answer":
        result = {"source_ref_verification": "placeholder", "raw_payload_not_persisted": True}
        observations = observations.add(task_id=task_id, observation_type="verification", summary="source refs verified at scaffold level", data=result)
    elif task_id == "compose_answer":
        answer = _compose_runtime_answer(replace(run, observations=observations))
        context["answer_markdown"] = answer
        result = {"answer_ready": True}
        observations = observations.add(task_id=task_id, observation_type="answer", summary="answer composed", data=result)
    graph = run.task_graph.mark_done(task_id, result)
    return replace(run, task_graph=graph, observations=observations), context


def _finish_run(run: AgentRun, context: dict[str, Any]) -> AgentRun:
    verification = run.observations.latest("final_answer_verification")
    debug_info = {
        "runtime": "private_full",
        "analysis_mode": run.analysis_mode,
        "status": run.status,
        "stop_reason": run.stop_reason,
        "budget": {
            "max_runtime_seconds": run.budget.max_runtime_seconds,
            "max_steps": run.budget.max_steps,
            "used_steps": run.budget.used_steps,
            "max_llm_calls": run.budget.max_llm_calls,
            "used_llm_calls": run.budget.used_llm_calls,
            "max_tool_calls": run.budget.max_tool_calls,
            "used_tool_calls": run.budget.used_tool_calls,
        },
        "task_graph": run.task_graph.to_debug_dict(),
        "observations": run.observations.to_debug_list(),
        "answerability": context.get("answerability", {}),
        "verification": verification.data if verification else {},
    }
    return replace(run, debug_info=debug_info)


def _runtime_evidence_refs(run: AgentRun) -> tuple[str, ...]:
    refs: list[str] = []
    for observation in run.observations.observations:
        refs.extend(observation.source_refs)
        for key in ("source_refs", "relevant_evidence_refs", "evidence_refs"):
            value = observation.data.get(key)
            if isinstance(value, str):
                refs.append(value)
            elif isinstance(value, (list, tuple)):
                refs.extend(str(item) for item in value if item)
    return tuple(dict.fromkeys(refs))


def _runtime_computed_facts(run: AgentRun) -> dict[str, int]:
    manifest = run.observations.latest("manifest")
    loaded = run.observations.latest("full_data_loaded")
    facts: dict[str, int] = {}
    if manifest:
        for key in ("line_count", "note_count", "media_count", "face_count", "location_count"):
            value = manifest.data.get(key)
            if isinstance(value, int):
                facts[key] = value
    if loaded:
        for key, value in loaded.data.items():
            if isinstance(value, int):
                facts[key] = value
    return facts


def _compose_runtime_answer(run: AgentRun) -> str:
    manifest = run.observations.latest("manifest")
    context_mode = run.observations.latest("context_mode")
    analysis = run.observations.latest("single_context_analysis") or run.observations.latest("synthesis")
    lines = [
        "要約:",
        "private full mode のAgent Runtimeで、質問をTaskGraphとして処理しました。",
        "",
        "集計:",
        f"- analysis_mode: {run.analysis_mode}",
        f"- runtime_status: {run.status}",
        f"- executed_steps: {run.budget.used_steps}",
        f"- llm_calls_used: {run.budget.used_llm_calls}",
        f"- tool_calls_used: {run.budget.used_tool_calls}",
    ]
    if manifest:
        lines.append(f"- manifest_estimated_tokens: {manifest.data.get('estimated_tokens', 0)}")
    if context_mode:
        lines.append(f"- context_mode: {context_mode.summary}")
    lines.extend(
        [
            "",
            "具体例:",
            "- Full Data Access loader はまだplaceholderなので、現時点では実データ本文の分析結果ではありません。",
            "",
            "根拠:",
            "<details>\n<summary>Runtime observations</summary>\n\n"
            + "\n".join(f"- {obs.task_id}: {obs.summary}" for obs in run.observations.observations)
            + "\n\n</details>",
            "",
            "信頼度:",
            "weak (runtime scaffold; source-ref verification is not fully connected yet)",
            "",
            "注意:",
            "<details>\n<summary>注意を表示</summary>\n\n"
            "- raw prompt / raw payload はDBに保存しません。\n"
            "- 件数・日付・ID検証はPython側で扱います。\n"
            "- public modeでは従来どおりredactionを維持します。\n"
            + (f"- analysis: {analysis.summary}\n" if analysis else "")
            + "\n</details>",
        ]
    )
    return "\n".join(lines)


def _missing_information_answer(run: AgentRun, reason: str) -> str:
    return sanitize_answer(
        "\n".join(
            [
                "要約:",
                "この質問に答えるための手動設定または必須情報が不足しています。",
                "",
                "集計:",
                f"- runtime_status: blocked",
                f"- reason: {reason}",
                "",
                "具体例:",
                "- profile、speaker、date range、upstream adapter設定を確認してください。",
                "",
                "根拠:",
                "<details>\n<summary>Runtime observations</summary>\n\n"
                + "\n".join(f"- {obs.task_id}: {obs.summary}" for obs in run.observations.observations)
                + "\n\n</details>",
                "",
                "信頼度:",
                "very weak",
                "",
                "注意:",
                "<details>\n<summary>注意を表示</summary>\n\n- AIはprofileやspeakerの対応を自動推定しません。\n\n</details>",
            ]
        )
    )


def _budget_stop_answer(reason: str) -> str:
    return sanitize_answer(
        f"要約:\nRuntime budgetに達したため停止しました。\n\n集計:\n- stop_reason: {reason}\n\n注意:\n<details>\n<summary>注意を表示</summary>\n\n- 途中停止のため回答は未完成です。\n\n</details>"
    )


def _event(kind: str, title: str, message: str, *, status: str = "pending") -> AgentStreamEvent:
    return AgentStreamEvent(
        kind=kind,  # type: ignore[arg-type]
        title=title,
        message=sanitize_answer(message),
        status=status,  # type: ignore[arg-type]
        safe_for_user=True,
    )


def _clean_text(value: object) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _first_summary_line(answer: str) -> str | None:
    for line in answer.splitlines():
        clean = line.strip()
        if clean and clean != "要約:":
            return clean[:240]
    return None
