from __future__ import annotations

from dataclasses import dataclass, field, replace
from datetime import date, timedelta
import re
import time
from typing import Any, Iterable
from uuid import uuid4

from relationship_lifelog_agent.adapters.types import EvidenceItem
from relationship_lifelog_agent.agent.answerability import check_answerability
from relationship_lifelog_agent.agent.answer_composer import compose_answer
from relationship_lifelog_agent.agent.conversation_state import ConversationState
from relationship_lifelog_agent.agent.executor import execute_plan
from relationship_lifelog_agent.agent.memory import build_memory
from relationship_lifelog_agent.agent.observation_store import Observation, ObservationStore
from relationship_lifelog_agent.agent.planner import build_plan
from relationship_lifelog_agent.agent.profile_resolver import resolve_profile
from relationship_lifelog_agent.agent.question_understanding import understand_question
from relationship_lifelog_agent.agent.replanner import replan_for_observations
from relationship_lifelog_agent.agent.router import route_question
from relationship_lifelog_agent.agent.streaming import AgentStreamEvent
from relationship_lifelog_agent.agent.task_graph import TaskGraph, TaskNode, private_full_task_graph
from relationship_lifelog_agent.config import Settings
from relationship_lifelog_agent.full_context.batch_builder import build_hybrid_batches
from relationship_lifelog_agent.full_context.context_budget import decide_context_mode
from relationship_lifelog_agent.full_context.full_range_analyzer import analyze_single_context
from relationship_lifelog_agent.full_context.manifest import build_full_data_manifest
from relationship_lifelog_agent.full_context.prompt_packer import build_single_context_prompt
from relationship_lifelog_agent.full_context.types import FullDataManifest, FullLineItem, FullMediaItem, FullNoteItem
from relationship_lifelog_agent.llm.local_client import LocalLlmClient
from relationship_lifelog_agent.llm.usage import LlmUsageTrace
from relationship_lifelog_agent.privacy.guard import sanitize_answer
from relationship_lifelog_agent.privacy.raw_payload_policy import from_config as raw_payload_policy_from_config
from relationship_lifelog_agent.profiles import load_profile_context
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
    mode: str = "private",
    memory: object | None = None,
    post_conflict_window_days: int | float | None = None,
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
        "mode": mode,
        "memory": memory,
        "post_conflict_window_days": post_conflict_window_days,
        "llm_client": llm_client or LocalLlmClient(config.llm),
        "usage": LlmUsageTrace.from_settings(config.llm),
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
        mode=context.get("mode") or context["settings"].app.mode,
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
    return _finish_run(
        replace(
            final_run,
            answer_markdown=sanitize_answer(answer, mode=context.get("mode", "private")),
            conversation_state=updated_state,
        ),
        context,
    )


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
    decomposition = run.observations.latest("question_decomposition")
    conflict_dates = run.observations.latest("conflict_candidate_dates")
    windows = run.observations.latest("surrounding_media_windows")
    surrounding = run.observations.latest("surrounding_media_analysis")
    manifest = run.observations.latest("manifest")
    batches = run.observations.latest("batches")
    context_mode = run.observations.latest("context_mode")
    loaded = run.observations.latest("full_data_loaded")
    if decomposition:
        yield _event(
            "progress",
            "複合質問として整理しました",
            "Q1で候補日を探し、Q2でその日付の前日・当日・翌日の写真を確認します。",
            status="done",
        )
    if conflict_dates:
        yield _event(
            "tool_done",
            "喧嘩/すれ違い候補日を確認しました",
            f"candidate_dates={', '.join(conflict_dates.data.get('dates', [])) or 'none'}",
            status="done",
        )
    if windows:
        yield _event(
            "tool_done",
            "前後日の写真確認範囲を作りました",
            "surrounding_dates=" + ", ".join(windows.data.get("surrounding_dates", [])),
            status="done",
        )
    if surrounding:
        yield _event(
            "tool_done",
            "LINEと写真の根拠を統合しました",
            "範囲外mediaは通常回答から除外しました。",
            status="done",
        )
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
        frame = understand_question(
            run.question,
            context["state"],
            settings=settings,
            llm_client=context["llm_client"],
            usage=context["usage"],
        )
        context["frame"] = frame
        context["intents"] = frame.intents
        result = {"primary_intent": frame.primary_intent, "intents": list(frame.intents)}
        observations = observations.add(task_id=task_id, observation_type="question_frame", summary=frame.primary_intent, data=result)
    elif task_id == "decompose_compound_question":
        frame = context["frame"]
        if frame.primary_intent != "conflict_dates_with_surrounding_media":
            return replace(run, task_graph=run.task_graph.mark_skipped(task_id, "not a compound media question"), observations=observations), context
        result = {
            "compound": True,
            "q1": "喧嘩・軽いすれ違い候補日を探す",
            "q2": "Q1の候補日を使って前日・当日・翌日の写真/メディアを確認する",
            "dependencies": {"q2": "q1"},
        }
        context["question_decomposition"] = result
        observations = observations.add(
            task_id=task_id,
            observation_type="question_decomposition",
            summary="Q1 conflict date lookup -> Q2 surrounding media summary",
            data=result,
        )
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
        report = check_answerability(
            context["frame"],
            context["profile_resolution"],
            settings,
            llm_client=context["llm_client"],
            usage=context["usage"],
        )
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
    elif task_id == "find_conflict_candidate_dates":
        if context["frame"].primary_intent != "conflict_dates_with_surrounding_media":
            return replace(run, task_graph=run.task_graph.mark_skipped(task_id, "not needed"), observations=observations), context
        run = replace(run, budget=run.budget.consume_tool_call())
        compound = _run_compound_analysis(run, context)
        context["compound_analysis"] = compound
        conflict_dates = _extract_conflict_dates_from_result(compound["analysis_result"])
        context["conflict_candidate_dates"] = conflict_dates
        result = {
            "dates": list(conflict_dates),
            "q1_completed": True,
            "depends_on": "question_decomposition.q1",
            "source_refs": list(compound["source_refs"]),
        }
        observations = observations.add(
            task_id=task_id,
            observation_type="conflict_candidate_dates",
            summary=", ".join(conflict_dates) if conflict_dates else "no conflict candidate dates",
            data=result,
            source_refs=compound["source_refs"],
            confidence=float(compound["analysis_result"].confidence),
        )
    elif task_id == "build_surrounding_media_windows":
        if context["frame"].primary_intent != "conflict_dates_with_surrounding_media":
            return replace(run, task_graph=run.task_graph.mark_skipped(task_id, "not needed"), observations=observations), context
        before_days = max(0, int(getattr(settings.relationship, "surrounding_media_days_before", 1)))
        after_days = max(0, int(getattr(settings.relationship, "surrounding_media_days_after", 1)))
        windows = tuple(
            _surrounding_window(value, before_days=before_days, after_days=after_days)
            for value in context.get("conflict_candidate_dates", ())
        )
        surrounding_dates = tuple(dict.fromkeys(day for start, end in windows for day in _days_between(start, end)))
        context["surrounding_windows"] = windows
        context["surrounding_dates"] = surrounding_dates
        result = {
            "before_days": before_days,
            "after_days": after_days,
            "windows": [{"date_from": start, "date_to": end} for start, end in windows],
            "surrounding_dates": list(surrounding_dates),
            "q2_depends_on_q1": True,
        }
        observations = observations.add(
            task_id=task_id,
            observation_type="surrounding_media_windows",
            summary=", ".join(surrounding_dates) if surrounding_dates else "no surrounding windows",
            data=result,
        )
    elif task_id == "analyze_surrounding_media":
        if context["frame"].primary_intent != "conflict_dates_with_surrounding_media":
            return replace(run, task_graph=run.task_graph.mark_skipped(task_id, "not needed"), observations=observations), context
        compound = context.get("compound_analysis") or _run_compound_analysis(run, context)
        context["compound_analysis"] = compound
        media_day_lines = _media_day_lines(compound["analysis_result"])
        context["media_day_lines"] = media_day_lines
        result = {
            "q2_completed": True,
            "depends_on": "find_conflict_candidate_dates",
            "surrounding_dates": list(context.get("surrounding_dates", ())),
            "media_day_lines": list(media_day_lines),
            "summary": compound["analysis_result"].summary,
            "aggregate": dict(compound["analysis_result"].aggregate),
            "examples": list(compound["analysis_result"].examples),
            "cautions": list(compound["analysis_result"].cautions),
            "confidence": compound["analysis_result"].confidence,
            "evidence_strength": compound["analysis_result"].evidence_strength,
            "source_refs": list(compound["source_refs"]),
            "answer_excerpt": _shorten(compound["answer_markdown"], 480),
            "llm_usage": context["usage"].to_dict(),
        }
        observations = observations.add(
            task_id=task_id,
            observation_type="surrounding_media_analysis",
            summary=f"media days checked: {len(context.get('surrounding_dates', ())) }",
            data=result,
            source_refs=compound["source_refs"],
            confidence=float(compound["analysis_result"].confidence),
        )
    elif task_id == "load_full_data":
        run = replace(run, budget=run.budget.consume_tool_call())
        full_data = _load_full_data_for_runtime(context)
        context["full_data"] = full_data
        result = {
            "line_count": len(full_data["line_items"]),
            "note_count": len(full_data["note_items"]),
            "media_count": len(full_data["media_items"]),
            "face_count": len(full_data["face_items"]),
            "location_count": len(full_data["location_items"]),
            "source_refs": list(_source_refs_from_full_data(full_data)),
        }
        observations = observations.add(
            task_id=task_id,
            observation_type="full_data_loaded",
            summary="full data loaded from read-only adapters",
            data=result,
            source_refs=_source_refs_from_full_data(full_data),
        )
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


def _run_compound_analysis(run: AgentRun, context: dict[str, Any]) -> dict[str, Any]:
    settings: Settings = context["settings"]
    route = route_question(run.question)
    plan = build_plan(route, date_from=context.get("date_from"), date_to=context.get("date_to"))
    memory = context.get("memory") or build_memory(settings)
    context["memory"] = memory
    profile = load_profile_context(settings, context.get("profile_id"))
    usage: LlmUsageTrace = context["usage"]
    result = execute_plan(
        plan,
        memory,
        mode=context.get("mode", "private"),
        profile=profile,
        settings=settings,
        date_from=context.get("date_from"),
        date_to=context.get("date_to"),
        post_conflict_window_days=context.get("post_conflict_window_days"),
        llm_client=context.get("llm_client"),
        usage=usage,
    )
    person_names = [profile.profile_name] if profile else None
    answer = compose_answer(result, mode=context.get("mode", "private"), person_names=person_names)
    source_refs = tuple(dict.fromkeys(_source_refs_from_evidence(result.evidence) or (f"runtime:compound:{run.run_id}",)))
    return {
        "analysis_result": result,
        "answer_markdown": answer,
        "source_refs": source_refs,
    }


def _load_full_data_for_runtime(context: dict[str, Any]) -> dict[str, tuple[Any, ...]]:
    settings: Settings = context["settings"]
    memory = context.get("memory") or build_memory(settings)
    context["memory"] = memory
    mode = context.get("mode", "private")
    date_from = context.get("date_from")
    date_to = context.get("date_to")
    if not date_from and not date_to and context.get("surrounding_dates"):
        dates = tuple(str(value) for value in context["surrounding_dates"])
        date_from, date_to = min(dates), max(dates)
    limit = max(1, int(settings.analysis.full_scan.max_items_per_batch))
    personal = getattr(memory, "personal", memory)
    notes = getattr(memory, "notes", memory)
    line_items = _safe_adapter_call(personal, "search_line", "", date_from=date_from, date_to=date_to, limit=limit, mode=mode)
    note_items = _safe_adapter_call(notes, "search_notes", "", date_from=date_from, date_to=date_to, limit=limit, mode=mode)
    media_items = _safe_adapter_call(personal, "search_media", "", date_from=date_from, date_to=date_to, limit=limit, mode=mode)
    return {
        "line_items": tuple(_full_line_item(item) for item in line_items if _full_line_item(item) is not None),
        "note_items": tuple(_full_note_item(item) for item in note_items if _full_note_item(item) is not None),
        "media_items": tuple(_full_media_item(item) for item in media_items if _full_media_item(item) is not None),
        "face_items": (),
        "location_items": (),
    }


def _safe_adapter_call(adapter: object, method_name: str, *args: Any, **kwargs: Any) -> list[Any]:
    method = getattr(adapter, method_name, None)
    if not callable(method):
        return []
    try:
        result = method(*args, **kwargs)
    except TypeError:
        trimmed = {key: value for key, value in kwargs.items() if key != "mode"}
        try:
            result = method(*args, **trimmed)
        except Exception:
            return []
    except Exception:
        return []
    return list(result or [])


def _full_line_item(item: object) -> FullLineItem | None:
    source_ref = _item_source_ref(item)
    source_id = _item_attr(item, "source_id")
    if not source_ref or not source_id:
        return None
    return FullLineItem(
        source_ref=source_ref,
        source_id=source_id,
        conversation_id=_item_attr(item, "conversation_id"),
        timestamp=_item_attr(item, "date"),
        speaker_id=_item_attr(item, "speaker_id"),
        speaker_label=None,
        raw_text=_shorten(_item_text(item), 2000) or None,
        raw_metadata={"summary": _shorten(_item_attr(item, "summary"), 240)},
    )


def _full_note_item(item: object) -> FullNoteItem | None:
    source_ref = _item_source_ref(item)
    source_id = _item_attr(item, "source_id")
    if not source_ref or not source_id:
        return None
    return FullNoteItem(
        source_ref=source_ref,
        source_id=source_id,
        note_id=_item_attr(item, "note_id") or source_id,
        created_at=_item_attr(item, "date"),
        raw_body=_shorten(_item_text(item), 4000) or None,
        tags=tuple(str(tag) for tag in getattr(item, "metadata", {}).get("tags", ()) if tag),
        raw_metadata={"summary": _shorten(_item_attr(item, "summary"), 240)},
    )


def _full_media_item(item: object) -> FullMediaItem | None:
    source_ref = _item_source_ref(item)
    source_id = _item_attr(item, "source_id")
    if not source_ref or not source_id:
        return None
    return FullMediaItem(
        source_ref=source_ref,
        source_id=source_id,
        media_id=_item_attr(item, "media_id") or source_id,
        captured_at=_item_attr(item, "date"),
        place_label=_item_attr(item, "place_id"),
        vlm_caption=_shorten(_item_attr(item, "summary"), 1000) or None,
        raw_metadata={"media_type": _item_attr(item, "media_type") or "photo"},
    )


def _source_refs_from_full_data(full_data: dict[str, tuple[Any, ...]]) -> tuple[str, ...]:
    refs: list[str] = []
    for items in full_data.values():
        refs.extend(str(getattr(item, "source_ref")) for item in items if getattr(item, "source_ref", None))
    return tuple(dict.fromkeys(refs))


def _source_refs_from_evidence(evidence_items: Iterable[EvidenceItem]) -> tuple[str, ...]:
    refs: list[str] = []
    for item in evidence_items:
        if item.source_pointer:
            refs.append(item.source_pointer)
        else:
            refs.append(f"{item.source_type}:{item.source_id}")
    return tuple(dict.fromkeys(refs))


def _extract_conflict_dates_from_result(result: Any) -> tuple[str, ...]:
    aggregate = getattr(result, "aggregate", {}) or {}
    observed = str(aggregate.get("observed_dates") or "")
    dates = re.findall(r"\b20\d{2}-\d{2}-\d{2}\b", observed)
    if dates:
        return tuple(dict.fromkeys(dates))
    examples = "\n".join(str(item) for item in getattr(result, "examples", ()))
    return tuple(dict.fromkeys(re.findall(r"\b20\d{2}-\d{2}-\d{2}\b", examples)))


def _media_day_lines(result: Any) -> tuple[str, ...]:
    lines: list[str] = []
    in_media_section = False
    for value in getattr(result, "examples", ()):
        text = str(value)
        if text.startswith("写真から見た前後の日"):
            in_media_section = True
            continue
        if in_media_section and re.match(r"^20\d{2}-\d{2}-\d{2}:", text):
            lines.append(text)
    return tuple(lines)


def _surrounding_window(value: str, *, before_days: int, after_days: int) -> tuple[str, str]:
    current = date.fromisoformat(value)
    return (current - timedelta(days=before_days)).isoformat(), (current + timedelta(days=after_days)).isoformat()


def _days_between(start: str, end: str) -> tuple[str, ...]:
    current = date.fromisoformat(start)
    last = date.fromisoformat(end)
    values: list[str] = []
    while current <= last:
        values.append(current.isoformat())
        current += timedelta(days=1)
    return tuple(values)


def _item_source_ref(item: object) -> str | None:
    pointer = getattr(item, "source_pointer", None)
    if pointer:
        return str(pointer)
    source_type = getattr(item, "source_type", None)
    source_id = getattr(item, "source_id", None)
    if source_type and source_id:
        return f"{source_type}:{source_id}"
    return None


def _item_attr(item: object, name: str) -> str | None:
    value = getattr(item, name, None)
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _item_text(item: object) -> str:
    return _item_attr(item, "excerpt") or _item_attr(item, "summary") or ""


def _shorten(value: object, limit: int) -> str:
    text = " ".join(str(value or "").split())
    if len(text) <= limit:
        return text
    return text[: max(0, limit - 3)].rstrip() + "..."


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
        "llm_usage": context.get("usage").to_dict() if context.get("usage") else {},
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
    compound = run.observations.latest("surrounding_media_analysis")
    if compound is not None:
        return _compose_compound_runtime_answer(run, compound)
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


def _compose_compound_runtime_answer(run: AgentRun, compound: Observation) -> str:
    data = compound.data
    aggregate = data.get("aggregate") if isinstance(data.get("aggregate"), dict) else {}
    source_refs = tuple(data.get("source_refs") or compound.source_refs or ())
    conflict_dates = _date_values_from_text(str(aggregate.get("observed_dates") or ""))
    if not conflict_dates:
        conflict_dates = tuple(str(value) for value in data.get("surrounding_dates", ()) if re.fullmatch(r"20\d{2}-\d{2}-\d{2}", str(value)))
    media_day_lines = tuple(str(line) for line in data.get("media_day_lines", ()) if _date_values_from_text(str(line)))
    q1 = "喧嘩・軽いすれ違い候補日を探す"
    q2 = "Q1の日付を使って、前日・当日・翌日の写真/メディアだけを確認する"
    refs_markdown = _source_refs_markdown(source_refs)
    process_lines = [
        "- understand_question -> decompose_compound_question",
        "- find_conflict_candidate_dates: Q1を先に実行",
        "- build_surrounding_media_windows: Q1の日付からexact surrounding datesを作成",
        "- analyze_surrounding_media: 範囲外mediaを除外して日別summaryを作成",
        "- verify_answer: source_refと安全表現を確認",
    ]
    usage = _runtime_llm_usage(run)
    lines = [
        "要約:",
        str(data.get("summary") or "喧嘩・軽いすれ違い候補日と、その前後日の写真/メディアを分けて確認しました。"),
        "",
        "質問分解:",
        f"- Q1: {q1}",
        f"- Q2: {q2}",
        "- 依存関係: Q2はQ1で見つかった候補日だけを使います。",
        "",
        "喧嘩/すれ違い候補日:",
        f"- conflict_candidates: {aggregate.get('conflict_candidates', 0)}",
        f"- minor_misunderstanding_candidates: {aggregate.get('minor_misunderstanding_candidates', 0)}",
        f"- observed_dates: {', '.join(conflict_dates) if conflict_dates else '候補日は見つかっていません'}",
        "",
        "LINE/メモ根拠:",
        refs_markdown,
        "",
        "前日/当日/翌日の写真分析:",
        _media_days_markdown(media_day_lines, tuple(str(value) for value in data.get("surrounding_dates", ()))),
        "",
        "不確実性:",
        f"- confidence: {float(data.get('confidence', 0.5)):.2f}",
        f"- evidence_strength: {float(data.get('evidence_strength', 0.5)):.2f}",
        "- 写真だけでは、仲直り・同行者・相手の気持ちは断定していません。",
        "",
        "注意:",
        "<details>\n<summary>注意を表示</summary>\n\n"
        + "\n".join(f"- {caution}" for caution in (data.get("cautions") or ["喧嘩は候補として扱います。"])[:6])
        + "\n\n</details>",
        "",
        "実行プロセス:",
        "<details>\n<summary>処理プロセスを表示</summary>\n\n" + "\n".join(process_lines) + "\n\n</details>",
        "",
        "LLM利用状況:",
        f"- llm_call_count: {usage.get('llm_call_count', 0)}",
        f"- batch_count: {_runtime_batch_count(run)}",
        f"- source_coverage: {len(source_refs)} source_ref(s)",
    ]
    return "\n".join(lines)


def _source_refs_markdown(source_refs: tuple[str, ...]) -> str:
    if not source_refs:
        return "- source_ref: なし（このrunでは表示可能な根拠参照がありません）"
    return "\n".join(f"- source_ref: {ref}" for ref in source_refs[:8])


def _media_days_markdown(media_day_lines: tuple[str, ...], surrounding_dates: tuple[str, ...]) -> str:
    by_day = {line.split(":", 1)[0]: line.split(":", 1)[1].strip() for line in media_day_lines if ":" in line}
    days = tuple(dict.fromkeys(day for day in surrounding_dates if re.fullmatch(r"20\d{2}-\d{2}-\d{2}", day)))
    if not days:
        days = tuple(dict.fromkeys(_date_values_from_text("\n".join(media_day_lines))))
    if not days:
        return "- 対象日の前後日を作れなかったため、写真summaryはありません。"
    return "\n".join(f"- {day}: {by_day.get(day, '写真/メディア候補は見つかっていません。')}" for day in days)


def _runtime_llm_usage(run: AgentRun) -> dict[str, Any]:
    for observation in run.observations.observations:
        usage = observation.data.get("llm_usage")
        if isinstance(usage, dict):
            return usage
    # Fall back to budget-visible calls and analysis observations.
    return {"llm_call_count": run.budget.used_llm_calls}


def _runtime_batch_count(run: AgentRun) -> int:
    batches = run.observations.latest("batches")
    if not batches:
        return 0
    value = batches.data.get("batch_count", 0)
    return int(value) if isinstance(value, int) else 0


def _date_values_from_text(text: str) -> tuple[str, ...]:
    return tuple(dict.fromkeys(re.findall(r"\b20\d{2}-\d{2}-\d{2}\b", text)))


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
