from __future__ import annotations

import json

from relationship_lifelog_agent.agent.agent_runtime import (
    AgentRun,
    RuntimeBudget,
    run_agent,
    stream_run_agent,
)
from relationship_lifelog_agent.agent.observation_store import ObservationStore
from relationship_lifelog_agent.agent.replanner import Replanner
from relationship_lifelog_agent.agent.reasoning_orchestrator import answer_with_reasoning, stream_answer
from relationship_lifelog_agent.agent.task_graph import TaskGraph, TaskNode
from relationship_lifelog_agent.config import AnalysisSettings, LlmSettings, PathSettings, RelationshipSettings, Settings
from relationship_lifelog_agent.db.repository import RelationshipRepository
from relationship_lifelog_agent.llm.local_client import LocalLlmClient


def test_agent_run_can_be_created(tmp_path) -> None:
    settings = _private_full_settings(tmp_path)

    run = run_agent("いおりとの記録を全体から見て", None, settings)

    assert isinstance(run, AgentRun)
    assert run.analysis_mode == "private_full_range"
    assert run.status == "succeeded"
    assert "要約:" in run.answer_markdown


def test_task_graph_dependency_order_is_respected() -> None:
    graph = TaskGraph()
    graph = graph.add_node(TaskNode("a", "A"))
    graph = graph.add_node(TaskNode("b", "B", ("a",)))
    graph = graph.add_node(TaskNode("c", "C", ("b",)))

    assert [node.task_id for node in graph.topological_order()] == ["a", "b", "c"]
    assert [node.task_id for node in graph.ready_nodes()] == ["a"]
    graph = graph.mark_done("a")
    assert [node.task_id for node in graph.ready_nodes()] == ["b"]


def test_runtime_stops_at_max_steps(tmp_path) -> None:
    settings = _private_full_settings(tmp_path)

    run = run_agent("いおりとの記録を全体から見て", None, settings, budget=RuntimeBudget(max_steps=2))

    assert run.status == "stopped"
    assert run.stop_reason == "max_steps reached"
    assert run.budget.used_steps == 2


def test_runtime_stops_at_max_llm_calls(tmp_path) -> None:
    settings = _private_full_settings(
        tmp_path,
        llm=LlmSettings(provider="ollama", model="fake", enabled=True),
    )
    client = _mock_full_synthesis_client(settings)

    run = run_agent(
        "いおりとの記録を全体から見て",
        None,
        settings,
        llm_client=client,
        budget=RuntimeBudget(max_llm_calls=0),
    )

    assert run.status == "stopped"
    assert run.stop_reason == "max_llm_calls reached"


def test_private_full_range_runs_full_context_order(tmp_path) -> None:
    settings = _private_full_settings(tmp_path)

    run = run_agent("いおりとの記録を全体から見て", None, settings)
    order = list(run.completed_task_order)

    assert order.index("load_full_data") < order.index("build_manifest")
    assert order.index("build_manifest") < order.index("decide_context_mode")
    assert order.index("decide_context_mode") < order.index("analyze_single_context")
    assert order.index("verify_answer") < order.index("compose_answer")


def test_replanner_can_add_task_before_answerable_path() -> None:
    graph = TaskGraph().add_node(TaskNode("check_answerability", "check_answerability"))
    observations = ObservationStore().add(
        task_id="check_answerability",
        observation_type="partial_answerability",
        summary="partially_answerable",
    )

    decision = Replanner().replan(graph, observations, max_steps_remaining=5)

    assert decision.answerable is True
    assert decision.added_tasks == ("collect_missing_information",)
    assert "collect_missing_information" in decision.graph.nodes


def test_missing_profile_stops_with_guidance(tmp_path) -> None:
    settings = Settings(
        analysis=AnalysisSettings(mode="private_full_range"),
        paths=PathSettings(relationship_db=str(tmp_path / "relationship.sqlite")),
        relationship=RelationshipSettings(default_profile_id=999),
    )

    run = run_agent("いおりとの記録を全体から見て", None, settings)

    assert run.status == "blocked"
    assert run.stop_reason == "missing_profile"
    assert "手動設定" in run.answer_markdown
    assert "AIはprofileやspeakerの対応を自動推定しません" in run.answer_markdown


def test_stream_run_agent_yields_final_answer(tmp_path) -> None:
    settings = _private_full_settings(tmp_path)

    events = list(stream_run_agent("いおりとの記録を全体から見て", None, settings))

    assert len(events) > 2
    assert events[0].kind == "progress"
    assert events[-1].kind == "final_answer"
    assert "要約:" in events[-1].message


def test_private_full_mode_orchestrator_uses_agent_runtime(tmp_path) -> None:
    settings = _private_full_settings(tmp_path)

    response = answer_with_reasoning("いおりとの記録を全体から見て", settings=settings)

    assert response.reasoning_summary == "private_full_runtime: succeeded"
    assert "Agent Runtime" in response.answer_markdown
    assert "load_full_data" in response.query_plan_summary


def test_private_full_mode_stream_answer_uses_agent_runtime(tmp_path) -> None:
    settings = _private_full_settings(tmp_path)

    events = list(stream_answer("いおりとの記録を全体から見て", settings=settings))

    assert events[0].message.startswith("private full mode")
    assert events[-1].kind == "final_answer"
    assert "Agent Runtime" in events[-1].message


def test_safe_window_runtime_is_skipped(tmp_path) -> None:
    settings = _private_full_settings(tmp_path, analysis_mode="safe_window")

    run = run_agent("いおりとの記録を全体から見て", None, settings)

    assert run.status == "skipped"
    assert "existing reasoning orchestrator fallback" in run.answer_markdown


def _private_full_settings(
    tmp_path,
    *,
    analysis_mode: str = "private_full_range",
    llm: LlmSettings | None = None,
) -> Settings:
    db_path = tmp_path / "relationship.sqlite"
    repo = RelationshipRepository(db_path)
    profile_id = repo.create_profile(
        "いおり",
        person_source_id="plr:person:target",
        line_speaker_source_id="plr:line_speaker:target",
        self_line_speaker_source_id="plr:line_speaker:self",
        relationship_label="partner",
    )
    return Settings(
        analysis=AnalysisSettings(mode=analysis_mode),
        paths=PathSettings(relationship_db=str(db_path)),
        relationship=RelationshipSettings(default_profile_id=profile_id),
        llm=llm or LlmSettings(enabled=False),
    )


def _mock_full_synthesis_client(settings: Settings) -> LocalLlmClient:
    def _post(_url: str, _payload: dict, _timeout: float) -> dict:
        content = {
            "summary": "structured full-context analysis completed",
            "answer": "要約: structured answer",
            "timeline": [],
            "evidence": [],
            "uncertainties": [],
            "followup_suggestions": [],
            "source_refs": [],
            "confidence": 0.5,
        }
        return {"message": {"content": json.dumps(content)}}

    return LocalLlmClient(settings.llm, http_post=_post)
