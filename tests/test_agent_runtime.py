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
from relationship_lifelog_agent.adapters.types import EventEvidence, LineEvidence, MediaEvidence
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


def test_private_full_runtime_decomposes_compound_question_and_orders_q1_before_q2(tmp_path) -> None:
    settings = _private_full_settings(tmp_path)

    run = run_agent(
        "いおりと喧嘩した日はいつ？その日の前後の日は写真から何をしていた？",
        None,
        settings,
        memory=_CompoundRuntimeMemory(),
    )

    order = list(run.completed_task_order)
    assert "decompose_compound_question" in order
    assert order.index("find_conflict_candidate_dates") < order.index("build_surrounding_media_windows")
    assert order.index("build_surrounding_media_windows") < order.index("analyze_surrounding_media")
    decomposition = run.observations.latest("question_decomposition")
    assert decomposition is not None
    assert decomposition.data["dependencies"] == {"q2": "q1"}


def test_private_full_runtime_builds_exact_surrounding_dates_and_excludes_out_of_range_media(tmp_path) -> None:
    settings = _private_full_settings(tmp_path)

    run = run_agent(
        "いおりと喧嘩した日はいつ？その日の前後の日は写真から何をしていた？",
        None,
        settings,
        memory=_CompoundRuntimeMemory(),
    )

    windows = run.observations.latest("surrounding_media_windows")
    assert windows is not None
    assert windows.data["surrounding_dates"] == ["2024-12-15", "2024-12-16", "2024-12-17"]
    assert "2024-12-15" in run.answer_markdown
    assert "2024-12-16" in run.answer_markdown
    assert "2024-12-17" in run.answer_markdown
    assert "2025-01-05" not in run.answer_markdown
    assert "2026-02-09" not in run.answer_markdown


def test_private_full_runtime_compound_answer_passes_final_verifier(tmp_path) -> None:
    settings = _private_full_settings(tmp_path)

    run = run_agent(
        "いおりと喧嘩した日はいつ？その日の前後の日は写真から何をしていた？",
        None,
        settings,
        memory=_CompoundRuntimeMemory(),
    )

    verification = run.observations.latest("final_answer_verification")
    assert verification is not None
    assert verification.data["ok"] is True
    assert "質問分解:" in run.answer_markdown
    assert "source_ref:" in run.answer_markdown
    assert "前日/当日/翌日の写真分析:" in run.answer_markdown


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


class _CompoundRuntimeMemory:
    backend = "test"

    def __init__(self) -> None:
        self.personal = self
        self.notes = self
        self._events = (
            EventEvidence(
                source_id="runtime-conflict-2024-12-16",
                date="2024-12-16",
                event_type="minor_misunderstanding",
                summary="短いLINE上の軽いすれ違い候補。",
                severity=1,
                confidence=0.62,
                evidence_strength=0.62,
            ),
        )
        self._line = (
            LineEvidence(
                source_id="runtime-line-2024-12-16",
                date="2024-12-16",
                summary="短い謝罪と予定確認がある軽いすれ違い候補。",
                excerpt="短い謝罪と予定確認。",
                confidence=0.62,
            ),
        )
        self._media = (
            MediaEvidence(source_id="runtime-media-before", date="2024-12-15", summary="家で過ごした写真候補。"),
            MediaEvidence(source_id="runtime-media-after", date="2024-12-17", summary="カフェのような場所の写真候補。"),
            MediaEvidence(source_id="runtime-media-outside-2025", date="2025-01-05", summary="範囲外media。"),
            MediaEvidence(source_id="runtime-media-outside-2026", date="2026-02-09", summary="範囲外media。"),
        )

    def search_events(self, date_from=None, date_to=None, event_type=None, **_kwargs):
        return [
            item
            for item in self._events
            if (event_type is None or item.event_type == event_type)
            and (not date_from or item.date >= date_from)
            and (not date_to or item.date <= date_to)
        ]

    def search_line(self, _query="", date_from=None, date_to=None, **_kwargs):
        return [
            item
            for item in self._line
            if (not date_from or item.date >= date_from)
            and (not date_to or item.date <= date_to)
        ]

    def search_media(self, _query="", date_from=None, date_to=None, **_kwargs):
        return [
            item
            for item in self._media
            if (not date_from or item.date >= date_from)
            and (not date_to or item.date <= date_to)
        ]

    def search_notes(self, *_args, **_kwargs):
        return []
