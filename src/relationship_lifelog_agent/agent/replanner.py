from __future__ import annotations

from dataclasses import dataclass

from relationship_lifelog_agent.agent.observation_store import ObservationStore
from relationship_lifelog_agent.agent.task_graph import TaskGraph, TaskNode


@dataclass(frozen=True)
class ReplanDecision:
    answerable: bool
    graph: TaskGraph
    added_tasks: tuple[str, ...] = ()
    stop_reason: str | None = None


class Replanner:
    """Small deterministic replanner for the private full runtime scaffold."""

    def replan(
        self,
        graph: TaskGraph,
        observations: ObservationStore,
        *,
        max_steps_remaining: int,
    ) -> ReplanDecision:
        return replan_for_observations(graph, observations, max_steps_remaining=max_steps_remaining)


def replan_for_observations(graph: TaskGraph, observations: ObservationStore, *, max_steps_remaining: int) -> ReplanDecision:
    missing = observations.latest("missing_information")
    if missing is not None:
        return ReplanDecision(
            answerable=False,
            graph=graph,
            stop_reason=missing.summary,
        )
    if max_steps_remaining <= 0:
        return ReplanDecision(answerable=False, graph=graph, stop_reason="max_steps reached")
    if observations.latest("partial_answerability") is not None and "collect_missing_information" not in graph.nodes:
        return ReplanDecision(
            answerable=True,
            graph=graph.add_node(TaskNode("collect_missing_information", "collect_missing_information")),
            added_tasks=("collect_missing_information",),
        )
    answerability = observations.latest("answerability")
    if answerability and answerability.data.get("status") in {"answerable", "answerable_with_caution", "partially_answerable"}:
        return ReplanDecision(answerable=True, graph=_ensure_compose_task(graph))
    return ReplanDecision(answerable=True, graph=graph)


def _ensure_compose_task(graph: TaskGraph) -> TaskGraph:
    if "compose_answer" in graph.nodes:
        return graph
    if "verify_answer" not in graph.nodes:
        graph = graph.add_node(TaskNode("verify_answer", "verify_answer"))
    return graph.add_node(TaskNode("compose_answer", "compose_answer", ("verify_answer",)))
