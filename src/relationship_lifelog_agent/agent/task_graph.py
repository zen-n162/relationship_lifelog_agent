from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Any, Literal


TaskStatus = Literal["pending", "running", "done", "skipped", "failed"]


@dataclass(frozen=True)
class TaskNode:
    task_id: str
    name: str
    depends_on: tuple[str, ...] = ()
    required: bool = True
    status: TaskStatus = "pending"
    result: dict[str, Any] = field(default_factory=dict)
    error: str | None = None


@dataclass(frozen=True)
class TaskGraph:
    nodes: dict[str, TaskNode] = field(default_factory=dict)

    def add_node(self, node: TaskNode) -> "TaskGraph":
        if node.task_id in self.nodes:
            raise ValueError(f"duplicate task_id: {node.task_id}")
        unknown = set(node.depends_on) - set(self.nodes)
        if unknown:
            raise ValueError(f"unknown dependencies for {node.task_id}: {', '.join(sorted(unknown))}")
        return replace(self, nodes={**self.nodes, node.task_id: node})

    def update_node(self, task_id: str, **changes: Any) -> "TaskGraph":
        if task_id not in self.nodes:
            raise KeyError(task_id)
        return replace(self, nodes={**self.nodes, task_id: replace(self.nodes[task_id], **changes)})

    def mark_running(self, task_id: str) -> "TaskGraph":
        return self.update_node(task_id, status="running")

    def mark_done(self, task_id: str, result: dict[str, Any] | None = None) -> "TaskGraph":
        return self.update_node(task_id, status="done", result=result or {}, error=None)

    def mark_skipped(self, task_id: str, reason: str) -> "TaskGraph":
        return self.update_node(task_id, status="skipped", error=reason)

    def mark_failed(self, task_id: str, error: str) -> "TaskGraph":
        return self.update_node(task_id, status="failed", error=error)

    def get(self, task_id: str) -> TaskNode:
        if task_id not in self.nodes:
            raise KeyError(task_id)
        return self.nodes[task_id]

    def ready_nodes(self) -> tuple[TaskNode, ...]:
        return tuple(
            node
            for node in self.topological_order()
            if node.status == "pending"
            and all(self.nodes[dep].status in {"done", "skipped"} for dep in node.depends_on)
        )

    def topological_order(self) -> tuple[TaskNode, ...]:
        indegree = {task_id: 0 for task_id in self.nodes}
        children: dict[str, list[str]] = {task_id: [] for task_id in self.nodes}
        for task_id, node in self.nodes.items():
            for dep in node.depends_on:
                indegree[task_id] += 1
                children[dep].append(task_id)
        queue = [task_id for task_id, degree in indegree.items() if degree == 0]
        ordered: list[TaskNode] = []
        while queue:
            task_id = queue.pop(0)
            ordered.append(self.nodes[task_id])
            for child in children[task_id]:
                indegree[child] -= 1
                if indegree[child] == 0:
                    queue.append(child)
        if len(ordered) != len(self.nodes):
            raise ValueError("task graph contains a cycle")
        return tuple(ordered)

    def is_complete(self) -> bool:
        return all(node.status in {"done", "skipped"} for node in self.nodes.values())

    def failed_nodes(self) -> tuple[TaskNode, ...]:
        return tuple(node for node in self.nodes.values() if node.status == "failed")

    def to_debug_dict(self) -> dict[str, dict[str, Any]]:
        return {
            task_id: {
                "name": node.name,
                "depends_on": list(node.depends_on),
                "required": node.required,
                "status": node.status,
                "result": node.result,
                "error": node.error,
            }
            for task_id, node in self.nodes.items()
        }


def private_full_task_graph() -> TaskGraph:
    graph = TaskGraph()
    for node in (
        TaskNode("understand_question", "understand_question"),
        TaskNode("decompose_compound_question", "decompose_compound_question", ("understand_question",), required=False),
        TaskNode("resolve_profile", "resolve_profile", ("decompose_compound_question",)),
        TaskNode("determine_scope", "determine_scope", ("understand_question",)),
        TaskNode("check_answerability", "check_answerability", ("resolve_profile", "determine_scope")),
        TaskNode("find_conflict_candidate_dates", "find_conflict_candidate_dates", ("check_answerability",), required=False),
        TaskNode(
            "build_surrounding_media_windows",
            "build_surrounding_media_windows",
            ("find_conflict_candidate_dates",),
            required=False,
        ),
        TaskNode(
            "analyze_surrounding_media",
            "analyze_surrounding_media",
            ("build_surrounding_media_windows",),
            required=False,
        ),
        TaskNode("load_full_data", "load_full_data", ("analyze_surrounding_media",)),
        TaskNode("build_manifest", "build_manifest", ("load_full_data",)),
        TaskNode("decide_context_mode", "decide_context_mode", ("build_manifest",)),
        TaskNode("build_batches", "build_batches", ("decide_context_mode",)),
        TaskNode("analyze_single_context", "analyze_single_context", ("decide_context_mode",), required=False),
        TaskNode("analyze_full_batch", "analyze_full_batch", ("build_batches",), required=False),
        TaskNode("synthesize_full_range", "synthesize_full_range", ("analyze_full_batch",), required=False),
        TaskNode("verify_answer", "verify_answer", ("analyze_single_context", "synthesize_full_range")),
        TaskNode("compose_answer", "compose_answer", ("verify_answer",)),
    ):
        graph = graph.add_node(node)
    return graph
