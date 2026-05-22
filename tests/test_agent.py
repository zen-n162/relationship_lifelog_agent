from relationship_lifelog_agent.adapters.mock import MockRelationshipMemory
from relationship_lifelog_agent.agent.executor import execute_plan
from relationship_lifelog_agent.agent.planner import build_plan
from relationship_lifelog_agent.agent.router import route_question


def test_planner_selects_adapter_calls_for_conflict_frequency() -> None:
    plan = build_plan(route_question("喧嘩はどのくらいしている？"))

    assert plan.primary_intent == "conflict_frequency"
    assert ("personal", "search_events") in {(call.adapter, call.method) for call in plan.calls}
    assert ("personal", "search_line") in {(call.adapter, call.method) for call in plan.calls}
    assert ("notes", "search_notes") in {(call.adapter, call.method) for call in plan.calls}


def test_executor_counts_conflict_candidates_from_mock_adapters() -> None:
    plan = build_plan(route_question("喧嘩はどのくらいしている？"))
    result = execute_plan(plan, MockRelationshipMemory())

    assert result.intent == "conflict_frequency"
    assert result.aggregate["conflict_candidates"] == 2
    assert result.aggregate["minor_misunderstanding_candidates"] == 1
    assert "喧嘩候補" in result.summary


def test_executor_finds_post_conflict_activity_dates() -> None:
    plan = build_plan(route_question("喧嘩の後どこへ行っている？"))
    result = execute_plan(plan, MockRelationshipMemory())

    assert result.intent == "post_conflict_activity"
    assert result.aggregate["activity_candidates"] == 2
    assert result.aggregate["days_after_conflict"] == "3, 3"
    assert any("外出候補" in example for example in result.examples)
