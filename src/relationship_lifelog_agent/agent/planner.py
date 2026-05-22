from __future__ import annotations

from dataclasses import dataclass, field
from calendar import monthrange
import re
from typing import Any

from relationship_lifelog_agent.agent.router import RouteResult


SUPPORTED_INTENTS = (
    "conflict_frequency",
    "conflict_timeline",
    "post_conflict_activity",
    "emotional_note_lookup",
    "monthly_relationship_review",
    "general_relationship_qa",
)


@dataclass(frozen=True)
class AdapterCall:
    adapter: str
    method: str
    query: str | None = None
    kwargs: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class QueryPlan:
    route: RouteResult
    primary_intent: str
    calls: tuple[AdapterCall, ...]
    month: str | None = None
    use_mock_adapters: bool = True
    requires_private_evidence: bool = False


def build_plan(route: RouteResult) -> QueryPlan:
    primary_intent = _primary_supported_intent(route.intents)
    month = _extract_month(route.question)
    return QueryPlan(
        route=route,
        primary_intent=primary_intent,
        calls=_calls_for_intent(primary_intent, month=month),
        month=month,
        use_mock_adapters=True,
        requires_private_evidence=False,
    )


def _primary_supported_intent(intents: tuple[str, ...]) -> str:
    for intent in intents:
        if intent in SUPPORTED_INTENTS:
            return intent
    return "general_relationship_qa"


def _calls_for_intent(intent: str, month: str | None) -> tuple[AdapterCall, ...]:
    if intent == "conflict_frequency":
        return (
            AdapterCall("personal", "search_events", kwargs={"event_type": None}),
            AdapterCall("personal", "search_line", query="喧嘩 すれ違い 謝罪"),
            AdapterCall("notes", "search_notes", query="反省 不安"),
        )
    if intent == "conflict_timeline":
        return (
            AdapterCall("personal", "search_events", kwargs={"event_type": None}),
            AdapterCall("personal", "search_line", query="喧嘩 すれ違い 謝罪"),
            AdapterCall("notes", "search_notes", query="反省 不安"),
        )
    if intent == "post_conflict_activity":
        return (
            AdapterCall("personal", "search_events", kwargs={"event_type": None}),
            AdapterCall("personal", "search_line", query="喧嘩 仲直り 通常会話"),
            AdapterCall("personal", "search_media", query="喧嘩後 外出 場所"),
            AdapterCall("notes", "search_notes", query="反省"),
        )
    if intent == "emotional_note_lookup":
        return (
            AdapterCall("notes", "search_notes", query="反省 不安 メモ"),
            AdapterCall("notes", "search_thoughts", query="反省 不安 自分側"),
        )
    if intent == "monthly_relationship_review":
        target_month = month or "2025-01"
        date_from, date_to = _month_range(target_month)
        return (
            AdapterCall("personal", "get_month_summary", kwargs={"month": target_month}),
            AdapterCall("notes", "get_monthly_reflection", kwargs={"month": target_month}),
            AdapterCall("personal", "search_events", kwargs={"date_from": date_from, "date_to": date_to}),
            AdapterCall("personal", "search_media", query="喧嘩後 外出", kwargs={"date_from": date_from, "date_to": date_to}),
            AdapterCall("notes", "search_notes", query="反省 不安", kwargs={"date_from": date_from, "date_to": date_to}),
        )
    return ()


def _extract_month(question: str) -> str | None:
    match = re.search(r"(20\d{2})年\s*(\d{1,2})月", question)
    if not match:
        return None
    year, month = match.groups()
    return f"{year}-{int(month):02d}"


def _month_range(month: str) -> tuple[str, str]:
    year_text, month_text = month.split("-", 1)
    year = int(year_text)
    month_num = int(month_text)
    last_day = monthrange(year, month_num)[1]
    return f"{year}-{month_num:02d}-01", f"{year}-{month_num:02d}-{last_day:02d}"
