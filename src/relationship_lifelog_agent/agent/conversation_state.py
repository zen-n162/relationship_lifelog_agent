from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass(frozen=True)
class ConversationState:
    active_profile_id: int | None = None
    active_profile_name: str | None = None
    last_question: str | None = None
    last_intents: tuple[str, ...] = ()
    last_date_range: tuple[str | None, str | None] = (None, None)
    last_observed_dates: tuple[str, ...] = ()
    last_event_ids: tuple[int, ...] = ()
    last_evidence_ids: tuple[str, ...] = ()
    last_answer_summary: str | None = None
    pending_setup_requirements: tuple[str, ...] = ()
    pending_review_targets: tuple[int, ...] = ()
    last_query_plan: dict[str, Any] = field(default_factory=dict)
    last_answerability_report: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_value(cls, value: object) -> "ConversationState":
        if isinstance(value, ConversationState):
            return value
        if not isinstance(value, dict):
            return cls()
        return cls(
            active_profile_id=_int_or_none(value.get("active_profile_id")),
            active_profile_name=_str_or_none(value.get("active_profile_name")),
            last_question=_str_or_none(value.get("last_question")),
            last_intents=_tuple_str(value.get("last_intents")),
            last_date_range=_date_range_tuple(value.get("last_date_range")),
            last_observed_dates=_tuple_str(value.get("last_observed_dates")),
            last_event_ids=tuple(item for item in (_int_or_none(raw) for raw in _list(value.get("last_event_ids"))) if item),
            last_evidence_ids=_tuple_str(value.get("last_evidence_ids")),
            last_answer_summary=_str_or_none(value.get("last_answer_summary")),
            pending_setup_requirements=_tuple_str(value.get("pending_setup_requirements")),
            pending_review_targets=tuple(
                item for item in (_int_or_none(raw) for raw in _list(value.get("pending_review_targets"))) if item
            ),
            last_query_plan=dict(value.get("last_query_plan") or {}),
            last_answerability_report=dict(value.get("last_answerability_report") or {}),
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _list(value: object) -> list[object]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    if isinstance(value, tuple):
        return list(value)
    return [value]


def _tuple_str(value: object) -> tuple[str, ...]:
    return tuple(str(item) for item in _list(value) if item not in (None, ""))


def _date_range_tuple(value: object) -> tuple[str | None, str | None]:
    items = list(_tuple_str(value))
    if len(items) >= 2:
        return items[0] or None, items[1] or None
    return None, None


def _int_or_none(value: object) -> int | None:
    try:
        parsed = int(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None
    return parsed if parsed > 0 else None


def _str_or_none(value: object) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None
