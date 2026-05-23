from __future__ import annotations

from dataclasses import dataclass
import re

from relationship_lifelog_agent.agent.conversation_state import ConversationState
from relationship_lifelog_agent.agent.router import route_question


@dataclass(frozen=True)
class QuestionFrame:
    raw_question: str
    normalized_question: str
    intents: tuple[str, ...]
    target_profile_name: str | None = None
    target_profile_id: int | None = None
    date_from: str | None = None
    date_to: str | None = None
    referenced_previous_answer: bool = False
    referenced_event_id: int | None = None
    requested_output: str = "answer"
    needs_evidence: bool = True
    risk_level: str = "relationship_private"

    @property
    def primary_intent(self) -> str:
        return self.intents[0] if self.intents else "general_relationship_qa"


def understand_question(question: str, state: ConversationState | None = None) -> QuestionFrame:
    clean = " ".join(str(question or "").strip().split())
    normalized = clean.lower()
    route = route_question(clean)
    referenced = _references_previous_answer(normalized)
    date_from, date_to = _extract_date_range(clean)
    if referenced and not date_from and state and state.last_observed_dates:
        date_from = state.last_observed_dates[0]
        date_to = state.last_observed_dates[0]
    profile_id = _extract_profile_id(clean)
    return QuestionFrame(
        raw_question=clean,
        normalized_question=normalized,
        intents=route.intents,
        target_profile_name=_extract_target_profile_name(clean),
        target_profile_id=profile_id,
        date_from=date_from,
        date_to=date_to,
        referenced_previous_answer=referenced,
        referenced_event_id=_extract_event_id(clean),
        requested_output=_requested_output(normalized, route.intents),
        needs_evidence=_needs_evidence(route.intents),
        risk_level=_risk_level(normalized),
    )


def _extract_target_profile_name(question: str) -> str | None:
    for pattern in (r"([^\s、。?？]+)との", r"([^\s、。?？]+)と"):
        match = re.search(pattern, question)
        if match:
            candidate = match.group(1).strip()
            if candidate and candidate not in {"その", "これ", "それ"}:
                return candidate
    return None


def _extract_date_range(question: str) -> tuple[str | None, str | None]:
    dates = re.findall(r"(20\d{2})[-/年](\d{1,2})[-/月](\d{1,2})", question)
    normalized = [f"{int(year):04d}-{int(month):02d}-{int(day):02d}" for year, month, day in dates]
    if len(normalized) >= 2:
        return normalized[0], normalized[1]
    if len(normalized) == 1:
        return normalized[0], normalized[0]
    month_match = re.search(r"(20\d{2})年\s*(\d{1,2})月", question)
    if month_match:
        year = int(month_match.group(1))
        month = int(month_match.group(2))
        return f"{year:04d}-{month:02d}-01", None
    return None, None


def _extract_profile_id(question: str) -> int | None:
    match = re.search(r"profile[_\s-]*id\s*[:=]?\s*(\d+)", question, flags=re.IGNORECASE)
    if not match:
        return None
    return int(match.group(1))


def _extract_event_id(question: str) -> int | None:
    match = re.search(r"(?:event[_\s-]*id|候補)\s*[:=]?\s*(\d+)", question, flags=re.IGNORECASE)
    if not match:
        return None
    return int(match.group(1))


def _references_previous_answer(text: str) -> bool:
    return any(token in text for token in ("その日", "その時", "それ", "同じ時期", "前回", "直前", "詳しく"))


def _requested_output(text: str, intents: tuple[str, ...]) -> str:
    if "line" in text or "LINE" in text or "ライン" in text:
        return "line_detail_summary"
    if "メモ" in text or "note" in text:
        return "note_context_summary"
    if "根拠" in text or "詳しく" in text:
        return "evidence_detail_summary"
    if "conflict_frequency" in intents:
        return "aggregate_with_examples"
    return "answer"


def _needs_evidence(intents: tuple[str, ...]) -> bool:
    return any(intent != "general_relationship_qa" for intent in intents)


def _risk_level(text: str) -> str:
    if any(token in text for token in ("喧嘩", "恋人", "関係", "line", "返信", "メモ")):
        return "relationship_private"
    return "general_private"
