from __future__ import annotations

from dataclasses import dataclass
import re

from relationship_lifelog_agent.agent.conversation_state import ConversationState
from relationship_lifelog_agent.agent.router import route_question
from relationship_lifelog_agent.config import Settings
from relationship_lifelog_agent.llm.local_client import LocalLlmClient
from relationship_lifelog_agent.llm.prompts import SYSTEM_POLICY
from relationship_lifelog_agent.llm.schemas import QUESTION_FRAME_SCHEMA
from relationship_lifelog_agent.llm.usage import LlmUsageTrace


ALLOWED_INTENTS = {
    "conflict_dates_with_surrounding_media",
    "conflict_date_lookup",
    "surrounding_media_summary",
    "line_window_analysis",
    "note_window_analysis",
    "media_day_summary",
    "conflict_frequency",
    "conflict_timeline",
    "post_conflict_activity",
    "emotional_note_lookup",
    "monthly_relationship_review",
    "general_relationship_qa",
    "evidence_lookup",
    "reply_delay_analysis",
    "reconciliation_duration",
    "promise_tracking",
    "outing_history",
    "relationship_memory_search",
}


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


def understand_question(
    question: str,
    state: ConversationState | None = None,
    *,
    settings: Settings | None = None,
    llm_client: LocalLlmClient | None = None,
    usage: LlmUsageTrace | None = None,
) -> QuestionFrame:
    clean = " ".join(str(question or "").strip().split())
    if settings and _should_use_llm(settings, llm_client):
        frame = _understand_with_llm(clean, state, llm_client, usage)
        if frame is not None:
            return frame
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


def _should_use_llm(settings: Settings, llm_client: LocalLlmClient | None) -> bool:
    return bool(
        settings.llm.enabled
        and settings.llm.model
        and settings.llm.use_for_question_understanding
        and llm_client
        and llm_client.is_configured()
    )


def _understand_with_llm(
    question: str,
    state: ConversationState | None,
    llm_client: LocalLlmClient | None,
    usage: LlmUsageTrace | None,
) -> QuestionFrame | None:
    assert llm_client is not None
    prompt = {
        "task": "Extract a safe QuestionFrame for the relationship evidence review app.",
        "question": question,
        "previous_observed_dates": list(state.last_observed_dates) if state else [],
        "allowed_intents": sorted(ALLOWED_INTENTS),
        "rules": [
            "Do not infer relationship labels.",
            "Do not infer person to LINE speaker links.",
            "Return null for unknown profile names.",
            "Use dates only if explicitly present or previous_observed_dates are referenced.",
        ],
    }
    result = llm_client.chat(
        [
            {"role": "system", "content": SYSTEM_POLICY},
            {"role": "user", "content": str(prompt)},
        ],
        schema=QUESTION_FRAME_SCHEMA,
        expect_json=True,
    )
    if not result.ok or not isinstance(result.data, dict):
        if usage:
            usage.record(
                stage="question_understanding",
                backend="rule",
                success=False,
                latency_ms=result.latency_ms,
                structured_output_used=result.structured_output_used,
                fallback_reason=result.error or "empty llm question frame",
            )
        return None
    try:
        frame = _frame_from_llm_data(question, result.data, state)
    except ValueError as exc:
        if usage:
            usage.record(
                stage="question_understanding",
                backend="rule",
                success=False,
                latency_ms=result.latency_ms,
                structured_output_used=result.structured_output_used,
                fallback_reason=str(exc),
            )
        return None
    if usage:
        usage.record(
            stage="question_understanding",
            backend="llm",
            success=True,
            latency_ms=result.latency_ms,
            structured_output_used=result.structured_output_used,
            schema_validation_success=True,
        )
    return frame


def _frame_from_llm_data(question: str, data: dict[str, object], state: ConversationState | None) -> QuestionFrame:
    clean = " ".join(question.strip().split())
    normalized = clean.lower()
    raw_intents = data.get("intents")
    if isinstance(raw_intents, str):
        raw_intents = [raw_intents]
    if not isinstance(raw_intents, list):
        raw_intent = data.get("intent")
        raw_intents = [raw_intent] if isinstance(raw_intent, str) else []
    routed_intents = route_question(clean).intents
    intents = tuple(dict.fromkeys(str(intent) for intent in raw_intents if isinstance(intent, str) and intent in ALLOWED_INTENTS))
    if "conflict_dates_with_surrounding_media" in routed_intents:
        intents = tuple(dict.fromkeys(("conflict_dates_with_surrounding_media", *routed_intents, *intents)))
    if not intents:
        intents = routed_intents
    referenced = bool(data.get("referenced_previous_answer"))
    date_from = _clean_date_value(data.get("date_from"))
    date_to = _clean_date_value(data.get("date_to"))
    if referenced and not date_from and state and state.last_observed_dates:
        date_from = state.last_observed_dates[0]
        date_to = state.last_observed_dates[0]
    return QuestionFrame(
        raw_question=clean,
        normalized_question=normalized,
        intents=intents,
        target_profile_name=_clean_optional_text(data.get("target_profile_name")) or _extract_target_profile_name(clean),
        target_profile_id=_extract_profile_id(clean),
        date_from=date_from,
        date_to=date_to,
        referenced_previous_answer=referenced or _references_previous_answer(normalized),
        referenced_event_id=_extract_event_id(clean),
        requested_output=_allowed_requested_output(data.get("requested_output")) or _requested_output(normalized, intents),
        needs_evidence=bool(data.get("needs_evidence", _needs_evidence(intents))),
        risk_level=_allowed_risk_level(data.get("risk_level")) or _risk_level(normalized),
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


def _clean_date_value(value: object) -> str | None:
    text = _clean_optional_text(value)
    if not text:
        return None
    match = re.fullmatch(r"20\d{2}-\d{2}-\d{2}", text)
    return text if match else None


def _clean_optional_text(value: object) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text or text.lower() == "null":
        return None
    return text[:80]


def _allowed_requested_output(value: object) -> str | None:
    text = _clean_optional_text(value)
    allowed = {
        "answer",
        "aggregate_with_examples",
        "line_detail_summary",
        "note_context_summary",
        "evidence_detail_summary",
        "compound_conflict_media_summary",
        "media_day_summary",
    }
    return text if text in allowed else None


def _allowed_risk_level(value: object) -> str | None:
    text = _clean_optional_text(value)
    return text if text in {"relationship_private", "general_private"} else None


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
    if "conflict_dates_with_surrounding_media" in intents:
        return "compound_conflict_media_summary"
    if "surrounding_media_summary" in intents:
        return "media_day_summary"
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
