from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class EvidenceContract:
    intent: str
    primary: tuple[str, ...]
    supporting: tuple[str, ...]
    excluded: tuple[str, ...]

    def excludes(self, evidence_kind: str) -> bool:
        return evidence_kind in self.excluded


CONFLICT_FREQUENCY_CONTRACT = EvidenceContract(
    intent="conflict_frequency",
    primary=("line_message", "line_signal_aggregate", "reviewed_relationship_event"),
    supporting=(
        "date_near_relationship_note",
        "reply_delay_metric",
        "call_log",
        "reconciliation_signal",
        "post_conflict_outing",
    ),
    excluded=(
        "undated_note",
        "1970-01-01 note",
        "unrelated_note",
        "unverified_face_cluster",
        "weak_signal_only",
        "long_raw_note_text",
        "long_raw_line_text",
    ),
)

POST_CONFLICT_ACTIVITY_CONTRACT = EvidenceContract(
    intent="post_conflict_activity",
    primary=("relationship_event", "media_event", "place_event", "person_verified_media_link"),
    supporting=("LINE promise", "note reflection", "date-near summary"),
    excluded=("unverified person candidate", "exact GPS in public mode", "photo path in public mode"),
)

CONFLICT_WITH_SURROUNDING_MEDIA_CONTRACT = EvidenceContract(
    intent="conflict_dates_with_surrounding_media",
    primary=("line_message", "line_window_analysis", "reviewed_relationship_event"),
    supporting=("media_day_summary", "media_caption", "date_near_relationship_note"),
    excluded=(
        "post_conflict_activity_window",
        "media_outside_exact_window",
        "undated_note",
        "1970-01-01 note",
        "unrelated_note",
        "long_raw_note_text",
        "long_raw_line_text",
        "exact GPS",
        "photo path",
        "face data",
    ),
)

DEFAULT_CONTRACT = EvidenceContract(
    intent="general_relationship_qa",
    primary=("reviewed_relationship_event", "line_message", "note"),
    supporting=("date-near summary",),
    excluded=("raw_private_text", "exact GPS", "face data", "private path"),
)


def evidence_contract_for_intent(intent: str) -> EvidenceContract:
    if intent == "conflict_dates_with_surrounding_media":
        return CONFLICT_WITH_SURROUNDING_MEDIA_CONTRACT
    if intent in {"conflict_frequency", "conflict_timeline"}:
        return CONFLICT_FREQUENCY_CONTRACT
    if intent == "post_conflict_activity":
        return POST_CONFLICT_ACTIVITY_CONTRACT
    return DEFAULT_CONTRACT
