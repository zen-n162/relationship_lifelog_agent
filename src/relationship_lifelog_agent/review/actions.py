from __future__ import annotations

from dataclasses import dataclass

from relationship_lifelog_agent.db.repository import RelationshipRepository


ALLOWED_REVIEW_ACTIONS = frozenset(
    {
        "verify",
        "reject",
        "mark_as_misunderstanding",
        "mark_as_joke",
        "mark_as_reconciled",
        "needs_reanalysis",
    }
)


@dataclass(frozen=True)
class ReviewAction:
    event_id: int
    action: str
    note: str | None = None


@dataclass(frozen=True)
class ReviewActionResult:
    event_id: int
    action: str
    review_action_id: int
    review_status: str
    status: str
    event_type: str


def apply_review_action(
    repo: RelationshipRepository,
    *,
    event_id: int,
    action: str,
    note: str | None = None,
) -> ReviewActionResult:
    if action not in ALLOWED_REVIEW_ACTIONS:
        allowed = ", ".join(sorted(ALLOWED_REVIEW_ACTIONS))
        raise ValueError(f"review action must be one of: {allowed}")

    event = repo.get_event(event_id)
    if event is None:
        raise ValueError(f"relationship event not found: {event_id}")

    updates = _event_updates_for_action(action)
    changed = repo.update_event(event_id, **updates)
    if changed == 0:
        raise ValueError(f"relationship event not updated: {event_id}")
    action_id = repo.save_review_action(event_id, action, note=note)
    updated = repo.get_event(event_id)
    if updated is None:
        raise ValueError(f"relationship event disappeared after review action: {event_id}")
    return ReviewActionResult(
        event_id=event_id,
        action=action,
        review_action_id=action_id,
        review_status=str(updated["review_status"]),
        status=str(updated["status"]),
        event_type=str(updated["event_type"]),
    )


def _event_updates_for_action(action: str) -> dict[str, object]:
    if action == "verify":
        return {"review_status": "verified", "status": "candidate"}
    if action == "reject":
        return {"review_status": "rejected", "status": "candidate"}
    if action == "mark_as_misunderstanding":
        return {
            "event_type": "minor_misunderstanding",
            "review_status": "corrected",
            "severity": 1,
            "status": "candidate",
        }
    if action == "mark_as_joke":
        return {"review_status": "corrected", "status": "hidden"}
    if action == "mark_as_reconciled":
        return {"event_type": "reconciliation", "review_status": "verified", "status": "candidate"}
    if action == "needs_reanalysis":
        return {"review_status": "needs_reanalysis", "status": "candidate"}
    raise ValueError(f"unsupported review action: {action}")
