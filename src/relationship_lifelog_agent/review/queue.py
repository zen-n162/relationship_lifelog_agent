from __future__ import annotations

from relationship_lifelog_agent.adapters.types import RelationshipEvent


def review_queue(events: list[RelationshipEvent]) -> list[RelationshipEvent]:
    return [event for event in events if event.review_status == "candidate"]
