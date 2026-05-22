from __future__ import annotations

from relationship_lifelog_agent.adapters.mock import MockRelationshipMemory


def build_mock_memory() -> MockRelationshipMemory:
    return MockRelationshipMemory()
