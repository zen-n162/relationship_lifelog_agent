from __future__ import annotations

from dataclasses import dataclass

from relationship_lifelog_agent.adapters.mock import MockRelationshipMemory
from relationship_lifelog_agent.adapters.notes_lifelog import NotesSqliteReadOnlyAdapter
from relationship_lifelog_agent.adapters.personal_lifelog import PersonalSqliteReadOnlyAdapter
from relationship_lifelog_agent.config import Settings


def build_mock_memory() -> MockRelationshipMemory:
    memory = MockRelationshipMemory()
    memory.backend = "mock"
    return memory


@dataclass
class RelationshipMemory:
    personal: object
    notes: object
    backend: str

    def warnings(self) -> list[str]:
        warnings: list[str] = []
        for adapter in (self.personal, self.notes):
            adapter_warnings = getattr(adapter, "warnings", None)
            if callable(adapter_warnings):
                warnings.extend(adapter_warnings())
        return list(dict.fromkeys(warnings))


def build_memory(settings: Settings | None = None) -> object:
    settings = settings or Settings()
    if settings.adapter.backend == "mock":
        return build_mock_memory()
    return RelationshipMemory(
        personal=PersonalSqliteReadOnlyAdapter(
            settings.paths.personal_lifelog_db,
            excerpt_chars=settings.privacy.max_private_quote_chars,
        ),
        notes=NotesSqliteReadOnlyAdapter(
            settings.paths.notes_lifelog_db,
            excerpt_chars=settings.privacy.max_private_quote_chars,
        ),
        backend="upstream_readonly",
    )
