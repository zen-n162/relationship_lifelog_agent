from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from relationship_lifelog_agent.adapters.types import MonthlyReflection, NoteEvidence, ThoughtEvidence


class NotesLifelogAdapter(Protocol):
    """Read-only interface for notes_lifelog_rag."""

    def search_notes(
        self,
        query: str,
        date_from: str | None = None,
        date_to: str | None = None,
        limit: int = 50,
    ) -> list[NoteEvidence]:
        ...

    def search_thoughts(
        self,
        query: str,
        date_from: str | None = None,
        date_to: str | None = None,
        limit: int = 50,
    ) -> list[ThoughtEvidence]:
        ...

    def get_monthly_reflection(self, month: str) -> MonthlyReflection:
        ...


@dataclass(frozen=True)
class DisabledNotesLifelogAdapter:
    root_path: str

    def _not_enabled(self) -> None:
        raise NotImplementedError("Real notes_lifelog_rag integration is not enabled in the MVP.")

    def search_notes(
        self,
        query: str,
        date_from: str | None = None,
        date_to: str | None = None,
        limit: int = 50,
    ) -> list[NoteEvidence]:
        del query, date_from, date_to, limit
        self._not_enabled()

    def search_thoughts(
        self,
        query: str,
        date_from: str | None = None,
        date_to: str | None = None,
        limit: int = 50,
    ) -> list[ThoughtEvidence]:
        del query, date_from, date_to, limit
        self._not_enabled()

    def get_monthly_reflection(self, month: str) -> MonthlyReflection:
        del month
        self._not_enabled()
