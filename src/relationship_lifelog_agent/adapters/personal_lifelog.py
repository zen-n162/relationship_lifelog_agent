from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from relationship_lifelog_agent.adapters.types import (
    EventEvidence,
    LineEvidence,
    MediaEvidence,
)


class PersonalLifelogAdapter(Protocol):
    """Read-only interface for personal_lifelog_rag.

    Implementations must not mutate upstream DBs. The MVP ships only mock
    implementations; real adapters should be added as read-only integrations.
    """

    def search_line(
        self,
        query: str,
        date_from: str | None = None,
        date_to: str | None = None,
        speaker_id: str | None = None,
        limit: int = 50,
    ) -> list[LineEvidence]:
        ...

    def search_media(
        self,
        query: str,
        date_from: str | None = None,
        date_to: str | None = None,
        person_id: str | None = None,
        place_id: str | None = None,
        limit: int = 50,
    ) -> list[MediaEvidence]:
        ...

    def search_events(
        self,
        date_from: str | None = None,
        date_to: str | None = None,
        event_type: str | None = None,
        person_id: str | None = None,
        place_id: str | None = None,
        limit: int = 50,
    ) -> list[EventEvidence]:
        ...

    def get_day_summary(self, date: str, mode: str = "private") -> dict[str, str]:
        ...

    def get_month_summary(self, month: str, mode: str = "private") -> dict[str, str]:
        ...


@dataclass(frozen=True)
class DisabledPersonalLifelogAdapter:
    root_path: str

    def _not_enabled(self) -> None:
        raise NotImplementedError("Real personal_lifelog_rag integration is not enabled in the MVP.")

    def search_line(
        self,
        query: str,
        date_from: str | None = None,
        date_to: str | None = None,
        speaker_id: str | None = None,
        limit: int = 50,
    ) -> list[LineEvidence]:
        del query, date_from, date_to, speaker_id, limit
        self._not_enabled()

    def search_media(
        self,
        query: str,
        date_from: str | None = None,
        date_to: str | None = None,
        person_id: str | None = None,
        place_id: str | None = None,
        limit: int = 50,
    ) -> list[MediaEvidence]:
        del query, date_from, date_to, person_id, place_id, limit
        self._not_enabled()

    def search_events(
        self,
        date_from: str | None = None,
        date_to: str | None = None,
        event_type: str | None = None,
        person_id: str | None = None,
        place_id: str | None = None,
        limit: int = 50,
    ) -> list[EventEvidence]:
        del date_from, date_to, event_type, person_id, place_id, limit
        self._not_enabled()

    def get_day_summary(self, date: str, mode: str = "private") -> dict[str, str]:
        del date, mode
        self._not_enabled()

    def get_month_summary(self, month: str, mode: str = "private") -> dict[str, str]:
        del month, mode
        self._not_enabled()
