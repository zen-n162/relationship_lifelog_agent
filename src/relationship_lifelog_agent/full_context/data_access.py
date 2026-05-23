from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable

from relationship_lifelog_agent.adapters.types import LineEvidence, MediaEvidence, NoteEvidence, ThoughtEvidence
from relationship_lifelog_agent.agent.memory import build_memory
from relationship_lifelog_agent.analytics.llm_cache import source_window_hash
from relationship_lifelog_agent.config import Settings
from relationship_lifelog_agent.full_context.types import (
    FullFaceItem,
    FullLineItem,
    FullLocationItem,
    FullMediaItem,
    FullNoteItem,
    FullScanBatch,
)


@dataclass(frozen=True)
class FullContextDataSet:
    line_items: tuple[FullLineItem, ...] = ()
    note_items: tuple[FullNoteItem, ...] = ()
    media_items: tuple[FullMediaItem, ...] = ()
    face_items: tuple[FullFaceItem, ...] = ()
    location_items: tuple[FullLocationItem, ...] = ()
    warnings: tuple[str, ...] = ()
    max_items_per_run: int = 0
    truncated_by_limit: bool = False

    def as_full_data_mapping(self) -> dict[str, tuple[Any, ...]]:
        return {
            "line_items": self.line_items,
            "note_items": self.note_items,
            "media_items": self.media_items,
            "face_items": self.face_items,
            "location_items": self.location_items,
        }


def load_full_context_data(
    settings: Settings,
    *,
    profile_id: int | None,
    date_from: str | None,
    date_to: str | None,
    scope: str,
    mode: str = "private",
    max_items_per_run: int | None = None,
    memory: object | None = None,
) -> FullContextDataSet:
    """Load full-context item metadata through existing read-only adapters.

    The returned dataclasses may carry short adapter excerpts for private full
    prompt packing, but this function does not print, log, or persist raw
    payloads. Upstream adapters remain responsible for read-only access.
    """

    del profile_id
    limit = max(1, int(max_items_per_run or settings.analysis.full_scan.max_items_per_run))
    memory = memory or build_memory(settings)
    personal = getattr(memory, "personal", memory)
    notes = getattr(memory, "notes", memory)
    query = ""
    line = _call_adapter(personal, "search_line", query, date_from=date_from, date_to=date_to, limit=limit, mode=mode)
    media = _call_adapter(personal, "search_media", query, date_from=date_from, date_to=date_to, limit=limit, mode=mode)
    note = _call_adapter(notes, "search_notes", query, date_from=date_from, date_to=date_to, limit=limit, mode=mode)
    thoughts = _call_adapter(notes, "search_thoughts", query, date_from=date_from, date_to=date_to, limit=limit, mode=mode)
    warnings = list(_adapter_warnings(memory))
    all_line_items = tuple(_line_item(item) for item in line if isinstance(item, LineEvidence))
    all_note_items = (
        *tuple(_note_item(item) for item in note if isinstance(item, NoteEvidence)),
        *tuple(_thought_item(item) for item in thoughts if isinstance(item, ThoughtEvidence)),
    )
    all_media_items = tuple(_media_item(item) for item in media if isinstance(item, MediaEvidence))
    limited = _limit_items(limit, all_line_items, all_note_items, all_media_items)
    loaded_total = len(all_line_items) + len(all_note_items) + len(all_media_items)
    returned_total = len(limited["line_items"]) + len(limited["note_items"]) + len(limited["media_items"])
    truncated = loaded_total > returned_total or any(
        count >= limit for count in (len(line), len(media), len(note), len(thoughts))
    )
    if truncated:
        warnings.append(
            "max_items_per_run に達したsourceがあります。全件確認には値を上げるか、期間を分割してください。"
        )
    if scope == "all" and (date_from or date_to):
        warnings.append("scope=all では date_from/date_to を使わず全期間として扱います。")
    return FullContextDataSet(
        line_items=limited["line_items"],
        note_items=limited["note_items"],
        media_items=limited["media_items"],
        warnings=tuple(dict.fromkeys(warnings)),
        max_items_per_run=limit,
        truncated_by_limit=truncated,
    )


def full_context_batch_input_hash(batch: FullScanBatch) -> str:
    return source_window_hash(
        {
            "date_from": batch.date_from,
            "date_to": batch.date_to,
            "source_types": list(batch.source_types),
            "source_refs": list(batch.source_refs),
            "item_count": len(batch.source_refs),
        }
    )


def _call_adapter(adapter: object, method_name: str, *args: Any, **kwargs: Any) -> list[Any]:
    method = getattr(adapter, method_name, None)
    if not callable(method):
        return []
    try:
        value = method(*args, **kwargs)
    except TypeError:
        trimmed = {key: value for key, value in kwargs.items() if key != "mode"}
        try:
            value = method(*args, **trimmed)
        except Exception:
            return []
    except Exception:
        return []
    return list(value or ())


def _adapter_warnings(memory: object) -> Iterable[str]:
    warnings = getattr(memory, "warnings", None)
    if callable(warnings):
        yield from (str(item) for item in warnings())
        return
    for adapter in (getattr(memory, "personal", None), getattr(memory, "notes", None)):
        adapter_warnings = getattr(adapter, "warnings", None)
        if callable(adapter_warnings):
            yield from (str(item) for item in adapter_warnings())


def _limit_items(
    limit: int,
    line_items: tuple[FullLineItem, ...],
    note_items: tuple[FullNoteItem, ...],
    media_items: tuple[FullMediaItem, ...],
) -> dict[str, tuple[Any, ...]]:
    """Apply max_items_per_run across all loaded source types."""

    remaining = max(0, limit)
    limited_line = line_items[:remaining]
    remaining -= len(limited_line)
    limited_notes = note_items[: max(0, remaining)]
    remaining -= len(limited_notes)
    limited_media = media_items[: max(0, remaining)]
    return {
        "line_items": limited_line,
        "note_items": limited_notes,
        "media_items": limited_media,
    }


def _line_item(item: LineEvidence) -> FullLineItem:
    return FullLineItem(
        source_ref=item.source_pointer or f"line_message:{item.source_id}",
        source_id=item.source_id,
        conversation_id=item.conversation_id,
        timestamp=item.date,
        speaker_id=item.speaker_id,
        speaker_label=None,
        raw_text=_short_text(item.excerpt or item.summary, 2000),
        raw_metadata={"summary": _short_text(item.summary, 240), "role": item.role},
    )


def _note_item(item: NoteEvidence) -> FullNoteItem:
    return FullNoteItem(
        source_ref=item.source_pointer or f"note:{item.source_id}",
        source_id=item.source_id,
        note_id=item.note_id or item.source_id,
        created_at=item.date,
        raw_body=_short_text(item.excerpt or item.summary, 4000),
        tags=tuple(str(tag) for tag in item.metadata.get("tags", ()) if tag),
        raw_metadata={"summary": _short_text(item.summary, 240), "category": item.category},
    )


def _thought_item(item: ThoughtEvidence) -> FullNoteItem:
    return FullNoteItem(
        source_ref=item.source_pointer or f"thought:{item.source_id}",
        source_id=item.source_id,
        note_id=item.thought_id or item.source_id,
        created_at=item.date,
        raw_body=_short_text(item.excerpt or item.summary, 4000),
        tags=tuple(str(tag) for tag in item.metadata.get("tags", ()) if tag),
        raw_metadata={"summary": _short_text(item.summary, 240), "thought_type": item.thought_type},
    )


def _media_item(item: MediaEvidence) -> FullMediaItem:
    return FullMediaItem(
        source_ref=item.source_pointer or f"media_caption:{item.source_id}",
        source_id=item.source_id,
        media_id=item.media_id or item.source_id,
        captured_at=item.date,
        place_label=item.place_id,
        vlm_caption=_short_text(item.excerpt or item.summary, 1000),
        raw_metadata={"summary": _short_text(item.summary, 240), "media_type": item.media_type},
    )


def _short_text(value: object, limit: int) -> str | None:
    text = " ".join(str(value or "").split())
    if not text:
        return None
    if len(text) <= limit:
        return text
    return text[: max(0, limit - 3)].rstrip() + "..."
