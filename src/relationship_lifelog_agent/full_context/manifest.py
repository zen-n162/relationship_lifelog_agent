from __future__ import annotations

from collections import defaultdict
from datetime import date, datetime
from typing import Any, Iterable

from relationship_lifelog_agent.full_context.context_budget import (
    DEFAULT_NUM_CTX,
    can_fit_single_context,
    estimate_bundle_chars,
    estimate_bundle_tokens,
)
from relationship_lifelog_agent.full_context.types import (
    FullDataManifest,
    FullFaceItem,
    FullLineItem,
    FullLocationItem,
    FullMediaItem,
    FullNoteItem,
)


def build_full_data_manifest(
    run_id: str,
    profile_id: int | None,
    date_from: str | None,
    date_to: str | None,
    line_items: Iterable[FullLineItem],
    note_items: Iterable[FullNoteItem],
    media_items: Iterable[FullMediaItem],
    face_items: Iterable[FullFaceItem],
    location_items: Iterable[FullLocationItem],
) -> FullDataManifest:
    lines = tuple(line_items)
    notes = tuple(note_items)
    media = tuple(media_items)
    faces = tuple(face_items)
    locations = tuple(location_items)
    estimated_chars = estimate_bundle_chars(
        line_items=lines,
        note_items=notes,
        media_items=media,
        face_items=faces,
        location_items=locations,
    )
    estimated_tokens = estimate_bundle_tokens(
        line_items=lines,
        note_items=notes,
        media_items=media,
        face_items=faces,
        location_items=locations,
    )
    fits = can_fit_single_context(estimated_tokens, {"num_ctx": DEFAULT_NUM_CTX})
    return FullDataManifest(
        run_id=run_id,
        profile_id=profile_id,
        date_from=date_from,
        date_to=date_to,
        line_count=len(lines),
        note_count=len(notes),
        media_count=len(media),
        face_count=len(faces),
        location_count=len(locations),
        date_coverage=_build_date_coverage(date_from, date_to, lines, notes, media, faces, locations),
        source_counts_by_day=_build_source_counts_by_day(lines, notes, media, faces, locations),
        available_payload_types=_available_payload_types(lines, notes, media, faces, locations),
        estimated_chars=estimated_chars,
        estimated_tokens=estimated_tokens,
        can_fit_single_context=fits,
        recommended_mode="single_context" if fits else "iterative_full_scan",
    )


def _build_source_counts_by_day(
    line_items: tuple[FullLineItem, ...],
    note_items: tuple[FullNoteItem, ...],
    media_items: tuple[FullMediaItem, ...],
    face_items: tuple[FullFaceItem, ...],
    location_items: tuple[FullLocationItem, ...],
) -> dict[str, dict[str, int]]:
    counts: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for source_type, items in (
        ("line", line_items),
        ("note", note_items),
        ("media", media_items),
        ("face", face_items),
        ("location", location_items),
    ):
        for item in items:
            counts[_item_day(item)][source_type] += 1
    return {
        day: {source_type: values[source_type] for source_type in sorted(values)}
        for day, values in sorted(counts.items())
    }


def _build_date_coverage(
    date_from: str | None,
    date_to: str | None,
    line_items: tuple[FullLineItem, ...],
    note_items: tuple[FullNoteItem, ...],
    media_items: tuple[FullMediaItem, ...],
    face_items: tuple[FullFaceItem, ...],
    location_items: tuple[FullLocationItem, ...],
) -> dict[str, Any]:
    days = sorted(
        {
            day
            for collection in (line_items, note_items, media_items, face_items, location_items)
            for item in collection
            if (day := _item_day(item)) != "unknown"
        }
    )
    requested_days = _inclusive_day_count(date_from, date_to)
    return {
        "requested_start": date_from,
        "requested_end": date_to,
        "observed_start": days[0] if days else None,
        "observed_end": days[-1] if days else None,
        "days_with_data": len(days),
        "requested_days": requested_days,
        "coverage_ratio": round(len(days) / requested_days, 3) if requested_days else None,
    }


def _available_payload_types(
    line_items: tuple[FullLineItem, ...],
    note_items: tuple[FullNoteItem, ...],
    media_items: tuple[FullMediaItem, ...],
    face_items: tuple[FullFaceItem, ...],
    location_items: tuple[FullLocationItem, ...],
) -> tuple[str, ...]:
    available: set[str] = set()
    if line_items:
        available.add("line")
    if any(item.raw_text for item in line_items):
        available.add("raw_line_text")
    if note_items:
        available.add("note")
    if any(item.raw_body for item in note_items):
        available.add("raw_note_body")
    if media_items:
        available.add("media")
    if any(item.file_path or item.thumbnail_path for item in media_items):
        available.add("photo_paths")
    if any(item.exact_gps for item in media_items) or any(
        item.latitude is not None and item.longitude is not None for item in location_items
    ):
        available.add("exact_gps")
    if any(item.vlm_caption for item in media_items):
        available.add("media_captions")
    if any(item.ocr_text for item in media_items):
        available.add("ocr_text")
    if face_items or any(item.face_crop_paths for item in media_items):
        available.add("face_crops")
    if any(item.embedding for item in face_items) or any(item.face_embeddings for item in media_items):
        available.add("face_embeddings")
    if location_items:
        available.add("locations")
    return tuple(sorted(available))


def _item_day(item: Any) -> str:
    timestamp = (
        getattr(item, "timestamp", None)
        or getattr(item, "created_at", None)
        or getattr(item, "captured_at", None)
        or _metadata_value(item, "timestamp")
        or _metadata_value(item, "created_at")
        or _metadata_value(item, "captured_at")
        or _metadata_value(item, "date")
    )
    parsed = _parse_date(timestamp)
    return parsed.isoformat() if parsed else "unknown"


def _metadata_value(item: Any, key: str) -> Any:
    metadata = getattr(item, "raw_metadata", None)
    if isinstance(metadata, dict):
        return metadata.get(key)
    return None


def _parse_date(value: Any) -> date | None:
    if value is None:
        return None
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    text = str(value).strip()
    if not text:
        return None
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    try:
        return datetime.fromisoformat(text).date()
    except ValueError:
        try:
            return date.fromisoformat(text[:10])
        except ValueError:
            return None


def _inclusive_day_count(date_from: str | None, date_to: str | None) -> int | None:
    start = _parse_date(date_from)
    end = _parse_date(date_to)
    if not start or not end or end < start:
        return None
    return (end - start).days + 1
