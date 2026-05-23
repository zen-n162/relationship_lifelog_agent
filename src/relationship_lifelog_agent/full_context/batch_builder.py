from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Sequence

from relationship_lifelog_agent.full_context.context_budget import (
    estimate_chars_for_item,
    estimate_tokens_for_item,
)
from relationship_lifelog_agent.full_context.manifest import _item_day
from relationship_lifelog_agent.full_context.types import (
    FullFaceItem,
    FullLineItem,
    FullLocationItem,
    FullMediaItem,
    FullNoteItem,
    FullScanBatch,
)


@dataclass(frozen=True)
class BatchCoverageReport:
    ok: bool
    total_source_refs: int
    covered_source_refs: int
    missing_source_refs: tuple[str, ...]
    duplicate_source_refs: tuple[str, ...]


@dataclass(frozen=True)
class _BatchItem:
    source_type: str
    item: Any
    source_ref: str
    day: str
    estimated_chars: int
    estimated_tokens: int
    sequence: int


def build_chronological_batches(
    *,
    run_id: str,
    line_items: Iterable[FullLineItem] = (),
    note_items: Iterable[FullNoteItem] = (),
    media_items: Iterable[FullMediaItem] = (),
    face_items: Iterable[FullFaceItem] = (),
    location_items: Iterable[FullLocationItem] = (),
    max_items_per_batch: int = 200,
    max_chars_per_batch: int = 80000,
    overlap_items: int = 5,
    max_batches_per_run: int | None = None,
) -> tuple[FullScanBatch, ...]:
    items = _collect_items(line_items, note_items, media_items, face_items, location_items)
    items = tuple(sorted(items, key=lambda item: (item.day == "unknown", item.day, item.sequence)))
    return _build_batches(
        run_id=run_id,
        items=items,
        strategy="chronological",
        max_items_per_batch=max_items_per_batch,
        max_chars_per_batch=max_chars_per_batch,
        overlap_items=overlap_items,
        max_batches_per_run=max_batches_per_run,
    )


def build_source_then_time_batches(
    *,
    run_id: str,
    line_items: Iterable[FullLineItem] = (),
    note_items: Iterable[FullNoteItem] = (),
    media_items: Iterable[FullMediaItem] = (),
    face_items: Iterable[FullFaceItem] = (),
    location_items: Iterable[FullLocationItem] = (),
    max_items_per_batch: int = 200,
    max_chars_per_batch: int = 80000,
    overlap_items: int = 5,
    max_batches_per_run: int | None = None,
) -> tuple[FullScanBatch, ...]:
    grouped = []
    for source_type, collection in (
        ("line", line_items),
        ("note", note_items),
        ("media", media_items),
        ("face", face_items),
        ("location", location_items),
    ):
        grouped.extend(
            sorted(
                _wrap_items(source_type, tuple(collection), start_sequence=len(grouped)),
                key=lambda item: (item.day == "unknown", item.day, item.sequence),
            )
        )
    return _build_batches(
        run_id=run_id,
        items=tuple(grouped),
        strategy="source_then_time",
        max_items_per_batch=max_items_per_batch,
        max_chars_per_batch=max_chars_per_batch,
        overlap_items=overlap_items,
        max_batches_per_run=max_batches_per_run,
    )


def build_hybrid_batches(
    *,
    run_id: str,
    line_items: Iterable[FullLineItem] = (),
    note_items: Iterable[FullNoteItem] = (),
    media_items: Iterable[FullMediaItem] = (),
    face_items: Iterable[FullFaceItem] = (),
    location_items: Iterable[FullLocationItem] = (),
    max_items_per_batch: int = 200,
    max_chars_per_batch: int = 80000,
    overlap_items: int = 5,
    max_batches_per_run: int | None = None,
) -> tuple[FullScanBatch, ...]:
    items = _collect_items(line_items, note_items, media_items, face_items, location_items)
    source_order = {"line": 0, "note": 1, "media": 2, "face": 3, "location": 4}
    items = tuple(
        sorted(
            items,
            key=lambda item: (
                item.day == "unknown",
                item.day,
                source_order.get(item.source_type, 99),
                item.sequence,
            ),
        )
    )
    return _build_batches(
        run_id=run_id,
        items=items,
        strategy="hybrid",
        max_items_per_batch=max_items_per_batch,
        max_chars_per_batch=max_chars_per_batch,
        overlap_items=overlap_items,
        max_batches_per_run=max_batches_per_run,
    )


def validate_batch_coverage(
    *,
    line_items: Iterable[FullLineItem] = (),
    note_items: Iterable[FullNoteItem] = (),
    media_items: Iterable[FullMediaItem] = (),
    face_items: Iterable[FullFaceItem] = (),
    location_items: Iterable[FullLocationItem] = (),
    batches: Iterable[FullScanBatch] = (),
) -> BatchCoverageReport:
    expected = {
        item.source_ref
        for collection in (line_items, note_items, media_items, face_items, location_items)
        for item in collection
    }
    seen: list[str] = [ref for batch in batches for ref in batch.source_refs]
    covered = set(seen)
    missing = tuple(sorted(expected - covered))
    duplicates = tuple(sorted({ref for ref in seen if seen.count(ref) > 1}))
    return BatchCoverageReport(
        ok=not missing,
        total_source_refs=len(expected),
        covered_source_refs=len(expected & covered),
        missing_source_refs=missing,
        duplicate_source_refs=duplicates,
    )


def _build_batches(
    *,
    run_id: str,
    items: Sequence[_BatchItem],
    strategy: str,
    max_items_per_batch: int,
    max_chars_per_batch: int,
    overlap_items: int,
    max_batches_per_run: int | None,
) -> tuple[FullScanBatch, ...]:
    if not items:
        return ()
    if max_items_per_batch < 1:
        raise ValueError("max_items_per_batch must be at least 1")
    if max_chars_per_batch < 1:
        raise ValueError("max_chars_per_batch must be at least 1")
    overlap = max(0, min(overlap_items, max_items_per_batch - 1))
    batches: list[FullScanBatch] = []
    start = 0
    while start < len(items):
        current: list[_BatchItem] = []
        current_chars = 0
        index = start
        while index < len(items) and len(current) < max_items_per_batch:
            item = items[index]
            if item.estimated_chars > max_chars_per_batch:
                raise ValueError(f"single item exceeds max_chars_per_batch: {item.source_ref}")
            if current and current_chars + item.estimated_chars > max_chars_per_batch:
                break
            current.append(item)
            current_chars += item.estimated_chars
            index += 1
        if not current:
            raise ValueError("unable to build a non-empty full-scan batch")
        batch_index = len(batches)
        batches.append(_make_batch(run_id, strategy, batch_index, current))
        if max_batches_per_run is not None and len(batches) >= max_batches_per_run and index < len(items):
            raise ValueError("max_batches_per_run would drop source refs")
        if index >= len(items):
            break
        start = max(index - overlap, start + 1)
    return tuple(batches)


def _make_batch(run_id: str, strategy: str, batch_index: int, items: Sequence[_BatchItem]) -> FullScanBatch:
    dates = sorted({item.day for item in items if item.day != "unknown"})
    source_types = tuple(sorted({item.source_type for item in items}))
    source_refs = tuple(item.source_ref for item in items)
    return FullScanBatch(
        batch_id=f"{run_id}:{strategy}:{batch_index + 1:04d}",
        batch_index=batch_index,
        date_from=dates[0] if dates else None,
        date_to=dates[-1] if dates else None,
        source_types=source_types,
        line_items=tuple(item.item for item in items if item.source_type == "line"),
        note_items=tuple(item.item for item in items if item.source_type == "note"),
        media_items=tuple(item.item for item in items if item.source_type == "media"),
        face_items=tuple(item.item for item in items if item.source_type == "face"),
        location_items=tuple(item.item for item in items if item.source_type == "location"),
        estimated_tokens=sum(item.estimated_tokens for item in items),
        coverage_summary=(
            f"source_refs={len(source_refs)} source_types={','.join(source_types)} "
            f"date_range={dates[0] if dates else 'unknown'}..{dates[-1] if dates else 'unknown'}"
        ),
        source_refs=source_refs,
    )


def _collect_items(
    line_items: Iterable[FullLineItem],
    note_items: Iterable[FullNoteItem],
    media_items: Iterable[FullMediaItem],
    face_items: Iterable[FullFaceItem],
    location_items: Iterable[FullLocationItem],
) -> tuple[_BatchItem, ...]:
    collected: list[_BatchItem] = []
    for source_type, collection in (
        ("line", tuple(line_items)),
        ("note", tuple(note_items)),
        ("media", tuple(media_items)),
        ("face", tuple(face_items)),
        ("location", tuple(location_items)),
    ):
        collected.extend(_wrap_items(source_type, collection, start_sequence=len(collected)))
    return tuple(collected)


def _wrap_items(source_type: str, items: Sequence[Any], *, start_sequence: int) -> tuple[_BatchItem, ...]:
    return tuple(
        _BatchItem(
            source_type=source_type,
            item=item,
            source_ref=item.source_ref,
            day=_item_day(item),
            estimated_chars=estimate_chars_for_item(item),
            estimated_tokens=estimate_tokens_for_item(item),
            sequence=start_sequence + index,
        )
        for index, item in enumerate(items)
    )
