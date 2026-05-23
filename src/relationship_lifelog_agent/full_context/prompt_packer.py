from __future__ import annotations

from dataclasses import asdict, is_dataclass
import json
from typing import Any, Iterable, Mapping
from uuid import uuid4

from relationship_lifelog_agent.full_context.context_budget import (
    estimate_tokens_for_text,
)
from relationship_lifelog_agent.full_context.types import (
    FullBatchAnalysis,
    FullDataManifest,
    FullFaceItem,
    FullLineItem,
    FullLocationItem,
    FullMediaItem,
    FullNoteItem,
    FullPromptBundle,
    FullScanBatch,
)
from relationship_lifelog_agent.privacy.raw_payload_policy import RawPayloadPolicy


def build_single_context_prompt(
    question: str,
    profile_context: Mapping[str, Any] | Any,
    manifest: FullDataManifest,
    full_data: Mapping[str, Iterable[Any]] | Any,
    raw_payload_policy: RawPayloadPolicy,
) -> FullPromptBundle:
    payload = {
        "task": "private_full_single_context_analysis",
        "instructions": _instructions(),
        "question": question,
        "profile_context": _safe_profile_context(profile_context),
        "manifest": _manifest_payload(manifest),
        "raw_payload_policy": raw_payload_policy.to_dict(),
        "data": {
            "line_items": [
                sanitize_or_include_item_by_policy(item, raw_payload_policy)
                for item in _get_items(full_data, "line_items")
            ],
            "note_items": [
                sanitize_or_include_item_by_policy(item, raw_payload_policy)
                for item in _get_items(full_data, "note_items")
            ],
            "media_items": [
                sanitize_or_include_item_by_policy(item, raw_payload_policy)
                for item in _get_items(full_data, "media_items")
            ],
            "face_items": [
                sanitize_or_include_item_by_policy(item, raw_payload_policy)
                for item in _get_items(full_data, "face_items")
            ],
            "location_items": [
                sanitize_or_include_item_by_policy(item, raw_payload_policy)
                for item in _get_items(full_data, "location_items")
            ],
        },
        "required_output": _required_output_schema(),
    }
    return _bundle("single-context", raw_payload_policy, payload)


def build_batch_prompt(
    question: str,
    profile_context: Mapping[str, Any] | Any,
    manifest: FullDataManifest,
    batch: FullScanBatch,
    raw_payload_policy: RawPayloadPolicy,
) -> FullPromptBundle:
    payload = {
        "task": "private_full_batch_analysis",
        "instructions": _instructions(),
        "question": question,
        "profile_context": _safe_profile_context(profile_context),
        "manifest": _manifest_payload(manifest),
        "batch": {
            "batch_id": batch.batch_id,
            "batch_index": batch.batch_index,
            "date_from": batch.date_from,
            "date_to": batch.date_to,
            "source_types": batch.source_types,
            "coverage_summary": batch.coverage_summary,
            "source_refs": batch.source_refs,
            "line_items": [sanitize_or_include_item_by_policy(item, raw_payload_policy) for item in batch.line_items],
            "note_items": [sanitize_or_include_item_by_policy(item, raw_payload_policy) for item in batch.note_items],
            "media_items": [sanitize_or_include_item_by_policy(item, raw_payload_policy) for item in batch.media_items],
            "face_items": [sanitize_or_include_item_by_policy(item, raw_payload_policy) for item in batch.face_items],
            "location_items": [
                sanitize_or_include_item_by_policy(item, raw_payload_policy) for item in batch.location_items
            ],
        },
        "raw_payload_policy": raw_payload_policy.to_dict(),
        "required_output": _required_output_schema(),
    }
    return _bundle(batch.batch_id, raw_payload_policy, payload)


def build_synthesis_prompt(
    question: str,
    manifest: FullDataManifest,
    batch_analyses: Iterable[FullBatchAnalysis],
) -> FullPromptBundle:
    analyses = tuple(batch_analyses)
    payload = {
        "task": "private_full_range_synthesis",
        "instructions": [
            "Synthesize only from the structured batch analyses.",
            "Keep source_ref and source_id references intact.",
            "Do not invent counts, dates, identities, relationship labels, or source references.",
            "Separate facts from inferences and keep uncertainty explicit.",
        ],
        "question": question,
        "manifest": _manifest_payload(manifest),
        "batch_analyses": [asdict(analysis) for analysis in analyses],
        "required_output": {
            "summary": "string",
            "answer": "string",
            "source_refs": ["string"],
            "uncertainties": ["string"],
            "confidence": "0.0-1.0",
        },
    }
    prompt_text = _json_prompt(payload)
    return FullPromptBundle(
        bundle_id=f"synthesis-{uuid4().hex[:12]}",
        analysis_mode=manifest.recommended_mode,
        prompt_text=prompt_text,
        source_refs=tuple(ref for analysis in analyses for ref in analysis.relevant_evidence_refs),
        estimated_chars=len(prompt_text),
        estimated_tokens=estimate_tokens_for_text(prompt_text),
        raw_payload_included=False,
        metadata={
            "prompt_logging_allowed": False,
            "raw_payload_cache_allowed": False,
            "included_payload_types": (),
        },
    )


def sanitize_or_include_item_by_policy(item: Any, raw_payload_policy: RawPayloadPolicy) -> dict[str, Any]:
    if isinstance(item, FullLineItem):
        payload = {
            "source_type": "line",
            "source_ref": item.source_ref,
            "source_id": item.source_id,
            "conversation_id": item.conversation_id,
            "timestamp": item.timestamp,
            "speaker_id": item.speaker_id,
            "speaker_label": item.speaker_label,
            "speaker_role": item.speaker_role,
            "message_type": item.message_type,
            "reply_to_id": item.reply_to_id,
            "attachments": item.attachments,
        }
        if raw_payload_policy.should_include_raw_line_text():
            payload["raw_text"] = item.raw_text
        else:
            payload["raw_text"] = _redacted_marker("raw_line_text")
        if raw_payload_policy.should_include_unverified_speaker_candidates():
            payload["raw_metadata"] = item.raw_metadata
        return _compact(payload)
    if isinstance(item, FullNoteItem):
        payload = {
            "source_type": "note",
            "source_ref": item.source_ref,
            "source_id": item.source_id,
            "note_id": item.note_id,
            "title": item.title,
            "created_at": item.created_at,
            "modified_at": item.modified_at,
            "tags": item.tags,
        }
        if raw_payload_policy.should_include_raw_note_text():
            payload["raw_body"] = item.raw_body
        else:
            payload["raw_body"] = _redacted_marker("raw_note_body")
        if raw_payload_policy.should_include_private_file_paths():
            payload["source_path"] = item.source_path
        elif item.source_path:
            payload["source_path"] = _redacted_marker("private_file_path")
        return _compact(payload)
    if isinstance(item, FullMediaItem):
        payload = {
            "source_type": "media",
            "source_ref": item.source_ref,
            "source_id": item.source_id,
            "media_id": item.media_id,
            "captured_at": item.captured_at,
            "place_label": item.place_label,
            "vlm_caption": item.vlm_caption,
            "ocr_text": item.ocr_text,
            "verified_people": item.verified_people,
        }
        if raw_payload_policy.private_full_enabled:
            payload["detected_people"] = item.detected_people
        elif item.detected_people:
            payload["detected_people"] = _redacted_marker("unverified_person_candidates")
        if raw_payload_policy.should_include_photo_paths():
            payload["file_path"] = item.file_path
            payload["thumbnail_path"] = item.thumbnail_path
        else:
            payload["file_path"] = _redacted_marker("photo_path") if item.file_path else None
            payload["thumbnail_path"] = _redacted_marker("photo_path") if item.thumbnail_path else None
        if raw_payload_policy.should_include_exact_gps():
            payload["exact_gps"] = item.exact_gps
        elif item.exact_gps:
            payload["exact_gps"] = _redacted_marker("exact_gps")
        if raw_payload_policy.should_include_face_crops():
            payload["face_crop_paths"] = item.face_crop_paths
        elif item.face_crop_paths:
            payload["face_crop_paths"] = [_redacted_marker("face_crop_path") for _ in item.face_crop_paths]
        if raw_payload_policy.should_include_face_embeddings():
            payload["face_embeddings"] = _embedding_payload(item.face_embeddings, raw_payload_policy)
        elif item.face_embeddings:
            payload["face_embeddings"] = _embedding_summary(item.face_embeddings)
        if raw_payload_policy.should_include_unverified_person_candidates():
            payload["unverified_person_candidates"] = item.unverified_person_candidates
        elif item.unverified_person_candidates:
            payload["unverified_person_candidates"] = _redacted_marker("unverified_person_candidates")
        return _compact(payload)
    if isinstance(item, FullFaceItem):
        payload = {
            "source_type": "face",
            "source_ref": item.source_ref,
            "source_id": item.source_id,
            "face_id": item.face_id,
            "media_id": item.media_id,
            "person_candidate_id": item.person_candidate_id,
            "verified_person_id": item.verified_person_id,
            "bbox": item.bbox,
            "confidence": item.confidence,
        }
        if raw_payload_policy.should_include_face_crops():
            payload["crop_path"] = item.crop_path
        elif item.crop_path:
            payload["crop_path"] = _redacted_marker("face_crop_path")
        if raw_payload_policy.should_include_face_embeddings() and item.embedding:
            payload["embedding"] = (
                item.embedding
                if raw_payload_policy.allow_raw_face_embedding_values
                else {"embedding_count": 1, "embedding_dim": len(item.embedding)}
            )
        elif item.embedding:
            payload["embedding"] = {"embedding_count": 1, "embedding_dim": len(item.embedding)}
        return _compact(payload)
    if isinstance(item, FullLocationItem):
        payload = {
            "source_type": "location",
            "source_ref": item.source_ref,
            "source_id": item.source_id,
            "timestamp": item.timestamp,
            "place_label": item.place_label,
        }
        if raw_payload_policy.should_include_exact_gps():
            payload["latitude"] = item.latitude
            payload["longitude"] = item.longitude
            payload["accuracy"] = item.accuracy
        elif item.latitude is not None or item.longitude is not None:
            payload["exact_gps"] = _redacted_marker("exact_gps")
        return _compact(payload)
    return _compact(_generic_item_payload(item))


def _bundle(bundle_id_prefix: str, raw_payload_policy: RawPayloadPolicy, payload: dict[str, Any]) -> FullPromptBundle:
    prompt_text = _json_prompt(payload)
    source_refs = tuple(_collect_source_refs(payload))
    included_payload_types = _included_payload_types(payload)
    raw_payload_included = any(
        key in included_payload_types
        for key in (
            "raw_line_text",
            "raw_note_body",
            "photo_paths",
            "exact_gps",
            "face_crop_paths",
            "face_embeddings",
            "private_file_paths",
            "unverified_person_candidates",
            "unverified_speaker_candidates",
        )
    )
    return FullPromptBundle(
        bundle_id=f"{bundle_id_prefix}-{uuid4().hex[:12]}",
        analysis_mode=raw_payload_policy.analysis_mode,
        prompt_text=prompt_text,
        source_refs=source_refs,
        estimated_chars=len(prompt_text),
        estimated_tokens=estimate_tokens_for_text(prompt_text),
        raw_payload_included=raw_payload_included,
        metadata={
            "included_payload_types": included_payload_types,
            "prompt_logging_allowed": raw_payload_policy.private_full_enabled
            and raw_payload_policy.allow_full_prompt_logging,
            "raw_payload_cache_allowed": raw_payload_policy.private_full_enabled
            and raw_payload_policy.allow_raw_payload_cache,
            "raw_text_included": "raw_line_text" in included_payload_types
            or "raw_note_body" in included_payload_types,
        },
    )


def _instructions() -> list[str]:
    return [
        "Analyze the provided local private memory data only.",
        "Return source_id and source_ref for every factual claim.",
        "Do not infer relationship labels or identity links.",
        "Do not claim certainty about another person's inner feelings.",
        "Do not change counts or dates; they are verified by Python/SQL.",
        "Separate facts from cautious inferences.",
    ]


def _required_output_schema() -> dict[str, Any]:
    return {
        "answer_summary": "string",
        "facts": [{"claim": "string", "source_ids": ["string"], "source_refs": ["string"]}],
        "inferences": [{"claim": "string", "confidence": "0.0-1.0", "source_refs": ["string"]}],
        "cautions": ["string"],
        "source_refs_used": ["string"],
    }


def _safe_profile_context(profile_context: Mapping[str, Any] | Any) -> dict[str, Any]:
    data = _as_dict(profile_context)
    return {
        key: value
        for key, value in data.items()
        if key
        in {
            "id",
            "profile_id",
            "profile_name",
            "person_source_id",
            "line_speaker_source_id",
            "line_speaker_group_source_id",
            "self_person_source_id",
            "self_line_speaker_source_id",
            "self_line_speaker_group_source_id",
            "visibility",
            "label_source",
        }
    }


def _manifest_payload(manifest: FullDataManifest) -> dict[str, Any]:
    return {
        "run_id": manifest.run_id,
        "profile_id": manifest.profile_id,
        "date_from": manifest.date_from,
        "date_to": manifest.date_to,
        "line_count": manifest.line_count,
        "note_count": manifest.note_count,
        "media_count": manifest.media_count,
        "face_count": manifest.face_count,
        "location_count": manifest.location_count,
        "date_coverage": manifest.date_coverage,
        "source_counts_by_day": manifest.source_counts_by_day,
        "available_payload_types": manifest.available_payload_types,
        "estimated_chars": manifest.estimated_chars,
        "estimated_tokens": manifest.estimated_tokens,
        "recommended_mode": manifest.recommended_mode,
    }


def _get_items(full_data: Mapping[str, Iterable[Any]] | Any, key: str) -> tuple[Any, ...]:
    if isinstance(full_data, Mapping):
        value = full_data.get(key, ())
    else:
        value = getattr(full_data, key, ())
    return tuple(value or ())


def _json_prompt(payload: dict[str, Any]) -> str:
    return json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True)


def _as_dict(value: Mapping[str, Any] | Any) -> dict[str, Any]:
    if value is None:
        return {}
    if isinstance(value, Mapping):
        return dict(value)
    if is_dataclass(value):
        return asdict(value)
    if hasattr(value, "to_dict"):
        result = value.to_dict()
        return dict(result) if isinstance(result, Mapping) else {}
    return {
        key: getattr(value, key)
        for key in dir(value)
        if not key.startswith("_") and not callable(getattr(value, key))
    }


def _generic_item_payload(item: Any) -> dict[str, Any]:
    data = _as_dict(item)
    return {
        "source_type": type(item).__name__,
        "source_ref": data.get("source_ref"),
        "source_id": data.get("source_id"),
    }


def _compact(payload: dict[str, Any]) -> dict[str, Any]:
    return {
        key: value
        for key, value in payload.items()
        if value is not None and value != () and value != [] and value != {}
    }


def _redacted_marker(kind: str) -> str:
    return f"[redacted:{kind}]"


def _embedding_payload(embeddings: tuple[tuple[float, ...], ...], policy: RawPayloadPolicy) -> Any:
    if policy.allow_raw_face_embedding_values:
        return embeddings
    return _embedding_summary(embeddings)


def _embedding_summary(embeddings: tuple[tuple[float, ...], ...]) -> dict[str, int]:
    first_dim = len(embeddings[0]) if embeddings else 0
    return {"embedding_count": len(embeddings), "embedding_dim": first_dim}


def _collect_source_refs(value: Any) -> list[str]:
    refs: list[str] = []
    if isinstance(value, Mapping):
        for key, item in value.items():
            if key == "source_ref" and isinstance(item, str):
                refs.append(item)
            else:
                refs.extend(_collect_source_refs(item))
    elif isinstance(value, (list, tuple)):
        for item in value:
            refs.extend(_collect_source_refs(item))
    return refs


def _included_payload_types(payload: Mapping[str, Any]) -> tuple[str, ...]:
    found: set[str] = set()
    for key, value in _walk_payload(payload):
        if key == "raw_text" and isinstance(value, str) and not value.startswith("[redacted:"):
            found.add("raw_line_text")
        if key == "raw_body" and isinstance(value, str) and not value.startswith("[redacted:"):
            found.add("raw_note_body")
        if key in {"file_path", "thumbnail_path"} and isinstance(value, str) and not _is_redacted_value(value):
            found.add("photo_paths")
        if key in {"source_path"} and isinstance(value, str) and not _is_redacted_value(value):
            found.add("private_file_paths")
        if key in {"exact_gps", "latitude", "longitude"} and value is not None and not _is_redacted_value(value):
            found.add("exact_gps")
        if key in {"face_crop_paths", "crop_path"} and value is not None and not _is_redacted_value(value):
            found.add("face_crop_paths")
        if key in {"face_embeddings", "embedding"} and value not in (None,):
            if isinstance(value, Mapping) and "embedding_dim" in value:
                found.add("face_embedding_summary")
            else:
                found.add("face_embeddings")
        if key == "unverified_person_candidates" and value is not None and not _is_redacted_value(value):
            found.add("unverified_person_candidates")
        if key == "raw_metadata" and value:
            found.add("unverified_speaker_candidates")
    return tuple(sorted(found))


def _walk_payload(value: Any) -> Iterable[tuple[str, Any]]:
    if isinstance(value, Mapping):
        for key, item in value.items():
            yield str(key), item
            yield from _walk_payload(item)
    elif isinstance(value, (list, tuple)):
        for item in value:
            yield from _walk_payload(item)


def _is_redacted_value(value: Any) -> bool:
    if isinstance(value, str):
        return value.startswith("[redacted:")
    if isinstance(value, (list, tuple)):
        return all(_is_redacted_value(item) for item in value)
    return False
