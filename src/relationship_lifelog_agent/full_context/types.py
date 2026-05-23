from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


JsonDict = dict[str, Any]


@dataclass(frozen=True)
class FullLineItem:
    source_ref: str
    source_id: str
    conversation_id: str | None
    timestamp: str | None
    speaker_id: str | None
    speaker_label: str | None
    speaker_role: str = "unknown"
    raw_text: str | None = None
    message_type: str = "text"
    reply_to_id: str | None = None
    attachments: tuple[str, ...] = ()
    raw_metadata: JsonDict = field(default_factory=dict)


@dataclass(frozen=True)
class FullNoteItem:
    source_ref: str
    source_id: str
    note_id: str
    title: str | None = None
    created_at: str | None = None
    modified_at: str | None = None
    raw_body: str | None = None
    source_path: str | None = None
    tags: tuple[str, ...] = ()
    raw_metadata: JsonDict = field(default_factory=dict)


@dataclass(frozen=True)
class FullMediaItem:
    source_ref: str
    source_id: str
    media_id: str
    file_path: str | None = None
    thumbnail_path: str | None = None
    captured_at: str | None = None
    exact_gps: JsonDict | None = None
    place_label: str | None = None
    vlm_caption: str | None = None
    ocr_text: str | None = None
    detected_people: tuple[JsonDict, ...] = ()
    verified_people: tuple[JsonDict, ...] = ()
    unverified_person_candidates: tuple[JsonDict, ...] = ()
    face_crop_paths: tuple[str, ...] = ()
    face_embeddings: tuple[tuple[float, ...], ...] = ()
    raw_metadata: JsonDict = field(default_factory=dict)


@dataclass(frozen=True)
class FullFaceItem:
    source_ref: str
    source_id: str
    face_id: str
    media_id: str | None = None
    person_candidate_id: str | None = None
    verified_person_id: str | None = None
    crop_path: str | None = None
    embedding: tuple[float, ...] | None = None
    bbox: JsonDict | None = None
    confidence: float | None = None
    raw_metadata: JsonDict = field(default_factory=dict)


@dataclass(frozen=True)
class FullLocationItem:
    source_ref: str
    source_id: str
    timestamp: str | None = None
    latitude: float | None = None
    longitude: float | None = None
    accuracy: float | None = None
    place_label: str | None = None
    raw_metadata: JsonDict = field(default_factory=dict)


@dataclass(frozen=True)
class FullDataManifest:
    run_id: str
    profile_id: int | None = None
    date_from: str | None = None
    date_to: str | None = None
    line_count: int = 0
    note_count: int = 0
    media_count: int = 0
    face_count: int = 0
    location_count: int = 0
    date_coverage: JsonDict = field(default_factory=dict)
    source_counts_by_day: JsonDict = field(default_factory=dict)
    available_payload_types: tuple[str, ...] = ()
    estimated_chars: int = 0
    estimated_tokens: int = 0
    can_fit_single_context: bool = False
    recommended_mode: str = "safe_window"
    raw_payload_policy: JsonDict = field(default_factory=dict)


@dataclass(frozen=True)
class FullPromptBundle:
    bundle_id: str
    analysis_mode: str
    prompt_text: str
    source_refs: tuple[str, ...]
    estimated_chars: int = 0
    estimated_tokens: int = 0
    raw_payload_included: bool = False
    metadata: JsonDict = field(default_factory=dict)


@dataclass(frozen=True)
class FullScanBatch:
    batch_id: str
    batch_index: int
    date_from: str | None = None
    date_to: str | None = None
    source_types: tuple[str, ...] = ()
    line_items: tuple[FullLineItem, ...] = ()
    note_items: tuple[FullNoteItem, ...] = ()
    media_items: tuple[FullMediaItem, ...] = ()
    face_items: tuple[FullFaceItem, ...] = ()
    location_items: tuple[FullLocationItem, ...] = ()
    estimated_tokens: int = 0
    coverage_summary: str = ""
    source_refs: tuple[str, ...] = ()


@dataclass(frozen=True)
class FullBatchAnalysis:
    batch_id: str
    key_findings: tuple[str, ...] = ()
    candidate_events: tuple[JsonDict, ...] = ()
    relevant_evidence_refs: tuple[str, ...] = ()
    uncertainties: tuple[str, ...] = ()
    suggested_next_steps: tuple[str, ...] = ()
    confidence: float = 0.5
    result_json: JsonDict = field(default_factory=dict)


@dataclass(frozen=True)
class FullRangeSynthesis:
    summary: str
    answer: str
    timeline: tuple[JsonDict, ...] = ()
    evidence: tuple[JsonDict, ...] = ()
    uncertainties: tuple[str, ...] = ()
    followup_suggestions: tuple[str, ...] = ()
    source_refs: tuple[str, ...] = ()
    confidence: float = 0.5
    result_json: JsonDict = field(default_factory=dict)
