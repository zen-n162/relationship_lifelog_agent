from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from relationship_lifelog_agent.config import Settings


PRIVATE_FULL_MODES = frozenset({"private_full_range", "private_full_corpus"})


@dataclass(frozen=True)
class RawPayloadPolicy:
    analysis_mode: str = "safe_window"
    allow_raw_line_text: bool = False
    allow_raw_note_text: bool = False
    allow_photo_paths: bool = False
    allow_exact_gps: bool = False
    allow_face_crops: bool = False
    allow_face_embeddings: bool = False
    allow_raw_face_embedding_values: bool = False
    allow_private_file_paths: bool = False
    allow_unverified_person_candidates: bool = False
    allow_unverified_speaker_candidates: bool = False
    allow_full_prompt_logging: bool = False
    allow_raw_payload_cache: bool = False

    @property
    def private_full_enabled(self) -> bool:
        return self.analysis_mode in PRIVATE_FULL_MODES

    def should_include_raw_line_text(self) -> bool:
        return self.private_full_enabled and self.allow_raw_line_text

    def should_include_raw_note_text(self) -> bool:
        return self.private_full_enabled and self.allow_raw_note_text

    def should_include_photo_paths(self) -> bool:
        return self.private_full_enabled and self.allow_photo_paths

    def should_include_exact_gps(self) -> bool:
        return self.private_full_enabled and self.allow_exact_gps

    def should_include_face_crops(self) -> bool:
        return self.private_full_enabled and self.allow_face_crops

    def should_include_face_embeddings(self) -> bool:
        return self.private_full_enabled and self.allow_face_embeddings

    def should_include_private_file_paths(self) -> bool:
        return self.private_full_enabled and self.allow_private_file_paths

    def should_include_unverified_person_candidates(self) -> bool:
        return self.private_full_enabled and self.allow_unverified_person_candidates

    def should_include_unverified_speaker_candidates(self) -> bool:
        return self.private_full_enabled and self.allow_unverified_speaker_candidates

    def to_dict(self) -> dict[str, Any]:
        return {
            "analysis_mode": self.analysis_mode,
            "private_full_enabled": self.private_full_enabled,
            "allow_raw_line_text": self.should_include_raw_line_text(),
            "allow_raw_note_text": self.should_include_raw_note_text(),
            "allow_photo_paths": self.should_include_photo_paths(),
            "allow_exact_gps": self.should_include_exact_gps(),
            "allow_face_crops": self.should_include_face_crops(),
            "allow_face_embeddings": self.should_include_face_embeddings(),
            "allow_raw_face_embedding_values": self.private_full_enabled and self.allow_raw_face_embedding_values,
            "allow_private_file_paths": self.should_include_private_file_paths(),
            "allow_unverified_person_candidates": self.should_include_unverified_person_candidates(),
            "allow_unverified_speaker_candidates": self.should_include_unverified_speaker_candidates(),
            "allow_full_prompt_logging": self.private_full_enabled and self.allow_full_prompt_logging,
            "allow_raw_payload_cache": self.private_full_enabled and self.allow_raw_payload_cache,
        }


def from_config(settings: Settings) -> RawPayloadPolicy:
    payload = settings.private_full_llm_payload
    return RawPayloadPolicy(
        analysis_mode=settings.analysis.mode,
        allow_raw_line_text=payload.allow_raw_line_text,
        allow_raw_note_text=payload.allow_raw_note_text,
        allow_photo_paths=payload.allow_photo_paths,
        allow_exact_gps=payload.allow_exact_gps,
        allow_face_crops=payload.allow_face_crops,
        allow_face_embeddings=payload.allow_face_embeddings,
        allow_raw_face_embedding_values=payload.allow_raw_face_embedding_values,
        allow_private_file_paths=payload.allow_private_file_paths,
        allow_unverified_person_candidates=payload.allow_unverified_person_candidates,
        allow_unverified_speaker_candidates=payload.allow_unverified_speaker_candidates,
        allow_full_prompt_logging=payload.allow_full_prompt_logging,
        allow_raw_payload_cache=payload.allow_raw_payload_cache,
    )


def is_private_full_enabled(policy: RawPayloadPolicy | Settings) -> bool:
    return _coerce_policy(policy).private_full_enabled


def should_include_raw_line_text(policy: RawPayloadPolicy | Settings) -> bool:
    return _coerce_policy(policy).should_include_raw_line_text()


def should_include_raw_note_text(policy: RawPayloadPolicy | Settings) -> bool:
    return _coerce_policy(policy).should_include_raw_note_text()


def should_include_photo_paths(policy: RawPayloadPolicy | Settings) -> bool:
    return _coerce_policy(policy).should_include_photo_paths()


def should_include_exact_gps(policy: RawPayloadPolicy | Settings) -> bool:
    return _coerce_policy(policy).should_include_exact_gps()


def should_include_face_embeddings(policy: RawPayloadPolicy | Settings) -> bool:
    return _coerce_policy(policy).should_include_face_embeddings()


def _coerce_policy(value: RawPayloadPolicy | Settings) -> RawPayloadPolicy:
    if isinstance(value, RawPayloadPolicy):
        return value
    return from_config(value)
