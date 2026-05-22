from __future__ import annotations

from dataclasses import replace
import re

from relationship_lifelog_agent.adapters.types import EvidenceItem
from relationship_lifelog_agent.privacy.policies import (
    PUBLIC_ANONYMOUS_PERSON_LABELS,
    PUBLIC_RELATIONSHIP_LABELS,
)


GPS_RE = re.compile(r"\b-?\d{1,3}\.\d{4,}\s*,\s*-?\d{1,3}\.\d{4,}\b")
SOURCE_PATH_RE = re.compile(r"(?i)\b(source[_\s-]*file[_\s-]*path|source[_\s-]*path)\s*[:=]\s*\S+")
PRIVATE_PATH_RE = re.compile(r"(?i)(/home/[^\s)>'\"]+|~/[^\s)>'\"]+)")
LINE_FULL_TEXT_RE = re.compile(r"(?im)\b(LINE全文|LINE full text|line_full_text)\s*[:=]\s*.+$")
NOTE_FULL_TEXT_RE = re.compile(r"(?im)\b(メモ全文|ノート全文|note full text|note_full_text)\s*[:=]\s*.+$")
FACE_CROP_RE = re.compile(r"(?i)(face[_\s-]*crop|顔\s*crop|face_crop_path)\s*[:=]?\s*\S*")
PHOTO_PATH_RE = re.compile(r"(?i)\S+\.(?:jpg|jpeg|png|heic|webp)\b")


def redact_public_text(text: str, person_names: list[str] | None = None) -> str:
    redacted = _redact_person_names(text, person_names or [])
    redacted = LINE_FULL_TEXT_RE.sub("[LINE full text redacted]", redacted)
    redacted = NOTE_FULL_TEXT_RE.sub("[note full text redacted]", redacted)
    redacted = SOURCE_PATH_RE.sub("[source file path redacted]", redacted)
    redacted = FACE_CROP_RE.sub("[face data redacted]", redacted)
    redacted = PHOTO_PATH_RE.sub("[real photo redacted]", redacted)
    redacted = GPS_RE.sub("[exact GPS redacted]", redacted)
    redacted = PRIVATE_PATH_RE.sub("[private path redacted]", redacted)
    for label in PUBLIC_RELATIONSHIP_LABELS:
        redacted = re.sub(re.escape(label), "[relationship label redacted]", redacted, flags=re.IGNORECASE)
    return redacted


def redact_evidence_for_mode(
    item: EvidenceItem,
    mode: str = "private",
    person_names: list[str] | None = None,
) -> EvidenceItem:
    if mode != "public":
        return item
    summary = redact_public_text(item.summary, person_names=person_names)
    return replace(item, summary=summary, excerpt=None)


def _redact_person_names(text: str, person_names: list[str]) -> str:
    redacted = text
    for index, name in enumerate(person_names):
        if not name:
            continue
        label = PUBLIC_ANONYMOUS_PERSON_LABELS[min(index, len(PUBLIC_ANONYMOUS_PERSON_LABELS) - 1)]
        redacted = redacted.replace(name, label)
    return redacted
