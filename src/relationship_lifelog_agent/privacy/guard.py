from __future__ import annotations

from dataclasses import dataclass
import re

from relationship_lifelog_agent.privacy.policies import (
    FORBIDDEN_PHRASES,
    FORBIDDEN_REPLACEMENTS,
    INNER_FEELING_REPLACEMENT,
    PUBLIC_RELATIONSHIP_LABELS,
)
from relationship_lifelog_agent.privacy.redaction import (
    FACE_CROP_RE,
    GPS_RE,
    LINE_FULL_TEXT_RE,
    NOTE_FULL_TEXT_RE,
    PHOTO_PATH_RE,
    PRIVATE_PATH_RE,
    SOURCE_PATH_RE,
    redact_public_text,
)


INNER_FEELING_ASSERTION_RE = re.compile(
    r"相手は[^。\n]*(?:冷めて|怒って|愛情がなかった|悲しんで|嬉し|嫌がって|不安だ|考えていた|思っていた|感じていた)[^。\n]*"
)


@dataclass(frozen=True)
class SafetyViolation:
    code: str
    text: str


def detect_forbidden_phrases(text: str) -> list[str]:
    return [phrase for phrase in FORBIDDEN_PHRASES if phrase in text]


def detect_answer_safety_violations(
    text: str,
    mode: str = "private",
    person_names: list[str] | None = None,
) -> list[SafetyViolation]:
    violations = [
        SafetyViolation(code="forbidden_phrase", text=phrase)
        for phrase in detect_forbidden_phrases(text)
    ]
    violations.extend(
        SafetyViolation(code="inner_feeling_assertion", text=match.group(0))
        for match in INNER_FEELING_ASSERTION_RE.finditer(text)
    )
    if mode == "public":
        violations.extend(_detect_public_mode_violations(text, person_names=person_names or []))
    return violations


def sanitize_answer(text: str, mode: str = "private", person_names: list[str] | None = None) -> str:
    safe = text
    for phrase, replacement in FORBIDDEN_REPLACEMENTS.items():
        safe = safe.replace(phrase, replacement)
    safe = INNER_FEELING_ASSERTION_RE.sub(INNER_FEELING_REPLACEMENT, safe)
    if mode == "public":
        safe = redact_public_text(safe, person_names=person_names)
    return safe


def assert_answer_safe(
    text: str,
    mode: str = "private",
    person_names: list[str] | None = None,
) -> None:
    violations = detect_answer_safety_violations(text, mode=mode, person_names=person_names)
    if violations:
        details = ", ".join(f"{item.code}: {item.text}" for item in violations)
        raise ValueError(f"Unsafe answer text: {details}")


def _detect_public_mode_violations(text: str, person_names: list[str]) -> list[SafetyViolation]:
    violations: list[SafetyViolation] = []
    for name in person_names:
        if name and name in text:
            violations.append(SafetyViolation(code="public_person_name", text=name))
    for label in PUBLIC_RELATIONSHIP_LABELS:
        if re.search(re.escape(label), text, flags=re.IGNORECASE):
            violations.append(SafetyViolation(code="public_relationship_label", text=label))
    pattern_checks = (
        ("public_line_full_text", LINE_FULL_TEXT_RE),
        ("public_note_full_text", NOTE_FULL_TEXT_RE),
        ("public_exact_gps", GPS_RE),
        ("public_face_crop", FACE_CROP_RE),
        ("public_real_photo", PHOTO_PATH_RE),
        ("public_source_file_path", SOURCE_PATH_RE),
        ("public_private_path", PRIVATE_PATH_RE),
    )
    for code, pattern in pattern_checks:
        violations.extend(SafetyViolation(code=code, text=match.group(0)) for match in pattern.finditer(text))
    return violations
