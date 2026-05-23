from __future__ import annotations

from dataclasses import asdict, dataclass, field, is_dataclass
from datetime import date
import re
from typing import Any, Iterable

from relationship_lifelog_agent.privacy.guard import detect_answer_safety_violations, sanitize_answer


SOURCE_REF_RE = re.compile(
    r"(?i)\b(?:source_ref|source_refs|source|ref)\s*[:=]\s*[`'\"]?([A-Za-z0-9_.:/#-]+)"
)
ISO_DATE_RE = re.compile(r"\b(20\d{2}-\d{2}-\d{2})\b")
RAW_LEAK_MARKERS = (
    "LINE全文",
    "line_full_text",
    "raw LINE",
    "メモ全文",
    "note_full_text",
    "raw note",
    "face_crop",
    "顔crop",
    "embedding:",
    "private path",
    "/home/",
    "~/",
)
UNSUPPORTED_ASSERTION_PATTERNS = (
    re.compile(r"確実に[^。\n]*(?:喧嘩|本人|恋人|親密|怒って|冷めて)[^。\n]*"),
    re.compile(r"(?:相手|この人)[^。\n]*(?:怒っていた|冷めていた|愛情がなかった|恋人です|親密な関係です)[^。\n]*"),
    re.compile(r"(?:person|人物|LINE speaker|speaker)[^。\n]*(?:自動リンク|自動で紐づけ|同一人物と判断)[^。\n]*"),
    re.compile(r"(?:断定できます|断定しました|確定です)"),
)


@dataclass(frozen=True)
class VerificationIssue:
    code: str
    severity: str
    message: str
    evidence: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class VerificationReport:
    ok: bool
    issues: tuple[VerificationIssue, ...] = ()
    cautions: tuple[str, ...] = ()
    corrected_answer: str | None = None
    fallback_used: bool = False

    @property
    def major_issues(self) -> tuple[VerificationIssue, ...]:
        return tuple(issue for issue in self.issues if issue.severity in {"major", "critical"})

    @property
    def minor_issues(self) -> tuple[VerificationIssue, ...]:
        return tuple(issue for issue in self.issues if issue.severity == "minor")

    def to_dict(self) -> dict[str, Any]:
        return {
            "ok": self.ok,
            "issues": [issue.to_dict() for issue in self.issues],
            "cautions": list(self.cautions),
            "corrected_answer": self.corrected_answer,
            "fallback_used": self.fallback_used,
        }

    def to_markdown(self) -> str:
        if not self.issues:
            return "- verification: ok"
        return "\n".join(f"- {issue.severity}: {issue.code} - {issue.message}" for issue in self.issues)


def verify_source_refs_exist(answer: str, manifest: Any = None, observations: Any = None) -> tuple[VerificationIssue, ...]:
    used_refs = _extract_source_refs(answer)
    if not used_refs:
        return ()
    allowed_refs = _collect_allowed_source_refs(manifest, observations)
    if not allowed_refs:
        return tuple(
            VerificationIssue(
                code="unverifiable_source_ref",
                severity="minor",
                message=f"source_ref {ref} could not be checked because no source_ref index is available.",
                evidence=ref,
            )
            for ref in used_refs
        )
    missing = tuple(ref for ref in used_refs if ref not in allowed_refs)
    return tuple(
        VerificationIssue(
            code="missing_source_ref",
            severity="major",
            message=f"source_ref {ref} is not present in the manifest or observations.",
            evidence=ref,
        )
        for ref in missing
    )


def verify_dates_within_scope(answer: str, date_from: str | None = None, date_to: str | None = None) -> tuple[VerificationIssue, ...]:
    if not date_from and not date_to:
        return ()
    start = _parse_date(date_from) if date_from else None
    end = _parse_date(date_to) if date_to else None
    issues: list[VerificationIssue] = []
    for value in ISO_DATE_RE.findall(answer):
        current = _parse_date(value)
        if current is None:
            continue
        if start and current < start or end and current > end:
            issues.append(
                VerificationIssue(
                    code="date_out_of_scope",
                    severity="major",
                    message=f"date {value} is outside the requested scope.",
                    evidence=value,
                )
            )
    return tuple(issues)


def verify_claims_supported(answer: str, evidence_refs: Iterable[str] = ()) -> tuple[VerificationIssue, ...]:
    refs = tuple(ref for ref in evidence_refs if ref)
    strong_claims = _find_unsupported_assertions(answer)
    if refs or not strong_claims:
        return ()
    return tuple(
        VerificationIssue(
            code="unsupported_claim",
            severity="major",
            message="A strong relationship or identity claim has no evidence source_ref.",
            evidence=claim,
        )
        for claim in strong_claims
    )


def verify_counts_against_python_facts(answer: str, computed_facts: dict[str, Any] | None = None) -> tuple[VerificationIssue, ...]:
    if not computed_facts:
        return ()
    issues: list[VerificationIssue] = []
    for key, expected in computed_facts.items():
        if not isinstance(expected, int):
            continue
        actual = _extract_named_count(answer, key)
        if actual is None:
            continue
        if actual != expected:
            issues.append(
                VerificationIssue(
                    code="count_mismatch",
                    severity="major",
                    message=f"{key} is {actual} in the answer, but Python computed {expected}.",
                    evidence=f"{key}: {actual}",
                )
            )
    return tuple(issues)


def verify_no_unsupported_relationship_assertions(answer: str) -> tuple[VerificationIssue, ...]:
    issues = [
        VerificationIssue(
            code="unsupported_relationship_assertion",
            severity="critical",
            message="Unsupported relationship, identity, or inner-feeling assertion detected.",
            evidence=claim,
        )
        for claim in _find_unsupported_assertions(answer)
    ]
    return tuple(issues)


def verify_no_public_leaks(answer: str, mode: str = "private") -> tuple[VerificationIssue, ...]:
    issues: list[VerificationIssue] = []
    if mode == "public":
        issues.extend(
            VerificationIssue(
                code=violation.code,
                severity="critical",
                message="Public mode output contains private or unsafe data.",
                evidence=violation.text,
            )
            for violation in detect_answer_safety_violations(answer, mode="public")
        )
    for marker in RAW_LEAK_MARKERS:
        if marker in answer and (mode == "public" or marker in {"/home/", "~/", "face_crop", "顔crop", "embedding:"}):
            issues.append(
                VerificationIssue(
                    code="raw_data_leak",
                    severity="critical" if mode == "public" else "major",
                    message=f"Answer contains raw/private marker: {marker}",
                    evidence=marker,
                )
            )
    return tuple(dict.fromkeys(issues))


def verify_final_answer(
    answer: str,
    *,
    manifest: Any = None,
    observations: Any = None,
    date_from: str | None = None,
    date_to: str | None = None,
    evidence_refs: Iterable[str] = (),
    computed_facts: dict[str, Any] | None = None,
    mode: str = "private",
) -> VerificationReport:
    issues = (
        *verify_source_refs_exist(answer, manifest, observations),
        *verify_dates_within_scope(answer, date_from, date_to),
        *verify_claims_supported(answer, evidence_refs),
        *verify_counts_against_python_facts(answer, computed_facts),
        *verify_no_unsupported_relationship_assertions(answer),
        *verify_no_public_leaks(answer, mode),
    )
    issues = _dedupe_issues(issues)
    corrected = sanitize_answer(answer, mode=mode)
    fallback_used = False
    if any(issue.severity in {"major", "critical"} for issue in issues):
        corrected = _append_verification_caution(corrected, issues)
        fallback_used = True
    else:
        corrected = _append_verification_caution(corrected, issues)
    return VerificationReport(
        ok=not any(issue.severity in {"major", "critical"} for issue in issues),
        issues=issues,
        cautions=tuple(issue.message for issue in issues),
        corrected_answer=corrected,
        fallback_used=fallback_used,
    )


def _extract_source_refs(answer: str) -> tuple[str, ...]:
    refs = list(SOURCE_REF_RE.findall(answer))
    refs.extend(re.findall(r"\b(?:plr|line|note|media|photo|gps|face|manual):[A-Za-z0-9_.:/#-]+", answer))
    return tuple(dict.fromkeys(ref.strip("`'\".,)") for ref in refs if ref))


def _collect_allowed_source_refs(manifest: Any, observations: Any) -> set[str]:
    refs: set[str] = set()
    refs.update(_refs_from_obj(manifest))
    refs.update(_refs_from_obj(observations))
    return refs


def _refs_from_obj(value: Any) -> set[str]:
    if value is None:
        return set()
    if hasattr(value, "observations"):
        return _refs_from_obj(getattr(value, "observations"))
    if is_dataclass(value):
        return _refs_from_obj(asdict(value))
    if isinstance(value, dict):
        refs: set[str] = set()
        for key, item in value.items():
            if key in {"source_ref", "source_refs", "relevant_evidence_refs", "evidence_refs"}:
                refs.update(_coerce_refs(item))
            refs.update(_refs_from_obj(item))
        return refs
    if isinstance(value, (list, tuple, set)):
        refs: set[str] = set()
        for item in value:
            refs.update(_refs_from_obj(item))
        return refs
    return set()


def _coerce_refs(value: Any) -> set[str]:
    if isinstance(value, str):
        return {value}
    if isinstance(value, (list, tuple, set)):
        return {str(item) for item in value if item}
    return set()


def _find_unsupported_assertions(answer: str) -> tuple[str, ...]:
    claims: list[str] = []
    for pattern in UNSUPPORTED_ASSERTION_PATTERNS:
        claims.extend(match.group(0) for match in pattern.finditer(answer))
    return tuple(dict.fromkeys(claim.strip() for claim in claims if claim.strip()))


def _extract_named_count(answer: str, key: str) -> int | None:
    patterns = (
        rf"(?im)^\s*-?\s*{re.escape(key)}\s*[:=]\s*(\d+)\b",
        rf"(?im)\b{re.escape(key)}\s*(?:is|は)?\s*(\d+)\b",
    )
    for pattern in patterns:
        match = re.search(pattern, answer)
        if match:
            return int(match.group(1))
    return None


def _append_verification_caution(answer: str, issues: tuple[VerificationIssue, ...]) -> str:
    if "Verification:" in answer or "検証:" in answer:
        return answer
    body = (
        ["- verification: ok"]
        if not issues
        else [f"- {issue.severity}: {issue.code} - {issue.message}" for issue in issues]
    )
    lines = [
        "",
        "検証:",
        "<details>",
        "<summary>Full-context answer verification</summary>",
        "",
        *body,
        "",
        "</details>",
    ]
    return answer.rstrip() + "\n" + "\n".join(lines)


def _dedupe_issues(issues: Iterable[VerificationIssue]) -> tuple[VerificationIssue, ...]:
    seen: set[tuple[str, str, str | None]] = set()
    unique: list[VerificationIssue] = []
    for issue in issues:
        key = (issue.code, issue.message, issue.evidence)
        if key in seen:
            continue
        seen.add(key)
        unique.append(issue)
    return tuple(unique)


def _parse_date(value: str | None) -> date | None:
    if not value:
        return None
    try:
        return date.fromisoformat(value)
    except ValueError:
        return None
