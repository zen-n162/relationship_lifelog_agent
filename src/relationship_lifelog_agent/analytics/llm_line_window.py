from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from relationship_lifelog_agent.adapters.types import LineEvidence
from relationship_lifelog_agent.llm.local_client import LocalLlmClient
from relationship_lifelog_agent.llm.prompts import SYSTEM_POLICY
from relationship_lifelog_agent.llm.schemas import LINE_WINDOW_ANALYSIS_SCHEMA
from relationship_lifelog_agent.llm.usage import LlmUsageTrace
from relationship_lifelog_agent.profiles import ProfileContext


PROMPT_VERSION = "line-window-v1"
MAX_MESSAGES_PER_WINDOW = 30
MAX_EXCERPT_CHARS = 80


@dataclass(frozen=True)
class LineWindowAnalysis:
    is_conflict_candidate: bool
    is_minor_misunderstanding: bool
    severity: int
    confidence: float
    estimated_start_date: str | None
    estimated_end_date: str | None
    summary: str
    evidence_message_refs: tuple[str, ...]
    evidence_excerpt_short: tuple[str, ...]
    is_joke_or_general_statement_possible: bool
    cautions: tuple[str, ...]
    backend: str = "rule"


def analyze_line_window_with_llm(
    profile: ProfileContext | None,
    messages: list[LineEvidence],
    date_window: tuple[str, str],
    llm_client: LocalLlmClient | None,
    *,
    mode: str = "private",
    usage: LlmUsageTrace | None = None,
) -> LineWindowAnalysis:
    safe_messages = [_safe_message(item, profile=profile, mode=mode) for item in messages[:MAX_MESSAGES_PER_WINDOW]]
    if not safe_messages:
        return _rule_line_window([], date_window, backend="rule")
    if llm_client and llm_client.is_configured():
        prompt = {
            "task": "Analyze a small, pre-filtered LINE window for conflict or minor misunderstanding candidates.",
            "date_window": {"date_from": date_window[0], "date_to": date_window[1]},
            "messages": safe_messages,
            "rules": [
                "Do not infer relationship labels.",
                "Do not infer speaker identity; speaker_role is provided by Python.",
                "Do not assert the other person's inner feelings.",
                "Distinguish conflict_candidate from minor_misunderstanding.",
                "Treat jokes or general statements cautiously.",
                "Use only the given short excerpts and source refs.",
            ],
        }
        result = llm_client.chat(
            [
                {"role": "system", "content": SYSTEM_POLICY},
                {"role": "user", "content": str(prompt)},
            ],
            schema=LINE_WINDOW_ANALYSIS_SCHEMA,
            expect_json=True,
        )
        if result.ok and isinstance(result.data, dict):
            if usage:
                usage.record(
                    stage="line_window_analysis",
                    backend="llm",
                    success=True,
                    latency_ms=result.latency_ms,
                    structured_output_used=result.structured_output_used,
                    schema_validation_success=True,
                )
            return _line_window_from_data(result.data)
        if usage:
            usage.record(
                stage="line_window_analysis",
                backend="rule",
                success=False,
                latency_ms=result.latency_ms,
                structured_output_used=result.structured_output_used,
                fallback_reason=result.error or "invalid llm line window analysis",
            )
    return _rule_line_window(safe_messages, date_window, backend="rule")


def _safe_message(item: LineEvidence, *, profile: ProfileContext | None, mode: str) -> dict[str, object]:
    role = "other"
    if profile and item.speaker_id:
        if item.speaker_id == profile.self_line_speaker_filter_source_id:
            role = "self"
        elif item.speaker_id == profile.target_line_speaker_source_id:
            role = "target"
    excerpt = None if mode == "public" else _short(_redact_private_markers(item.excerpt or item.summary), MAX_EXCERPT_CHARS)
    return {
        "ref": item.source_pointer or f"line_message:{item.source_id}",
        "date": item.date,
        "speaker_role": role,
        "summary": _short(_redact_private_markers(item.summary), 120),
        "excerpt": excerpt,
        "confidence": round(item.confidence, 3),
    }


def _line_window_from_data(data: dict[str, Any]) -> LineWindowAnalysis:
    return LineWindowAnalysis(
        is_conflict_candidate=bool(data.get("is_conflict_candidate")),
        is_minor_misunderstanding=bool(data.get("is_minor_misunderstanding")),
        severity=max(0, min(3, int(data.get("severity", 0)))),
        confidence=max(0.0, min(1.0, float(data.get("confidence", 0.5)))),
        estimated_start_date=_optional_str(data.get("estimated_start_date")),
        estimated_end_date=_optional_str(data.get("estimated_end_date")),
        summary=_short(str(data.get("summary") or "LINE window analysis."), 180),
        evidence_message_refs=tuple(str(item)[:120] for item in data.get("evidence_message_refs", []) if isinstance(item, str)),
        evidence_excerpt_short=tuple(_short(str(item), MAX_EXCERPT_CHARS) for item in data.get("evidence_excerpt_short", []) if isinstance(item, str)),
        is_joke_or_general_statement_possible=bool(data.get("is_joke_or_general_statement_possible")),
        cautions=tuple(_short(str(item), 160) for item in data.get("cautions", []) if isinstance(item, str)),
        backend="llm",
    )


def _rule_line_window(messages: list[dict[str, object]], date_window: tuple[str, str], *, backend: str) -> LineWindowAnalysis:
    haystack = " ".join(str(item.get("summary", "")) + " " + str(item.get("excerpt") or "") for item in messages)
    has_conflict = any(token in haystack for token in ("喧嘩", "謝罪", "不満", "予定変更"))
    has_minor = any(token in haystack for token in ("すれ違い", "返信", "予定確認"))
    severity = 2 if has_conflict else 1 if has_minor else 0
    return LineWindowAnalysis(
        is_conflict_candidate=has_conflict,
        is_minor_misunderstanding=has_minor and not has_conflict,
        severity=severity,
        confidence=0.55 if (has_conflict or has_minor) else 0.25,
        estimated_start_date=date_window[0],
        estimated_end_date=date_window[1],
        summary="短いLINE windowから喧嘩・すれ違い候補を安全に確認しました。",
        evidence_message_refs=tuple(str(item.get("ref")) for item in messages[:5] if item.get("ref")),
        evidence_excerpt_short=tuple(str(item.get("excerpt")) for item in messages[:3] if item.get("excerpt")),
        is_joke_or_general_statement_possible=False,
        cautions=("LINE windowのみでは喧嘩とは断定できません。",),
        backend=backend,
    )


def _optional_str(value: object) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text or text.lower() == "null":
        return None
    return text[:20]


def _short(value: str, limit: int) -> str:
    text = " ".join(str(value or "").split())
    if len(text) <= limit:
        return text
    return text[: limit - 3] + "..."


def _redact_private_markers(value: str) -> str:
    parts: list[str] = []
    for token in str(value or "").split():
        if "/" in token or "\\" in token:
            parts.append("[redacted_path]")
        else:
            parts.append(token)
    return " ".join(parts)
