from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from typing import Any

from relationship_lifelog_agent.adapters.types import NoteEvidence, ThoughtEvidence
from relationship_lifelog_agent.llm.local_client import LocalLlmClient
from relationship_lifelog_agent.llm.prompts import SYSTEM_POLICY
from relationship_lifelog_agent.llm.schemas import NOTE_WINDOW_ANALYSIS_SCHEMA
from relationship_lifelog_agent.llm.usage import LlmUsageTrace
from relationship_lifelog_agent.profiles import ProfileContext


PROMPT_VERSION = "note-window-v1"
MAX_NOTES_PER_WINDOW = 12
MAX_NOTE_EXCERPT_CHARS = 120
UNRELATED_TERMS = (
    "GEN ROCK",
    "WS",
    "ワークショップ",
    "履修",
    "財団",
    "ES",
    "OMG",
    "飲み会",
    "研究",
    "技術",
    "コード",
    "論文",
    "学業",
)


@dataclass(frozen=True)
class NoteWindowAnalysis:
    source_id: str
    date: str
    relationship_relevance: float
    date_relevance: float
    own_thought_likelihood: float
    lyrics_or_quote_likelihood: float
    conflict_related: bool
    summary: str
    confidence: float
    include_as_supporting_evidence: bool
    cautions: tuple[str, ...]
    backend: str = "rule"


def analyze_note_window_with_llm(
    notes: list[NoteEvidence | ThoughtEvidence],
    date_window: tuple[str, str],
    target_profile: ProfileContext | None,
    llm_client: LocalLlmClient | None,
    *,
    mode: str = "private",
    usage: LlmUsageTrace | None = None,
) -> list[NoteWindowAnalysis]:
    safe_notes = [
        item
        for item in (_safe_note(note, date_window=date_window, profile=target_profile, mode=mode) for note in notes)
        if item is not None
    ][:MAX_NOTES_PER_WINDOW]
    analyses: list[NoteWindowAnalysis] = []
    for safe_note in safe_notes:
        if llm_client and llm_client.is_configured():
            result = llm_client.chat(
                [
                    {"role": "system", "content": SYSTEM_POLICY},
                    {
                        "role": "user",
                        "content": str(
                            {
                                "task": "Score one short note window for relationship relevance.",
                                "date_window": {"date_from": date_window[0], "date_to": date_window[1]},
                                "note": safe_note,
                                "rules": [
                                    "Exclude unrelated school, research, tech, dance workshop, drinking party apology, or grant notes.",
                                    "Do not treat lyrics, quotes, or fiction as the user's actual feeling without caution.",
                                    "Do not infer the other person's inner feelings.",
                                    "Use only the short excerpt and metadata provided.",
                                ],
                            }
                        ),
                    },
                ],
                schema=NOTE_WINDOW_ANALYSIS_SCHEMA,
                expect_json=True,
            )
            if result.ok and isinstance(result.data, dict):
                if usage:
                    usage.record(
                        stage="note_window_analysis",
                        backend="llm",
                        success=True,
                        latency_ms=result.latency_ms,
                        structured_output_used=result.structured_output_used,
                        schema_validation_success=True,
                    )
                analyses.append(_note_window_from_data(safe_note, result.data))
                continue
            if usage:
                usage.record(
                    stage="note_window_analysis",
                    backend="rule",
                    success=False,
                    latency_ms=result.latency_ms,
                    structured_output_used=result.structured_output_used,
                    fallback_reason=result.error or "invalid llm note window analysis",
                )
        analyses.append(_rule_note_window(safe_note))
    return analyses


def _safe_note(
    item: NoteEvidence | ThoughtEvidence,
    *,
    date_window: tuple[str, str],
    profile: ProfileContext | None,
    mode: str,
) -> dict[str, object] | None:
    if _unknown_or_sentinel(item.date):
        return None
    if not (date_window[0] <= item.date <= date_window[1]):
        return None
    text = _redact_private_markers(f"{item.summary} {item.excerpt or ''} {' '.join(str(value) for value in item.metadata.values())}")
    if any(term in text for term in UNRELATED_TERMS):
        return None
    excerpt = None if mode == "public" else _short(_redact_private_markers(item.excerpt or item.summary), MAX_NOTE_EXCERPT_CHARS)
    return {
        "source_id": item.source_id,
        "date": item.date,
        "summary": _short(item.summary, 140),
        "excerpt": excerpt,
        "profile_name_mentioned": bool(profile and profile.profile_name and profile.profile_name in text),
        "category": getattr(item, "category", getattr(item, "thought_type", "")),
    }


def _note_window_from_data(note: dict[str, object], data: dict[str, Any]) -> NoteWindowAnalysis:
    return NoteWindowAnalysis(
        source_id=str(note["source_id"]),
        date=str(note["date"]),
        relationship_relevance=_score(data.get("relationship_relevance")),
        date_relevance=_score(data.get("date_relevance")),
        own_thought_likelihood=_score(data.get("own_thought_likelihood")),
        lyrics_or_quote_likelihood=_score(data.get("lyrics_or_quote_likelihood")),
        conflict_related=bool(data.get("conflict_related")),
        summary=_short(str(data.get("summary") or note.get("summary") or ""), 180),
        confidence=_score(data.get("confidence")),
        include_as_supporting_evidence=bool(data.get("include_as_supporting_evidence")),
        cautions=tuple(_short(str(item), 160) for item in data.get("cautions", []) if isinstance(item, str)),
        backend="llm",
    )


def _rule_note_window(note: dict[str, object]) -> NoteWindowAnalysis:
    text = f"{note.get('summary', '')} {note.get('excerpt') or ''} {note.get('category', '')}"
    relationship_relevance = 0.85 if note.get("profile_name_mentioned") else 0.65 if _has_any(text, "関係", "喧嘩", "すれ違い", "反省", "不安", "謝罪") else 0.25
    conflict_related = relationship_relevance >= 0.6 and _has_any(text, "喧嘩", "すれ違い", "反省", "不安", "謝罪")
    include = relationship_relevance >= 0.5 and conflict_related
    return NoteWindowAnalysis(
        source_id=str(note["source_id"]),
        date=str(note["date"]),
        relationship_relevance=relationship_relevance,
        date_relevance=1.0,
        own_thought_likelihood=0.7,
        lyrics_or_quote_likelihood=0.1,
        conflict_related=conflict_related,
        summary=_short(str(note.get("summary") or ""), 180),
        confidence=0.55 if include else 0.3,
        include_as_supporting_evidence=include,
        cautions=("noteは自分側の記録候補であり、相手の内面は断定しません。",),
        backend="rule",
    )


def _unknown_or_sentinel(value: str) -> bool:
    try:
        return date.fromisoformat(str(value)[:10]) <= date(1970, 1, 1)
    except (TypeError, ValueError):
        return True


def _score(value: object) -> float:
    try:
        return max(0.0, min(1.0, float(value)))
    except (TypeError, ValueError):
        return 0.5


def _has_any(text: str, *needles: str) -> bool:
    return any(needle in text for needle in needles)


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
