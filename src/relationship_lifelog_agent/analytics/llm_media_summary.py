from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Any

from relationship_lifelog_agent.adapters.types import MediaEvidence
from relationship_lifelog_agent.llm.local_client import LocalLlmClient
from relationship_lifelog_agent.llm.prompts import SYSTEM_POLICY
from relationship_lifelog_agent.llm.schemas import MEDIA_DAY_SUMMARY_SCHEMA
from relationship_lifelog_agent.llm.usage import LlmUsageTrace
from relationship_lifelog_agent.profiles import ProfileContext


PROMPT_VERSION = "media-day-v1"
MAX_MEDIA_ITEMS_PER_DAY = 30
MAX_MEDIA_EXCERPT_CHARS = 80


@dataclass(frozen=True)
class MediaDaySummary:
    date: str
    has_media: bool
    likely_activities: tuple[str, ...]
    place_summaries: tuple[str, ...]
    people_roles_present: tuple[str, ...]
    summary: str
    confidence: float
    evidence_media_refs: tuple[str, ...]
    cautions: tuple[str, ...]
    backend: str = "rule"


def summarize_media_day_with_llm(
    day: str,
    media_items: list[MediaEvidence],
    profile: ProfileContext | None,
    llm_client: LocalLlmClient | None,
    *,
    mode: str = "private",
    usage: LlmUsageTrace | None = None,
) -> MediaDaySummary:
    safe_items = [_safe_media(item, profile=profile, mode=mode) for item in media_items[:MAX_MEDIA_ITEMS_PER_DAY]]
    if llm_client and llm_client.is_configured() and safe_items:
        result = llm_client.chat(
            [
                {"role": "system", "content": SYSTEM_POLICY},
                {
                    "role": "user",
                    "content": str(
                        {
                            "task": "Summarize one day's media metadata without image paths, exact GPS, faces, or raw files.",
                            "date": day,
                            "media_count": len(media_items),
                            "media_items": safe_items,
                            "rules": [
                                "Do not infer dating status, intimacy, or relationship labels.",
                                "Do not infer that people were together unless verified metadata says so.",
                                "Do not include image paths, exact GPS, face crop, or embedding.",
                                "Summarize observable activities only.",
                            ],
                        }
                    ),
                },
            ],
            schema=MEDIA_DAY_SUMMARY_SCHEMA,
            expect_json=True,
        )
        if result.ok and isinstance(result.data, dict):
            if usage:
                usage.record(
                    stage="media_day_summary",
                    backend="llm",
                    success=True,
                    latency_ms=result.latency_ms,
                    structured_output_used=result.structured_output_used,
                    schema_validation_success=True,
                )
            return _media_day_from_data(day, result.data)
        if usage:
            usage.record(
                stage="media_day_summary",
                backend="rule",
                success=False,
                latency_ms=result.latency_ms,
                structured_output_used=result.structured_output_used,
                fallback_reason=result.error or "invalid llm media day summary",
            )
    return _rule_media_day(day, safe_items)


def _safe_media(item: MediaEvidence, *, profile: ProfileContext | None, mode: str) -> dict[str, object]:
    del profile
    excerpt = None if mode == "public" else _short(_redact_private_markers(item.excerpt or item.summary), MAX_MEDIA_EXCERPT_CHARS)
    return {
        "ref": item.source_pointer or f"media_caption:{item.source_id}",
        "date": item.date,
        "media_type": item.media_type,
        "summary": _short(_redact_private_markers(item.summary), 140),
        "caption_excerpt": excerpt,
        "place_label": _safe_place_label(item.place_id),
        "confidence": round(item.confidence, 3),
    }


def _media_day_from_data(day: str, data: dict[str, Any]) -> MediaDaySummary:
    return MediaDaySummary(
        date=str(data.get("date") or day)[:10],
        has_media=bool(data.get("has_media")),
        likely_activities=tuple(_short(str(item), 80) for item in data.get("likely_activities", []) if isinstance(item, str)),
        place_summaries=tuple(_short(str(item), 80) for item in data.get("place_summaries", []) if isinstance(item, str)),
        people_roles_present=tuple(_short(str(item), 80) for item in data.get("people_roles_present", []) if isinstance(item, str)),
        summary=_short(str(data.get("summary") or ""), 180),
        confidence=_score(data.get("confidence")),
        evidence_media_refs=tuple(str(item)[:120] for item in data.get("evidence_media_refs", []) if isinstance(item, str)),
        cautions=tuple(_short(str(item), 160) for item in data.get("cautions", []) if isinstance(item, str)),
        backend="llm",
    )


def _rule_media_day(day: str, media_items: list[dict[str, object]]) -> MediaDaySummary:
    if not media_items:
        return MediaDaySummary(
            date=day,
            has_media=False,
            likely_activities=(),
            place_summaries=(),
            people_roles_present=(),
            summary="写真・メディアから確認できる活動候補はありません。",
            confidence=0.35,
            evidence_media_refs=(),
            cautions=("写真がない日については、活動の有無を断定できません。",),
        )
    haystack = " ".join(str(item.get("summary", "")) + " " + str(item.get("caption_excerpt") or "") for item in media_items)
    activities: list[str] = []
    if any(token in haystack for token in ("カフェ", "喫茶")):
        activities.append("カフェのような場所")
    if any(token in haystack for token in ("食事", "ご飯", "レストラン")):
        activities.append("食事に関係する外出")
    if not activities:
        activities.append("写真・メディアに残る活動候補")
    return MediaDaySummary(
        date=day,
        has_media=True,
        likely_activities=tuple(activities),
        place_summaries=(),
        people_roles_present=(),
        summary=f"{len(media_items)}件のメディア候補から、{', '.join(activities)}が見えます。",
        confidence=0.5,
        evidence_media_refs=tuple(str(item.get("ref")) for item in media_items[:5] if item.get("ref")),
        cautions=("写真だけでは一緒にいたことや仲直りは断定できません。",),
    )


def _safe_place_label(value: str | None) -> str | None:
    if not value:
        return None
    text = str(value)
    if "/" in text or "\\" in text:
        return "[redacted_path]"
    return _short(text, 80)


def _redact_private_markers(value: str) -> str:
    parts: list[str] = []
    for token in str(value or "").split():
        if "/" in token or "\\" in token:
            parts.append("[redacted_path]")
        elif "embedding" in token.lower() or "face" in token.lower():
            parts.append("[redacted_face_data]")
        elif re.search(r"\d{1,3}\.\d{3,}", token):
            parts.append("[redacted_gps]")
        else:
            parts.append(token)
    return " ".join(parts)


def _score(value: object) -> float:
    try:
        return max(0.0, min(1.0, float(value)))
    except (TypeError, ValueError):
        return 0.5


def _short(value: str, limit: int) -> str:
    text = " ".join(str(value or "").split())
    if len(text) <= limit:
        return text
    return text[: limit - 3] + "..."
