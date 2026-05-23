from __future__ import annotations

from relationship_lifelog_agent.adapters.types import EvidenceItem
from relationship_lifelog_agent.analytics.evidence_scoring import AnalysisResult, confidence_label
from relationship_lifelog_agent.privacy.guard import sanitize_answer
from relationship_lifelog_agent.privacy.redaction import redact_evidence_for_mode


MAX_DISPLAY_EVIDENCE_ITEMS = 5


def compose_answer(
    result: AnalysisResult,
    mode: str = "private",
    person_names: list[str] | None = None,
) -> str:
    evidence = [
        redact_evidence_for_mode(item, mode=mode, person_names=person_names)
        for item in result.evidence[:MAX_DISPLAY_EVIDENCE_ITEMS]
    ]
    parts = [
        "要約:\n" + result.summary,
        "集計:\n" + _format_aggregate(result.aggregate),
        "具体例:\n" + _format_examples(result.examples),
        "根拠:\n" + _collapsible_markdown("根拠を表示", _format_evidence(evidence)),
        "信頼度:\n"
        + f"{confidence_label(result.confidence)} "
        + f"(confidence={result.confidence:.2f}, evidence_strength={result.evidence_strength:.2f})",
        "注意:\n" + _collapsible_markdown("注意を表示", _format_cautions(result.cautions)),
    ]
    return sanitize_answer("\n\n".join(parts), mode=mode, person_names=person_names)


def _format_aggregate(values: dict[str, object]) -> str:
    if not values:
        return "- 集計対象はありません。"
    return "\n".join(f"- {key}: {value}" for key, value in values.items())


def _format_examples(examples: list[str]) -> str:
    if not examples:
        return "- 具体例はありません。"
    return "\n".join(f"- {example}" for example in examples)


def _format_evidence(evidence: list[EvidenceItem]) -> str:
    if not evidence:
        return "- 根拠アイテムはありません。"
    lines = []
    for item in evidence:
        excerpt = f" / excerpt: {item.excerpt}" if item.excerpt else ""
        lines.append(
            f"- {item.date} [{item.source_type}:{item.role}] {item.summary}{excerpt}"
        )
    return "\n".join(lines)


def _format_cautions(cautions: list[str]) -> str:
    if not cautions:
        return "- 候補として扱い、断定しません。"
    return "\n".join(f"- {caution}" for caution in cautions)


def _collapsible_markdown(summary: str, body: str) -> str:
    return f"<details>\n<summary>{summary}</summary>\n\n{body}\n\n</details>"
