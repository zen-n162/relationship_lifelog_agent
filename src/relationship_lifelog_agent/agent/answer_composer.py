from __future__ import annotations

from relationship_lifelog_agent.adapters.types import EvidenceItem
from relationship_lifelog_agent.analytics.evidence_scoring import AnalysisResult, confidence_label
from relationship_lifelog_agent.config import Settings
from relationship_lifelog_agent.llm.local_client import LocalLlmClient
from relationship_lifelog_agent.llm.prompts import FORBIDDEN_PHRASES, SYSTEM_POLICY
from relationship_lifelog_agent.llm.schemas import ANSWER_COMPOSITION_SCHEMA
from relationship_lifelog_agent.llm.usage import LlmUsageTrace
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


def compose_answer_with_llm(
    template_answer: str,
    *,
    question: str,
    settings: Settings,
    llm_client: LocalLlmClient | None,
    usage: LlmUsageTrace | None,
    mode: str = "private",
    person_names: list[str] | None = None,
) -> str:
    if not (
        settings.llm.enabled
        and settings.llm.model
        and settings.llm.use_for_answer_composition
        and llm_client
        and llm_client.is_configured()
    ):
        return template_answer
    safe_template = sanitize_answer(template_answer, mode=mode, person_names=person_names)
    prompt = {
        "task": "Rewrite the already-sanitized Japanese answer in a natural but cautious style.",
        "question": question,
        "template_answer": safe_template,
        "rules": [
            "Keep all counts, dates, aggregate lines, confidence values, evidence strength values unchanged.",
            "Do not add new facts.",
            "Do not infer relationship labels.",
            "Do not infer inner feelings.",
            "Do not include forbidden phrases.",
            "Keep sections: 要約, 集計, 具体例, 根拠, 信頼度, 注意.",
        ],
        "forbidden_phrases": list(FORBIDDEN_PHRASES),
    }
    result = llm_client.chat(
        [
            {"role": "system", "content": SYSTEM_POLICY},
            {"role": "user", "content": str(prompt)},
        ],
        schema=ANSWER_COMPOSITION_SCHEMA,
        expect_json=True,
    )
    if not result.ok or not isinstance(result.data, dict):
        _record_answer_fallback(usage, result.latency_ms, result.structured_output_used, result.error or "empty llm answer")
        return template_answer
    answer = str(result.data.get("answer_markdown") or "").strip()
    if not _llm_answer_preserves_template_facts(answer, safe_template):
        _record_answer_fallback(usage, result.latency_ms, result.structured_output_used, "llm answer changed required facts")
        return template_answer
    sanitized = sanitize_answer(answer, mode=mode, person_names=person_names)
    if usage:
        usage.record(
            stage="answer_composition",
            backend="llm",
            success=True,
            latency_ms=result.latency_ms,
            structured_output_used=result.structured_output_used,
            schema_validation_success=True,
        )
    return sanitized


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


def _llm_answer_preserves_template_facts(answer: str, template_answer: str) -> bool:
    if not answer:
        return False
    for section in ("要約:", "集計:", "具体例:", "根拠:", "信頼度:", "注意:"):
        if section not in answer:
            return False
    aggregate = _section_body(template_answer, "集計:", "具体例:")
    required_lines = [line.strip() for line in aggregate.splitlines() if line.strip().startswith("- ")]
    return all(line in answer for line in required_lines)


def _section_body(text: str, start: str, end: str) -> str:
    if start not in text:
        return ""
    body = text.split(start, 1)[1]
    if end in body:
        body = body.split(end, 1)[0]
    return body


def _record_answer_fallback(
    usage: LlmUsageTrace | None,
    latency_ms: float,
    structured_output_used: bool,
    reason: str,
) -> None:
    if usage:
        usage.record(
            stage="answer_composition",
            backend="template",
            success=False,
            latency_ms=latency_ms,
            structured_output_used=structured_output_used,
            fallback_reason=reason,
        )
