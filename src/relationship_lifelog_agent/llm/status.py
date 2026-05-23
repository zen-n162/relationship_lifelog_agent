from __future__ import annotations

from dataclasses import asdict, dataclass
import json

from relationship_lifelog_agent.config import Settings, load_config
from relationship_lifelog_agent.llm.local_client import LocalLlmClient


STATUS_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "properties": {"ok": {"type": "boolean"}},
    "required": ["ok"],
}


@dataclass(frozen=True)
class LlmStatusReport:
    enabled: bool
    provider: str
    model: str | None
    base_url: str
    connectivity: str
    test_prompt_success: bool
    structured_output_success: bool
    fallback_to_rules: bool
    error: str | None = None


def run_llm_status(config_path: str | None = None, *, client: LocalLlmClient | None = None) -> LlmStatusReport:
    settings = load_config(config_path)
    return check_llm_status(settings, client=client)


def check_llm_status(settings: Settings, *, client: LocalLlmClient | None = None) -> LlmStatusReport:
    llm = settings.llm
    provider = "ollama" if llm.provider in {"local", "ollama"} else llm.provider
    if not llm.enabled:
        return LlmStatusReport(False, provider, llm.model, llm.base_url, "disabled", False, False, llm.fallback_to_rules)
    if not llm.model:
        return LlmStatusReport(True, provider, None, llm.base_url, "model_unset", False, False, llm.fallback_to_rules)
    client = client or LocalLlmClient(llm)
    result = client.chat(
        [
            {"role": "system", "content": "Return JSON only."},
            {"role": "user", "content": '{"ok": true} とだけ同じ意味のJSONを返してください。'},
        ],
        schema=STATUS_SCHEMA,
        expect_json=True,
    )
    return LlmStatusReport(
        enabled=True,
        provider=provider,
        model=llm.model,
        base_url=llm.base_url,
        connectivity="ok" if result.ok else "error",
        test_prompt_success=result.ok,
        structured_output_success=bool(result.ok and result.data and result.data.get("ok") is True),
        fallback_to_rules=llm.fallback_to_rules,
        error=result.error,
    )


def render_llm_status_text(report: LlmStatusReport) -> str:
    lines = ["relationship_lifelog_agent llm status:"]
    for key, value in asdict(report).items():
        lines.append(f"- {key}: {value}")
    return "\n".join(lines)


def render_llm_status_json(report: LlmStatusReport) -> str:
    return json.dumps(asdict(report), ensure_ascii=False, indent=2)
