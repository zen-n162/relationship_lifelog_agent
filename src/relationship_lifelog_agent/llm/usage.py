from __future__ import annotations

from dataclasses import asdict, dataclass, field

from relationship_lifelog_agent.config import LlmSettings


@dataclass
class LlmCallTrace:
    stage: str
    backend: str
    success: bool
    latency_ms: float = 0.0
    structured_output_used: bool = False
    schema_validation_success: bool = False
    fallback_reason: str | None = None


@dataclass
class LlmUsageTrace:
    llm_enabled: bool
    llm_provider: str
    llm_model: str | None
    llm_call_count: int = 0
    question_understanding_backend: str = "rule"
    information_needs_backend: str = "rule"
    query_planning_backend: str = "rule"
    answer_composition_backend: str = "template"
    fallback_used: bool = False
    fallback_reasons: list[str] = field(default_factory=list)
    structured_output_used: bool = False
    schema_validation_success: bool = False
    latency_ms_by_stage: dict[str, float] = field(default_factory=dict)
    calls: list[LlmCallTrace] = field(default_factory=list)

    @classmethod
    def from_settings(cls, settings: LlmSettings) -> "LlmUsageTrace":
        provider = "ollama" if settings.provider in {"local", "ollama"} else settings.provider
        return cls(
            llm_enabled=bool(settings.enabled and settings.model),
            llm_provider=provider,
            llm_model=settings.model,
        )

    def record(
        self,
        *,
        stage: str,
        backend: str,
        success: bool,
        latency_ms: float = 0.0,
        structured_output_used: bool = False,
        schema_validation_success: bool = False,
        fallback_reason: str | None = None,
    ) -> None:
        if backend == "llm" or fallback_reason:
            self.llm_call_count += 1
        setattr(self, f"{stage}_backend", backend)
        self.structured_output_used = self.structured_output_used or structured_output_used
        self.schema_validation_success = self.schema_validation_success or schema_validation_success
        if latency_ms:
            self.latency_ms_by_stage[stage] = round(latency_ms, 2)
        if fallback_reason:
            self.fallback_used = True
            self.fallback_reasons.append(f"{stage}: {fallback_reason}")
        self.calls.append(
            LlmCallTrace(
                stage=stage,
                backend=backend,
                success=success,
                latency_ms=round(latency_ms, 2),
                structured_output_used=structured_output_used,
                schema_validation_success=schema_validation_success,
                fallback_reason=fallback_reason,
            )
        )

    def to_dict(self) -> dict[str, object]:
        data = asdict(self)
        data["calls"] = [asdict(call) for call in self.calls]
        return data

    def user_summary(self) -> str:
        if self.llm_call_count <= 0:
            return "今回はrule-based plannerで処理しました。"
        stages = []
        if self.question_understanding_backend == "llm":
            stages.append("質問理解")
        if self.information_needs_backend == "llm":
            stages.append("必要情報の整理")
        if self.query_planning_backend == "llm":
            stages.append("検索計画")
        if self.answer_composition_backend == "llm":
            stages.append("回答文の整形")
        used = "、".join(stages) if stages else "一部処理"
        suffix = "fallbackあり" if self.fallback_used else "fallbackなし"
        return f"{used}にはlocal LLMを使い、件数・日付・profile確定はPythonで確認しました。({suffix})"
