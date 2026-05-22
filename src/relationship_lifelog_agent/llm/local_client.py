from __future__ import annotations


class LocalLlmClient:
    def __init__(self, model: str | None = None) -> None:
        self.model = model

    def is_configured(self) -> bool:
        return bool(self.model)

    def generate(self, prompt: str) -> str:
        del prompt
        raise RuntimeError("Local LLM is not configured. Automatic model download is disabled.")
