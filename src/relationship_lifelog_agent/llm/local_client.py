from __future__ import annotations

from dataclasses import dataclass
import json
import time
from typing import Any, Callable
from urllib.error import HTTPError, URLError
from urllib.parse import urlparse
from urllib.request import Request, urlopen

from relationship_lifelog_agent.config import LlmSettings


HttpPost = Callable[[str, dict[str, Any], float], dict[str, Any]]


@dataclass(frozen=True)
class LlmCallResult:
    ok: bool
    content: str = ""
    data: dict[str, Any] | None = None
    error: str | None = None
    latency_ms: float = 0.0
    structured_output_used: bool = False


class LocalLlmClient:
    """Small Ollama chat client with explicit local-only safety checks."""

    def __init__(self, settings: LlmSettings | str | None = None, *, http_post: HttpPost | None = None) -> None:
        if isinstance(settings, LlmSettings):
            self.settings = settings
        else:
            self.settings = LlmSettings(model=settings)
        self._http_post = http_post or _urllib_post_json

    @property
    def provider(self) -> str:
        return "ollama" if self.settings.provider in {"local", "ollama"} else self.settings.provider

    @property
    def model(self) -> str | None:
        return self.settings.model

    def is_configured(self) -> bool:
        return bool(self.settings.enabled and self.settings.model and self.provider == "ollama")

    def chat(
        self,
        messages: list[dict[str, str]],
        *,
        schema: dict[str, Any] | None = None,
        expect_json: bool = True,
    ) -> LlmCallResult:
        if not self.is_configured():
            return LlmCallResult(ok=False, error="llm disabled or model unset")
        if not _is_local_url(self.settings.base_url):
            return LlmCallResult(ok=False, error="llm base_url is not local")

        structured = bool(self.settings.require_structured_output and expect_json)
        payload: dict[str, Any] = {
            "model": self.settings.model,
            "messages": messages,
            "stream": False,
            "think": False,
            "options": {"temperature": self.settings.temperature, "num_ctx": self.settings.num_ctx, "num_predict": 1200},
        }
        if structured:
            payload["format"] = schema or "json"

        url = self.settings.base_url.rstrip("/") + "/api/chat"
        started = time.perf_counter()
        try:
            response = self._http_post(url, payload, self.settings.timeout_seconds)
        except (OSError, ValueError, RuntimeError) as exc:
            return LlmCallResult(
                ok=False,
                error=f"{exc.__class__.__name__}: {exc}",
                latency_ms=(time.perf_counter() - started) * 1000,
                structured_output_used=structured,
            )
        latency_ms = (time.perf_counter() - started) * 1000
        content = str(response.get("message", {}).get("content", "")).strip()
        parsed = None
        if expect_json:
            try:
                parsed = json.loads(content)
            except json.JSONDecodeError as exc:
                return LlmCallResult(
                    ok=False,
                    content=content,
                    error=f"invalid json: {exc.msg}",
                    latency_ms=latency_ms,
                    structured_output_used=structured,
                )
        return LlmCallResult(
            ok=True,
            content=content,
            data=parsed if isinstance(parsed, dict) else None,
            latency_ms=latency_ms,
            structured_output_used=structured,
        )

    def generate(self, prompt: str) -> str:
        result = self.chat([{"role": "user", "content": prompt}], expect_json=False)
        if not result.ok:
            raise RuntimeError(result.error or "local LLM generation failed")
        return result.content


def _urllib_post_json(url: str, payload: dict[str, Any], timeout: float) -> dict[str, Any]:
    request = Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urlopen(request, timeout=timeout) as response:
            return json.loads(response.read().decode("utf-8"))
    except HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"ollama http {exc.code}: {body[:200]}") from exc
    except URLError as exc:
        raise RuntimeError(f"ollama connection failed: {exc.reason}") from exc


def _is_local_url(base_url: str) -> bool:
    parsed = urlparse(base_url)
    return parsed.scheme in {"http", "https"} and parsed.hostname in {"127.0.0.1", "localhost", "::1"}
