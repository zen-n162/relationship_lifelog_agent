from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class AppSettings:
    name: str = "relationship_lifelog_agent"
    mode: str = "private"
    host: str = "127.0.0.1"
    port: int = 7862
    allow_external_api: bool = False
    allow_model_auto_download: bool = False
    allow_gradio_share: bool = False


@dataclass(frozen=True)
class PathSettings:
    personal_lifelog_rag: str = "~/MyApplication/personal_lifelog_rag"
    notes_lifelog_rag: str = "~/MyApplication/notes_lifelog_rag"
    relationship_db: str = "~/MyApplication/relationship_lifelog_agent/data/relationship.sqlite"
    personal_lifelog_db: str | None = None
    notes_lifelog_db: str | None = None
    personal_lifelog_export_dir: str | None = None
    notes_lifelog_export_dir: str | None = None


@dataclass(frozen=True)
class AdapterSettings:
    backend: str = "mock"
    upstream_access_mode: str = "readonly"
    prefer: str = "auto"
    allow_sqlite_readonly: bool = True
    allow_cli_subprocess: bool = True
    allow_python_import: bool = True
    copy_raw_upstream_data: bool = False


@dataclass(frozen=True)
class UiSettings:
    style: str = "chatgpt_like"
    show_sidebar: bool = False
    show_debug_by_default: bool = False
    show_evidence_collapsed: bool = True
    max_evidence_items: int = 8


@dataclass(frozen=True)
class RelationshipSettings:
    default_profile_id: int = 1
    post_conflict_window_days: int = 14
    conflict_min_confidence: float = 0.45
    evidence_min_strength_for_answer: float = 0.35
    require_manual_partner_label: bool = True


@dataclass(frozen=True)
class PrivacySettings:
    public_mode_enabled: bool = False
    redact_exact_gps: bool = True
    redact_line_full_text: bool = True
    redact_note_full_text: bool = True
    allow_short_quotes_private: bool = True
    max_private_quote_chars: int = 120
    forbid_public_relationship_labels: bool = True


@dataclass(frozen=True)
class LlmSettings:
    provider: str = "local"
    model: str | None = None
    base_url: str = "http://127.0.0.1:11434"
    temperature: float = 0.2
    max_context_items: int = 30
    enabled: bool = False
    require_structured_output: bool = True
    fallback_to_rules: bool = True
    use_for_question_understanding: bool = False
    use_for_information_needs: bool = False
    use_for_query_planning: bool = False
    use_for_answer_composition: bool = False
    timeout_seconds: float = 120.0


@dataclass(frozen=True)
class Settings:
    app: AppSettings = field(default_factory=AppSettings)
    adapter: AdapterSettings = field(default_factory=AdapterSettings)
    paths: PathSettings = field(default_factory=PathSettings)
    ui: UiSettings = field(default_factory=UiSettings)
    relationship: RelationshipSettings = field(default_factory=RelationshipSettings)
    privacy: PrivacySettings = field(default_factory=PrivacySettings)
    llm: LlmSettings = field(default_factory=LlmSettings)


def _merge_dataclass(cls: type, values: dict[str, Any] | None) -> Any:
    defaults = cls()
    if not values:
        return defaults
    allowed = defaults.__dataclass_fields__.keys()
    clean = {key: value for key, value in values.items() if key in allowed}
    return cls(**{**defaults.__dict__, **clean})


def load_config(path: str | Path | None = None) -> Settings:
    data: dict[str, Any] = {}
    if path:
        config_path = Path(path).expanduser()
        if config_path.exists():
            data = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}

    settings = Settings(
        app=_merge_dataclass(AppSettings, data.get("app")),
        adapter=_merge_dataclass(AdapterSettings, data.get("adapter")),
        paths=_merge_dataclass(PathSettings, data.get("paths")),
        ui=_merge_dataclass(UiSettings, data.get("ui")),
        relationship=_merge_dataclass(RelationshipSettings, data.get("relationship")),
        privacy=_merge_dataclass(PrivacySettings, data.get("privacy")),
        llm=_merge_dataclass(LlmSettings, data.get("llm")),
    )
    validate_local_safety(settings)
    return settings


def validate_local_safety(settings: Settings) -> None:
    if settings.app.allow_external_api:
        raise ValueError("External APIs must remain disabled.")
    if settings.app.allow_model_auto_download:
        raise ValueError("Automatic model downloads must remain disabled.")
    if settings.app.allow_gradio_share:
        raise ValueError("Gradio share must remain disabled.")
    if settings.app.host != "127.0.0.1":
        raise ValueError("Default server host must be 127.0.0.1.")
    if settings.adapter.backend not in {"mock", "upstream_readonly"}:
        raise ValueError("Adapter backend must be mock or upstream_readonly.")
    if settings.adapter.upstream_access_mode != "readonly":
        raise ValueError("Upstream access mode must remain readonly.")
    if settings.adapter.copy_raw_upstream_data:
        raise ValueError("Raw upstream data must not be copied.")
    if settings.adapter.backend == "upstream_readonly" and not settings.adapter.allow_sqlite_readonly:
        raise ValueError("upstream_readonly requires allow_sqlite_readonly=true.")
    if settings.llm.provider not in {"local", "ollama"}:
        raise ValueError("LLM provider must be local or ollama.")
    if settings.llm.enabled and not _is_local_llm_url(settings.llm.base_url):
        raise ValueError("LLM base_url must point to 127.0.0.1 or localhost.")


def _is_local_llm_url(base_url: str) -> bool:
    from urllib.parse import urlparse

    parsed = urlparse(base_url)
    if parsed.scheme not in {"http", "https"}:
        return False
    return parsed.hostname in {"127.0.0.1", "localhost", "::1"}


def gradio_launch_kwargs(settings: Settings, *, port: int | None = None) -> dict[str, Any]:
    server_port = settings.app.port if port is None else port
    if server_port < 1 or server_port > 65535:
        raise ValueError("Port must be between 1 and 65535.")
    return {
        "server_name": settings.app.host,
        "server_port": server_port,
        "share": False,
        "allowed_paths": [],
        "footer_links": [],
        "enable_monitoring": False,
    }
