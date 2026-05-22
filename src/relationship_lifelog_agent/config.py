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
    temperature: float = 0.2
    max_context_items: int = 30


@dataclass(frozen=True)
class Settings:
    app: AppSettings = field(default_factory=AppSettings)
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


def gradio_launch_kwargs(settings: Settings) -> dict[str, Any]:
    return {
        "server_name": settings.app.host,
        "server_port": settings.app.port,
        "share": False,
        "allowed_paths": [],
        "footer_links": [],
        "enable_monitoring": False,
    }
