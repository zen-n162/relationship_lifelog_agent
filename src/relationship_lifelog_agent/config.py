from __future__ import annotations

from dataclasses import dataclass, field, fields, is_dataclass
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
class FullScanSettings:
    batch_strategy: str = "hybrid"
    max_items_per_batch: int = 200
    max_chars_per_batch: int = 80000
    overlap_items: int = 5
    max_batches_per_run: int = 200
    max_reanalysis_rounds: int = 5
    summarize_after_each_batch: bool = True
    preserve_raw_refs: bool = True


@dataclass(frozen=True)
class AnalysisSettings:
    mode: str = "safe_window"
    default_scope: str = "ask_llm"
    allow_full_corpus: bool = True
    allow_full_range: bool = True
    allow_single_context_full_prompt: bool = True
    allow_iterative_full_scan: bool = True
    full_scan: FullScanSettings = field(default_factory=FullScanSettings)


@dataclass(frozen=True)
class PrivateFullLlmPayloadSettings:
    allow_raw_line_text: bool = True
    allow_raw_note_text: bool = True
    allow_photo_paths: bool = True
    allow_exact_gps: bool = True
    allow_face_crops: bool = True
    allow_face_embeddings: bool = True
    allow_raw_face_embedding_values: bool = False
    allow_private_file_paths: bool = True
    allow_unverified_person_candidates: bool = True
    allow_unverified_speaker_candidates: bool = True
    allow_full_prompt_logging: bool = False
    allow_raw_payload_cache: bool = False


@dataclass(frozen=True)
class UiSettings:
    style: str = "chatgpt_like"
    show_sidebar: bool = False
    show_debug_by_default: bool = False
    show_evidence_collapsed: bool = True
    max_evidence_items: int = 8
    stream_progress: bool = True
    show_progress_messages: bool = True
    show_tool_usage: bool = True
    show_safe_reasoning_summary: bool = True
    show_raw_llm_thinking: bool = False
    progress_messages_collapsed_when_done: bool = True


@dataclass(frozen=True)
class RelationshipSettings:
    default_profile_id: int = 1
    post_conflict_window_days: int = 14
    surrounding_media_days_before: int = 1
    surrounding_media_days_after: int = 1
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
    num_ctx: int = 32768
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
class VisionSettings:
    enabled: bool = False
    provider: str = "ollama"
    model: str | None = None
    pass_images: bool = False
    pass_face_crops: bool = False
    max_images_per_batch: int = 10


@dataclass(frozen=True)
class Settings:
    app: AppSettings = field(default_factory=AppSettings)
    adapter: AdapterSettings = field(default_factory=AdapterSettings)
    analysis: AnalysisSettings = field(default_factory=AnalysisSettings)
    private_full_llm_payload: PrivateFullLlmPayloadSettings = field(default_factory=PrivateFullLlmPayloadSettings)
    paths: PathSettings = field(default_factory=PathSettings)
    ui: UiSettings = field(default_factory=UiSettings)
    relationship: RelationshipSettings = field(default_factory=RelationshipSettings)
    privacy: PrivacySettings = field(default_factory=PrivacySettings)
    llm: LlmSettings = field(default_factory=LlmSettings)
    vision: VisionSettings = field(default_factory=VisionSettings)


def _merge_dataclass(cls: type, values: dict[str, Any] | None) -> Any:
    defaults = cls()
    if not values or not isinstance(values, dict):
        return defaults
    merged: dict[str, Any] = {}
    for item in fields(defaults):
        default_value = getattr(defaults, item.name)
        if item.name not in values:
            merged[item.name] = default_value
            continue
        value = values[item.name]
        if is_dataclass(default_value) and isinstance(value, dict):
            merged[item.name] = _merge_dataclass(type(default_value), value)
        else:
            merged[item.name] = value
    return cls(**merged)


def load_config(path: str | Path | None = None) -> Settings:
    data: dict[str, Any] = {}
    if path:
        config_path = Path(path).expanduser()
        if config_path.exists():
            data = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}

    settings = Settings(
        app=_merge_dataclass(AppSettings, data.get("app")),
        adapter=_merge_dataclass(AdapterSettings, data.get("adapter")),
        analysis=_merge_dataclass(AnalysisSettings, data.get("analysis")),
        private_full_llm_payload=_merge_dataclass(
            PrivateFullLlmPayloadSettings,
            data.get("private_full_llm_payload"),
        ),
        paths=_merge_dataclass(PathSettings, data.get("paths")),
        ui=_merge_dataclass(UiSettings, data.get("ui")),
        relationship=_merge_dataclass(RelationshipSettings, data.get("relationship")),
        privacy=_merge_dataclass(PrivacySettings, data.get("privacy")),
        llm=_merge_dataclass(LlmSettings, data.get("llm")),
        vision=_merge_dataclass(VisionSettings, data.get("vision")),
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
    if settings.analysis.mode not in {"safe_window", "private_full_range", "private_full_corpus"}:
        raise ValueError("Analysis mode must be safe_window, private_full_range, or private_full_corpus.")
    if settings.analysis.mode == "private_full_range" and not settings.analysis.allow_full_range:
        raise ValueError("private_full_range requires allow_full_range=true.")
    if settings.analysis.mode == "private_full_corpus" and not settings.analysis.allow_full_corpus:
        raise ValueError("private_full_corpus requires allow_full_corpus=true.")
    if settings.analysis.full_scan.batch_strategy not in {"chronological", "source_then_time", "hybrid"}:
        raise ValueError("full_scan.batch_strategy must be chronological, source_then_time, or hybrid.")
    if settings.ui.show_raw_llm_thinking:
        raise ValueError("Raw LLM thinking must not be shown.")
    if settings.llm.provider not in {"local", "ollama"}:
        raise ValueError("LLM provider must be local or ollama.")
    if settings.llm.enabled and not _is_local_llm_url(settings.llm.base_url):
        raise ValueError("LLM base_url must point to 127.0.0.1 or localhost.")
    if settings.vision.provider not in {"ollama"}:
        raise ValueError("Vision provider must be ollama.")
    if settings.vision.enabled and settings.vision.model and not _is_local_llm_url(settings.llm.base_url):
        raise ValueError("Vision model access must use a local LLM base_url.")


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
