from __future__ import annotations

from dataclasses import dataclass, fields, is_dataclass, replace
import importlib.util
import json
from pathlib import Path
import sqlite3
import subprocess
import sys
from typing import Any

import yaml

from relationship_lifelog_agent.adapters.upstream_sqlite import UpstreamReadOnlyError, open_readonly_sqlite
from relationship_lifelog_agent.config import (
    AdapterSettings,
    AnalysisSettings,
    AppSettings,
    LlmSettings,
    PathSettings,
    PrivateFullLlmPayloadSettings,
    PrivacySettings,
    RelationshipSettings,
    Settings,
    UiSettings,
    VisionSettings,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
EXPECTED_RELATIONSHIP_TABLES = frozenset(
    {
        "relationship_profiles",
        "relationship_events",
        "relationship_event_evidence",
        "interaction_metrics",
        "post_conflict_activities",
        "relationship_review_actions",
    }
)
STATUS_ORDER = {"OK": 0, "WARN": 1, "ERROR": 2}


@dataclass(frozen=True)
class DoctorCheck:
    name: str
    status: str
    message: str
    details: dict[str, Any]


@dataclass(frozen=True)
class DoctorReport:
    status: str
    checks: list[DoctorCheck]

    @property
    def has_errors(self) -> bool:
        return self.status == "ERROR"


def run_doctor(
    *,
    config_path: str | Path | None = None,
    backend: str | None = None,
    project_root: Path = PROJECT_ROOT,
) -> DoctorReport:
    root = project_root.resolve()
    example_path = root / "config.example.yaml"
    local_path = root / "config.local.yaml"
    selected_config = _select_config_path(config_path, example_path, local_path)
    settings, config_error = _load_settings_unvalidated(selected_config)
    if backend:
        settings = replace(settings, adapter=replace(settings.adapter, backend=backend))

    checks: list[DoctorCheck] = []
    checks.append(_check_file("config.example_yaml", example_path, required=True))
    checks.append(_check_config_local(local_path, root))
    checks.extend(_check_safety_settings(settings, config_error))
    checks.append(_check_relationship_db(settings.paths.relationship_db))
    checks.append(_check_upstream_db("personal_lifelog_db", settings.paths.personal_lifelog_db))
    checks.append(_check_upstream_db("notes_lifelog_db", settings.paths.notes_lifelog_db))
    checks.append(_check_privacy(settings.privacy))
    checks.append(_check_eval(root))

    if selected_config:
        checks.insert(
            0,
            DoctorCheck(
                name="config_source",
                status="OK",
                message="config source selected",
                details={"path": _redacted_path(selected_config), "explicit": config_path is not None},
            ),
        )

    overall = _overall_status(checks)
    return DoctorReport(status=overall, checks=checks)


def render_doctor_text(report: DoctorReport) -> str:
    lines = [f"relationship_lifelog_agent doctor: {report.status}"]
    for check in report.checks:
        lines.append(f"[{check.status}] {check.name}: {check.message}")
        redacted = _redact_for_output(check.details)
        for key, value in redacted.items():
            lines.append(f"  - {key}: {json.dumps(value, ensure_ascii=False)}")
    return "\n".join(lines)


def render_doctor_json(report: DoctorReport) -> str:
    payload = {
        "status": report.status,
        "checks": [
            {
                "name": check.name,
                "status": check.status,
                "message": check.message,
                "details": _redact_for_output(check.details),
            }
            for check in report.checks
        ],
    }
    return json.dumps(payload, ensure_ascii=False, indent=2)


def _select_config_path(config_path: str | Path | None, example_path: Path, local_path: Path) -> Path | None:
    if config_path:
        return Path(config_path).expanduser()
    if local_path.exists():
        return local_path
    if example_path.exists():
        return example_path
    return None


def _load_settings_unvalidated(config_path: Path | None) -> tuple[Settings, str | None]:
    data: dict[str, Any] = {}
    if config_path:
        try:
            data = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
        except Exception as exc:  # pragma: no cover - defensive for malformed private configs
            return Settings(), f"config could not be read: {exc.__class__.__name__}"
    try:
        return (
            Settings(
                app=_merge(AppSettings, data.get("app")),
                adapter=_merge(AdapterSettings, data.get("adapter")),
                analysis=_merge(AnalysisSettings, data.get("analysis")),
                private_full_llm_payload=_merge(
                    PrivateFullLlmPayloadSettings,
                    data.get("private_full_llm_payload"),
                ),
                paths=_merge(PathSettings, data.get("paths")),
                ui=_merge(UiSettings, data.get("ui")),
                relationship=_merge(RelationshipSettings, data.get("relationship")),
                privacy=_merge(PrivacySettings, data.get("privacy")),
                llm=_merge(LlmSettings, data.get("llm")),
                vision=_merge(VisionSettings, data.get("vision")),
            ),
            None,
        )
    except Exception as exc:  # pragma: no cover - defensive for malformed private configs
        return Settings(), f"config could not be parsed: {exc.__class__.__name__}"


def _merge(cls: type, values: dict[str, Any] | None) -> Any:
    default = cls()
    if not isinstance(values, dict):
        return default
    merged: dict[str, Any] = {}
    for item in fields(default):
        default_value = getattr(default, item.name)
        if item.name not in values:
            merged[item.name] = default_value
            continue
        value = values[item.name]
        if is_dataclass(default_value) and isinstance(value, dict):
            merged[item.name] = _merge(type(default_value), value)
        else:
            merged[item.name] = value
    return cls(**merged)


def _check_file(name: str, path: Path, *, required: bool) -> DoctorCheck:
    if path.exists():
        return DoctorCheck(name, "OK", "file exists", {"path": _redacted_path(path)})
    status = "ERROR" if required else "WARN"
    return DoctorCheck(name, status, "file is missing", {"path": _redacted_path(path)})


def _check_config_local(path: Path, root: Path) -> DoctorCheck:
    if not path.exists():
        return DoctorCheck(
            "config.local_yaml",
            "WARN",
            "config.local.yaml is not present; using example/default settings",
            {"path": _redacted_path(path), "exists": False},
        )
    tracked = _git_tracks_config_local(root)
    ignored = _gitignore_mentions_config_local(root)
    if tracked:
        return DoctorCheck(
            "config.local_yaml",
            "ERROR",
            "config.local.yaml exists but is tracked by git",
            {"path": _redacted_path(path), "exists": True, "tracked": True, "ignored": ignored},
        )
    status = "OK" if ignored else "WARN"
    message = "config.local.yaml exists and is not tracked"
    if not ignored:
        message = "config.local.yaml exists, but .gitignore entry was not found"
    return DoctorCheck(
        "config.local_yaml",
        status,
        message,
        {"path": _redacted_path(path), "exists": True, "tracked": False, "ignored": ignored},
    )


def _check_safety_settings(settings: Settings, config_error: str | None) -> list[DoctorCheck]:
    checks: list[DoctorCheck] = []
    if config_error:
        checks.append(DoctorCheck("config_parse", "ERROR", config_error, {}))
    checks.extend(
        [
            _bool_check("app.host", settings.app.host == "127.0.0.1", settings.app.host, "127.0.0.1"),
            _false_check("app.allow_external_api", settings.app.allow_external_api),
            _false_check("app.allow_model_auto_download", settings.app.allow_model_auto_download),
            _false_check("app.allow_gradio_share", settings.app.allow_gradio_share),
            _choice_check("adapter.backend", settings.adapter.backend, {"mock", "upstream_readonly"}),
            _bool_check(
                "adapter.upstream_access_mode",
                settings.adapter.upstream_access_mode == "readonly",
                settings.adapter.upstream_access_mode,
                "readonly",
            ),
            _false_check("adapter.copy_raw_upstream_data", settings.adapter.copy_raw_upstream_data),
        ]
    )
    if settings.adapter.backend == "upstream_readonly" and not settings.adapter.allow_sqlite_readonly:
        checks.append(
            DoctorCheck(
                "adapter.allow_sqlite_readonly",
                "ERROR",
                "upstream_readonly backend requires SQLite read-only access to be enabled",
                {"value": settings.adapter.allow_sqlite_readonly},
            )
        )
    else:
        checks.append(
            DoctorCheck(
                "adapter.allow_sqlite_readonly",
                "OK",
                "SQLite read-only adapter setting is compatible",
                {"value": settings.adapter.allow_sqlite_readonly},
            )
        )
    return checks


def _bool_check(name: str, ok: bool, value: Any, expected: Any) -> DoctorCheck:
    if ok:
        return DoctorCheck(name, "OK", "setting is safe", {"value": value})
    return DoctorCheck(name, "ERROR", "setting is unsafe", {"value": value, "expected": expected})


def _false_check(name: str, value: bool) -> DoctorCheck:
    if value is False:
        return DoctorCheck(name, "OK", "setting is disabled", {"value": value})
    return DoctorCheck(name, "ERROR", "setting must remain disabled", {"value": value, "expected": False})


def _choice_check(name: str, value: str, allowed: set[str]) -> DoctorCheck:
    if value in allowed:
        return DoctorCheck(name, "OK", "setting is allowed", {"value": value})
    return DoctorCheck(name, "ERROR", "setting is not allowed", {"value": value, "allowed": sorted(allowed)})


def _check_relationship_db(db_path: str | None) -> DoctorCheck:
    if not db_path:
        return DoctorCheck("relationship_db", "ERROR", "relationship_db path is not configured", {})
    path = Path(db_path).expanduser()
    details: dict[str, Any] = {"path": _redacted_path(path), "exists": path.exists()}
    if not path.parent.exists():
        return DoctorCheck("relationship_db", "WARN", "relationship_db parent directory does not exist", details)
    if not path.exists():
        return DoctorCheck("relationship_db", "WARN", "relationship DB is not initialized yet", details)
    try:
        with sqlite3.connect(path) as conn:
            tables = _sqlite_table_counts(conn)
            missing = sorted(EXPECTED_RELATIONSHIP_TABLES - set(tables))
            profile_count = _safe_count(conn, "relationship_profiles") if "relationship_profiles" in tables else 0
    except sqlite3.Error:
        return DoctorCheck("relationship_db", "ERROR", "relationship DB cannot be opened", details)
    details.update(
        {
            "schema_initialized": not missing,
            "missing_tables": missing,
            "table_count": len(tables),
            "profile_count": profile_count,
        }
    )
    if missing:
        return DoctorCheck("relationship_db", "ERROR", "relationship DB schema is incomplete", details)
    if profile_count < 1:
        return DoctorCheck("relationship_db", "WARN", "relationship DB has no manually configured profiles", details)
    return DoctorCheck("relationship_db", "OK", "relationship DB schema is initialized", details)


def _check_upstream_db(name: str, db_path: str | None) -> DoctorCheck:
    if not db_path:
        return DoctorCheck(name, "WARN", f"{name} is not configured", {"configured": False})
    details: dict[str, Any] = {"path": _redacted_path(db_path), "configured": True}
    try:
        conn = open_readonly_sqlite(db_path)
    except UpstreamReadOnlyError:
        return DoctorCheck(name, "ERROR", f"{name} cannot be opened read-only", details)
    except sqlite3.Error:
        return DoctorCheck(name, "ERROR", f"{name} SQLite read-only connection failed", details)
    try:
        tables = _sqlite_table_counts(conn)
    finally:
        conn.close()
    safe_tables, redacted_count = _redact_sensitive_table_counts(tables)
    details.update(
        {
            "table_count": len(tables),
            "tables": safe_tables,
            "redacted_sensitive_table_count": redacted_count,
        }
    )
    return DoctorCheck(name, "OK", f"{name} opened read-only", details)


def _check_privacy(settings: PrivacySettings) -> DoctorCheck:
    flags = {
        "redact_exact_gps": settings.redact_exact_gps,
        "redact_line_full_text": settings.redact_line_full_text,
        "redact_note_full_text": settings.redact_note_full_text,
        "forbid_public_relationship_labels": settings.forbid_public_relationship_labels,
    }
    unsafe = sorted(name for name, value in flags.items() if value is not True)
    if unsafe:
        return DoctorCheck("privacy.redaction", "ERROR", "public/private redaction settings are unsafe", {"unsafe": unsafe})
    return DoctorCheck("privacy.redaction", "OK", "public/private redaction settings are enabled", flags)


def _check_eval(root: Path) -> DoctorCheck:
    eval_path = root / "eval" / "run_eval.py"
    private_cases = root / "eval" / "private_eval_cases.yaml"
    safety_cases = root / "eval" / "safety_eval_cases.yaml"
    if not eval_path.exists() or not private_cases.exists() or not safety_cases.exists():
        return DoctorCheck("eval", "ERROR", "eval files are missing", {"files_present": False})
    spec = importlib.util.spec_from_file_location("relationship_doctor_eval", eval_path)
    if spec is None or spec.loader is None:
        return DoctorCheck("eval", "ERROR", "eval runner cannot be imported", {"files_present": True})
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    try:
        spec.loader.exec_module(module)
        result = module.run_eval(case_paths=[private_cases, safety_cases])
    except Exception as exc:  # pragma: no cover - defensive for eval regressions
        return DoctorCheck("eval", "ERROR", "eval runner failed", {"error_type": exc.__class__.__name__})
    details = {"passed": result.passed, "failed": result.failed, "skipped": result.skipped}
    if result.failed:
        return DoctorCheck("eval", "ERROR", "eval has failing cases", details)
    return DoctorCheck("eval", "OK", "eval is executable", details)


def _sqlite_table_counts(conn: sqlite3.Connection) -> dict[str, int | str]:
    rows = conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%' ORDER BY name"
    ).fetchall()
    table_counts: dict[str, int | str] = {}
    for row in rows:
        name = str(row[0])
        table_counts[name] = _safe_count(conn, name)
    return table_counts


def _redact_sensitive_table_counts(tables: dict[str, int | str]) -> tuple[dict[str, int | str], int]:
    safe: dict[str, int | str] = {}
    redacted_count = 0
    for name, count in tables.items():
        if _is_sensitive_upstream_table(name):
            redacted_count += 1
            continue
        safe[name] = count
    return safe, redacted_count


def _is_sensitive_upstream_table(name: str) -> bool:
    lowered = name.lower()
    sensitive_tokens = (
        "face",
        "embedding",
        "gps",
        "location",
        "photo",
        "path",
    )
    return any(token in lowered for token in sensitive_tokens)


def _safe_count(conn: sqlite3.Connection, table_name: str) -> int | str:
    try:
        row = conn.execute(f"SELECT COUNT(*) FROM {_quote_identifier(table_name)}").fetchone()
    except sqlite3.Error:
        return "unavailable"
    return int(row[0]) if row else 0


def _quote_identifier(value: str) -> str:
    return '"' + value.replace('"', '""') + '"'


def _git_tracks_config_local(root: Path) -> bool:
    try:
        result = subprocess.run(
            ["git", "-C", str(root), "ls-files", "--error-unmatch", "config.local.yaml"],
            check=False,
            capture_output=True,
            text=True,
            timeout=5,
        )
    except Exception:
        return False
    return result.returncode == 0


def _gitignore_mentions_config_local(root: Path) -> bool:
    path = root / ".gitignore"
    if not path.exists():
        return False
    lines = [line.strip() for line in path.read_text(encoding="utf-8").splitlines()]
    return "config.local.yaml" in lines


def _overall_status(checks: list[DoctorCheck]) -> str:
    value = max((STATUS_ORDER.get(check.status, 2) for check in checks), default=0)
    for status, order in STATUS_ORDER.items():
        if order == value:
            return status
    return "ERROR"


def _redacted_path(path: str | Path) -> str:
    del path
    return "[redacted_path]"


def _redact_for_output(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _redact_for_output(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_redact_for_output(item) for item in value]
    if isinstance(value, tuple):
        return [_redact_for_output(item) for item in value]
    if isinstance(value, str) and _looks_like_path(value):
        return "[redacted_path]"
    return value


def _looks_like_path(value: str) -> bool:
    return value.startswith("~") or value.startswith("/") or "\\" in value or "/." in value
