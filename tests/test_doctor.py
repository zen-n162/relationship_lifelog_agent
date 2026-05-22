from __future__ import annotations

import json
from pathlib import Path
import sqlite3

import pytest

from relationship_lifelog_agent.cli import main as cli_main
from relationship_lifelog_agent.db.repository import RelationshipRepository
from relationship_lifelog_agent.doctor import render_doctor_json, render_doctor_text, run_doctor


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_mock_backend_doctor_has_no_errors(tmp_path) -> None:
    config_path = _write_config(tmp_path, relationship_db=_initialized_relationship_db(tmp_path))

    report = run_doctor(config_path=config_path, backend="mock", project_root=PROJECT_ROOT)

    assert report.status in {"OK", "WARN"}
    assert not report.has_errors
    assert _check_status(report, "adapter.backend") == "OK"
    assert _check_status(report, "eval") == "OK"


def test_upstream_unconfigured_is_warn_not_error(tmp_path) -> None:
    config_path = _write_config(tmp_path, relationship_db=_initialized_relationship_db(tmp_path))

    report = run_doctor(config_path=config_path, backend="upstream_readonly", project_root=PROJECT_ROOT)

    assert not report.has_errors
    assert _check_status(report, "personal_lifelog_db") == "WARN"
    assert _check_status(report, "notes_lifelog_db") == "WARN"


def test_doctor_flags_share_true_as_error(tmp_path) -> None:
    config_path = _write_config(
        tmp_path,
        relationship_db=_initialized_relationship_db(tmp_path),
        extra_app={"allow_gradio_share": True},
    )

    report = run_doctor(config_path=config_path, backend="mock", project_root=PROJECT_ROOT)
    rendered = render_doctor_text(report)

    assert report.has_errors
    assert _check_status(report, "app.allow_gradio_share") == "ERROR"
    assert "share=True" not in rendered


def test_doctor_output_redacts_private_paths(tmp_path) -> None:
    relationship_db = _initialized_relationship_db(tmp_path / "private-root")
    config_path = _write_config(tmp_path / "private-root", relationship_db=relationship_db)

    report = run_doctor(config_path=config_path, backend="mock", project_root=PROJECT_ROOT)
    rendered_json = render_doctor_json(report)
    rendered_text = render_doctor_text(report)

    assert str(tmp_path) not in rendered_json
    assert str(tmp_path) not in rendered_text
    assert "[redacted_path]" in rendered_json
    json.loads(rendered_json)


def test_doctor_counts_upstream_tables_without_rows(tmp_path) -> None:
    upstream_db = tmp_path / "personal.sqlite"
    conn = sqlite3.connect(upstream_db)
    conn.execute("CREATE TABLE line_messages (id INTEGER PRIMARY KEY, text TEXT)")
    conn.execute("INSERT INTO line_messages (text) VALUES ('synthetic private row')")
    conn.commit()
    conn.close()
    config_path = _write_config(
        tmp_path,
        relationship_db=_initialized_relationship_db(tmp_path),
        extra_paths={"personal_lifelog_db": str(upstream_db)},
    )

    report = run_doctor(config_path=config_path, backend="upstream_readonly", project_root=PROJECT_ROOT)
    personal = next(check for check in report.checks if check.name == "personal_lifelog_db")
    rendered = render_doctor_json(report)

    assert personal.status == "OK"
    assert personal.details["tables"]["line_messages"] == 1
    assert "synthetic private row" not in rendered


def test_doctor_cli_exits_nonzero_on_error(tmp_path) -> None:
    config_path = _write_config(
        tmp_path,
        relationship_db=_initialized_relationship_db(tmp_path),
        extra_app={"allow_external_api": True},
    )

    with pytest.raises(SystemExit) as exc:
        cli_main(["--config", str(config_path), "doctor", "--backend", "mock", "--format", "json"])

    assert exc.value.code == 1


def _initialized_relationship_db(tmp_path: Path) -> Path:
    db_path = tmp_path / "relationship.sqlite"
    repo = RelationshipRepository(db_path)
    repo.create_profile("manual profile", relationship_label="partner")
    return db_path


def _write_config(
    tmp_path: Path,
    *,
    relationship_db: Path,
    extra_app: dict[str, object] | None = None,
    extra_paths: dict[str, object] | None = None,
) -> Path:
    tmp_path.mkdir(parents=True, exist_ok=True)
    app = {
        "host": "127.0.0.1",
        "allow_external_api": False,
        "allow_model_auto_download": False,
        "allow_gradio_share": False,
    }
    app.update(extra_app or {})
    paths = {
        "relationship_db": str(relationship_db),
        "personal_lifelog_db": None,
        "notes_lifelog_db": None,
    }
    paths.update(extra_paths or {})
    text = "\n".join(
        [
            "app:",
            f"  host: {app['host']}",
            f"  allow_external_api: {str(app['allow_external_api']).lower()}",
            f"  allow_model_auto_download: {str(app['allow_model_auto_download']).lower()}",
            f"  allow_gradio_share: {str(app['allow_gradio_share']).lower()}",
            "adapter:",
            "  backend: mock",
            "  upstream_access_mode: readonly",
            "  copy_raw_upstream_data: false",
            "paths:",
            f"  relationship_db: {paths['relationship_db']}",
            f"  personal_lifelog_db: {_yaml_nullable(paths['personal_lifelog_db'])}",
            f"  notes_lifelog_db: {_yaml_nullable(paths['notes_lifelog_db'])}",
        ]
    )
    config_path = tmp_path / "config.local.yaml"
    config_path.write_text(text, encoding="utf-8")
    return config_path


def _yaml_nullable(value: object) -> str:
    if value is None:
        return "null"
    return str(value)


def _check_status(report, name: str) -> str:
    return next(check.status for check in report.checks if check.name == name)
