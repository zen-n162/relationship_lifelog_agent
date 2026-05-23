from pathlib import Path
import importlib.util
import sys

import yaml


EVAL_DIR = Path(__file__).resolve().parents[1] / "eval"
_SPEC = importlib.util.spec_from_file_location(
    "relationship_eval_run_full_context_eval",
    EVAL_DIR / "run_full_context_eval.py",
)
assert _SPEC is not None and _SPEC.loader is not None
run_full_context_eval_module = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = run_full_context_eval_module
_SPEC.loader.exec_module(run_full_context_eval_module)

main = run_full_context_eval_module.main
render_json = run_full_context_eval_module.render_json
run_eval = run_full_context_eval_module.run_eval


def test_full_context_eval_runner_passes_mock_cases() -> None:
    result = run_eval(case_path=EVAL_DIR / "full_context_eval_cases.yaml", mock=True)

    assert result.failed == 0
    assert result.passed >= 10
    assert {item.category for item in result.results} >= {
        "compound_question",
        "line_analysis",
        "note_analysis",
        "people_place",
        "full_corpus_summary",
        "safety",
        "source_ref",
    }
    assert all(item.metrics["safety_violation_count"] == 0 for item in result.results if item.status == "pass")


def test_full_context_eval_main_mock_writes_json(tmp_path) -> None:
    output_path = tmp_path / "full_context_eval.json"

    exit_code = main(["--mock", "--format", "json", "--output", str(output_path)])

    assert exit_code == 0
    assert output_path.exists()
    rendered = output_path.read_text(encoding="utf-8")
    assert '"ok": true' in rendered
    assert "task_decomposition_correctness" in rendered


def test_full_context_eval_real_data_cases_require_explicit_flag(tmp_path) -> None:
    case_path = tmp_path / "cases.yaml"
    case_path.write_text(
        yaml.safe_dump(
            {
                "cases": [
                    {
                        "id": "real_data_guard",
                        "category": "source_ref",
                        "question": "実データで全期間を見て",
                        "backend": "upstream_readonly",
                        "analysis_mode": "private_full_corpus",
                    }
                ]
            },
            allow_unicode=True,
        ),
        encoding="utf-8",
    )

    result = run_eval(case_path=case_path, mock=True, real_data=False)

    assert result.failed == 0
    assert result.skipped == 1
    assert result.results[0].metrics["skip_reason"] == "real data eval requires --real-data"


def test_full_context_eval_json_renderer_includes_metrics() -> None:
    result = run_eval(case_path=EVAL_DIR / "full_context_eval_cases.yaml", mock=True, limit=1)

    rendered = render_json(result)

    assert '"ok": true' in rendered
    assert "source_ref_validity" in rendered
