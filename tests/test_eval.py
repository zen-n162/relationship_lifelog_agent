from pathlib import Path
import importlib.util
import sys


EVAL_DIR = Path(__file__).resolve().parents[1] / "eval"
_SPEC = importlib.util.spec_from_file_location("relationship_eval_run_eval", EVAL_DIR / "run_eval.py")
assert _SPEC is not None and _SPEC.loader is not None
run_eval_module = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = run_eval_module
_SPEC.loader.exec_module(run_eval_module)

evaluate_answer_text = run_eval_module.evaluate_answer_text
main = run_eval_module.main
render_json = run_eval_module.render_json
run_eval = run_eval_module.run_eval


def test_eval_runner_passes_default_mock_cases() -> None:
    result = run_eval(
        case_paths=[
            EVAL_DIR / "private_eval_cases.yaml",
            EVAL_DIR / "safety_eval_cases.yaml",
        ]
    )

    assert result.failed == 0
    assert result.passed >= 8
    assert result.skipped >= 1


def test_eval_detects_forbidden_phrase() -> None:
    failures = evaluate_answer_text(
        "要約:\n相手は冷めていた。\n\n注意:\n- none",
        case={
            "id": "forbidden_fixture",
            "mode": "private",
            "forbidden_phrases": ["相手は冷めていた"],
            "required_sections": ["要約", "注意"],
        },
        intents=["general_relationship_qa"],
    )

    assert any("forbidden phrase present" in failure for failure in failures)
    assert any("safety violation" in failure for failure in failures)


def test_eval_main_can_write_json_result(tmp_path) -> None:
    output_path = tmp_path / "eval_result.json"

    exit_code = main(["--format", "json", "--output", str(output_path)])

    assert exit_code == 0
    assert output_path.exists()
    assert '"ok": true' in output_path.read_text(encoding="utf-8")


def test_eval_json_renderer_reports_failure() -> None:
    result = run_eval(case_paths=[EVAL_DIR / "private_eval_cases.yaml"])
    rendered = render_json(result)

    assert '"ok": true' in rendered
    assert '"failed": 0' in rendered
