from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import json
from pathlib import Path
import re
import sys
import tempfile
from typing import Any

import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from relationship_lifelog_agent.agent.agent_runtime import RuntimeBudget, run_agent  # noqa: E402
from relationship_lifelog_agent.agent.question_understanding import understand_question  # noqa: E402
from relationship_lifelog_agent.config import (  # noqa: E402
    AdapterSettings,
    AnalysisSettings,
    FullScanSettings,
    LlmSettings,
    PathSettings,
    RelationshipSettings,
    Settings,
    load_config,
)
from relationship_lifelog_agent.db.repository import RelationshipRepository  # noqa: E402
from relationship_lifelog_agent.privacy.guard import detect_answer_safety_violations  # noqa: E402
from relationship_lifelog_agent.verification.full_answer_verifier import verify_final_answer  # noqa: E402


RAW_LEAK_PATTERNS = (
    re.compile(r"LINE全文|line_full_text|raw LINE", re.IGNORECASE),
    re.compile(r"メモ全文|note_full_text|raw note", re.IGNORECASE),
    re.compile(r"/home/[^\s)>'\"]+|~/[^\s)>'\"]+"),
    re.compile(r"\b-?\d{1,3}\.\d{4,}\s*,\s*-?\d{1,3}\.\d{4,}\b"),
    re.compile(r"face[_\s-]*crop|顔\s*crop|embedding:", re.IGNORECASE),
)


@dataclass(frozen=True)
class FullContextEvalCaseResult:
    case_id: str
    category: str
    question: str
    backend: str
    analysis_mode: str
    status: str
    failures: list[str]
    metrics: dict[str, Any]
    answer_preview: str


@dataclass(frozen=True)
class FullContextEvalRunResult:
    passed: int
    failed: int
    skipped: int
    results: list[FullContextEvalCaseResult]

    @property
    def ok(self) -> bool:
        return self.failed == 0


def run_eval(
    *,
    case_path: Path,
    config_path: Path | None = None,
    mock: bool = True,
    real_data: bool = False,
    limit: int | None = None,
) -> FullContextEvalRunResult:
    cases = _load_cases(case_path)
    if limit is not None:
        cases = cases[: max(0, limit)]
    with tempfile.TemporaryDirectory(prefix="relationship_full_context_eval_") as tmp:
        base_settings = _eval_settings(Path(tmp), config_path=config_path, mock=mock)
        results = [
            evaluate_case(case, base_settings=base_settings, mock=mock, real_data=real_data)
            for case in cases
        ]
    return FullContextEvalRunResult(
        passed=sum(1 for result in results if result.status == "pass"),
        failed=sum(1 for result in results if result.status == "fail"),
        skipped=sum(1 for result in results if result.status == "skip"),
        results=results,
    )


def evaluate_case(
    case: dict[str, Any],
    *,
    base_settings: Settings,
    mock: bool = True,
    real_data: bool = False,
) -> FullContextEvalCaseResult:
    backend = str(case.get("backend", "mock"))
    if backend == "upstream_readonly" and not real_data:
        return FullContextEvalCaseResult(
            case_id=str(case["id"]),
            category=str(case.get("category", "uncategorized")),
            question=str(case["question"]),
            backend=backend,
            analysis_mode=str(case.get("analysis_mode", base_settings.analysis.mode)),
            status="skip",
            failures=[],
            metrics={"skip_reason": "real data eval requires --real-data"},
            answer_preview="skipped; real data mode is explicit only",
        )
    if backend != "mock" and mock and not real_data:
        return FullContextEvalCaseResult(
            case_id=str(case["id"]),
            category=str(case.get("category", "uncategorized")),
            question=str(case["question"]),
            backend=backend,
            analysis_mode=str(case.get("analysis_mode", base_settings.analysis.mode)),
            status="skip",
            failures=[],
            metrics={"skip_reason": "non-mock backend skipped in --mock mode"},
            answer_preview="skipped",
        )

    settings = _settings_for_case(base_settings, case, backend=backend)
    question = str(case["question"])
    frame = understand_question(question, settings=settings)
    run = run_agent(
        question,
        None,
        settings,
        selected_profile_id=settings.relationship.default_profile_id,
        date_from=case.get("date_from"),
        date_to=case.get("date_to"),
        mode=str(case.get("mode", "private")),
        budget=RuntimeBudget(
            max_steps=int(case.get("max_steps", 50)),
            max_llm_calls=int(case.get("max_llm_calls", 0)),
            max_tool_calls=int(case.get("max_tool_calls", 50)),
            max_runtime_seconds=float(case.get("max_runtime_seconds", settings.analysis.max_runtime_seconds)),
        ),
    )
    metrics = _metrics_for_case(case, frame, run)
    failures = _failures_for_case(case, metrics, run.answer_markdown)
    return FullContextEvalCaseResult(
        case_id=str(case["id"]),
        category=str(case.get("category", "uncategorized")),
        question=question,
        backend=backend,
        analysis_mode=settings.analysis.mode,
        status="fail" if failures else "pass",
        failures=failures,
        metrics=metrics,
        answer_preview=_preview(run.answer_markdown),
    )


def _metrics_for_case(case: dict[str, Any], frame: Any, run: Any) -> dict[str, Any]:
    answer = run.answer_markdown
    observation_types = [obs.observation_type for obs in run.observations.observations]
    completed_tasks = list(run.completed_task_order)
    source_refs = _collect_source_refs(run)
    verification = verify_final_answer(
        answer,
        manifest=(run.observations.latest("manifest").data if run.observations.latest("manifest") else None),
        observations=run.observations,
        date_from=case.get("date_from"),
        date_to=case.get("date_to"),
        evidence_refs=source_refs,
        computed_facts=_computed_facts(run),
        mode=str(case.get("mode", "private")),
    )
    safety_violations = detect_answer_safety_violations(answer, mode=str(case.get("mode", "private")))
    issue_codes = [issue.code for issue in verification.issues]
    return {
        "task_decomposition_correctness": _task_decomposition_score(case, observation_types, completed_tasks),
        "required_information_identified": int("answerability" in observation_types or "missing_information" in observation_types),
        "source_coverage": len(source_refs),
        "unsupported_claims_count": issue_codes.count("unsupported_claim") + issue_codes.count("unsupported_relationship_assertion"),
        "date_range_violations": issue_codes.count("date_out_of_scope"),
        "source_ref_validity": 0 if any(code == "missing_source_ref" for code in issue_codes) else 1,
        "missing_source_refs": issue_codes.count("missing_source_ref"),
        "safety_violation_count": len(safety_violations) + len(_raw_leak_failures(answer)),
        "answer_completeness": _answer_completeness_score(case, answer),
        "intents": list(frame.intents),
        "primary_intent": frame.primary_intent,
        "observation_types": observation_types,
        "completed_tasks": completed_tasks,
        "run_status": run.status,
        "verification_ok": verification.ok,
        "verification_issue_codes": issue_codes,
        "batch_count": _batch_count(run),
        "llm_calls_used": run.budget.used_llm_calls,
    }


def _failures_for_case(case: dict[str, Any], metrics: dict[str, Any], answer: str) -> list[str]:
    failures: list[str] = []
    for expected in case.get("expected_intents", []) or []:
        if expected not in metrics["intents"]:
            failures.append(f"missing expected intent: {expected}")
    for observation_type in case.get("required_observation_types", []) or []:
        if observation_type not in metrics["observation_types"]:
            failures.append(f"missing observation_type: {observation_type}")
    for section in case.get("required_sections", []) or []:
        if f"{section}:" not in answer:
            failures.append(f"missing required section: {section}")
    for text in [*(case.get("required_substrings", []) or []), *(case.get("expected_answer_substrings", []) or [])]:
        if text not in answer:
            failures.append(f"missing required substring: {text}")
    for text in case.get("forbidden_phrases", []) or []:
        if text in answer:
            failures.append(f"forbidden phrase present: {text}")
    expected_order = list(case.get("expected_task_order", []) or [])
    if expected_order and not _is_subsequence(expected_order, metrics["completed_tasks"]):
        failures.append(f"task order mismatch: expected subsequence {expected_order}")
    if case.get("require_source_refs") and metrics["source_coverage"] <= 0:
        failures.append("source refs required but none found")
    if metrics["date_range_violations"] > int(case.get("max_date_range_violations", 999)):
        failures.append("date range violations exceeded")
    if metrics["missing_source_refs"] > int(case.get("max_missing_source_refs", 999)):
        failures.append("missing source refs exceeded")
    if metrics["unsupported_claims_count"] > int(case.get("max_unsupported_claims", 999)):
        failures.append("unsupported claims exceeded")
    if metrics["safety_violation_count"] > int(case.get("max_safety_violations", 999)):
        failures.append("safety violations exceeded")
    return failures


def _eval_settings(tmp_dir: Path, *, config_path: Path | None, mock: bool) -> Settings:
    loaded = load_config(config_path) if config_path else Settings()
    db_path = tmp_dir / "relationship_full_context_eval.sqlite"
    repo = RelationshipRepository(db_path)
    profile_id = repo.create_profile(
        "いおり",
        person_source_id="plr:person:eval-target",
        line_speaker_source_id="mock-speaker-user",
        self_line_speaker_source_id="mock-speaker-self",
        relationship_label="partner",
    )
    return Settings(
        app=loaded.app,
        adapter=AdapterSettings(backend="mock" if mock else loaded.adapter.backend),
        analysis=AnalysisSettings(
            mode="private_full_range",
            default_scope="auto",
            max_runtime_seconds=loaded.analysis.max_runtime_seconds,
            max_llm_calls=0,
            max_tool_calls=loaded.analysis.max_tool_calls,
            full_scan=FullScanSettings(
                batch_strategy=loaded.analysis.full_scan.batch_strategy,
                max_items_per_run=loaded.analysis.full_scan.max_items_per_run,
                max_items_per_batch=loaded.analysis.full_scan.max_items_per_batch,
                max_chars_per_batch=loaded.analysis.full_scan.max_chars_per_batch,
                overlap_items=loaded.analysis.full_scan.overlap_items,
                max_batches_per_run=loaded.analysis.full_scan.max_batches_per_run,
            ),
        ),
        paths=PathSettings(
            relationship_db=str(db_path),
            personal_lifelog_rag=loaded.paths.personal_lifelog_rag,
            notes_lifelog_rag=loaded.paths.notes_lifelog_rag,
            personal_lifelog_db=loaded.paths.personal_lifelog_db,
            notes_lifelog_db=loaded.paths.notes_lifelog_db,
            personal_lifelog_export_dir=loaded.paths.personal_lifelog_export_dir,
            notes_lifelog_export_dir=loaded.paths.notes_lifelog_export_dir,
        ),
        relationship=RelationshipSettings(
            default_profile_id=profile_id,
            post_conflict_window_days=loaded.relationship.post_conflict_window_days,
            surrounding_media_days_before=loaded.relationship.surrounding_media_days_before,
            surrounding_media_days_after=loaded.relationship.surrounding_media_days_after,
            conflict_min_confidence=loaded.relationship.conflict_min_confidence,
            evidence_min_strength_for_answer=loaded.relationship.evidence_min_strength_for_answer,
            require_manual_partner_label=loaded.relationship.require_manual_partner_label,
        ),
        privacy=loaded.privacy,
        llm=LlmSettings(enabled=False),
    )


def _settings_for_case(settings: Settings, case: dict[str, Any], *, backend: str) -> Settings:
    analysis_mode = str(case.get("analysis_mode", settings.analysis.mode))
    default_scope = str(case.get("date_scope", settings.analysis.default_scope))
    return Settings(
        app=settings.app,
        adapter=AdapterSettings(backend=backend),
        analysis=AnalysisSettings(
            mode=analysis_mode,
            default_scope=default_scope,
            max_runtime_seconds=settings.analysis.max_runtime_seconds,
            max_llm_calls=0,
            max_tool_calls=settings.analysis.max_tool_calls,
            full_scan=settings.analysis.full_scan,
        ),
        paths=settings.paths,
        relationship=settings.relationship,
        privacy=settings.privacy,
        llm=settings.llm,
    )


def _load_cases(path: Path) -> list[dict[str, Any]]:
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    return [dict(case) for case in data.get("cases", [])]


def _task_decomposition_score(case: dict[str, Any], observation_types: list[str], completed_tasks: list[str]) -> int:
    required = list(case.get("required_observation_types", []) or [])
    expected_order = list(case.get("expected_task_order", []) or [])
    required_ok = all(item in observation_types for item in required)
    order_ok = not expected_order or _is_subsequence(expected_order, completed_tasks)
    return int(required_ok and order_ok)


def _answer_completeness_score(case: dict[str, Any], answer: str) -> float:
    checks = [f"{section}:" for section in (case.get("required_sections", []) or [])]
    checks.extend(case.get("required_substrings", []) or [])
    checks.extend(case.get("expected_answer_substrings", []) or [])
    if not checks:
        return 1.0 if answer.strip() else 0.0
    return round(sum(1 for check in checks if check in answer) / len(checks), 3)


def _collect_source_refs(run: Any) -> tuple[str, ...]:
    refs: list[str] = []
    for obs in run.observations.observations:
        refs.extend(obs.source_refs)
        for key in ("source_refs", "relevant_evidence_refs", "evidence_refs"):
            value = obs.data.get(key)
            if isinstance(value, str):
                refs.append(value)
            elif isinstance(value, (list, tuple)):
                refs.extend(str(item) for item in value if item)
    return tuple(dict.fromkeys(refs))


def _computed_facts(run: Any) -> dict[str, int]:
    facts: dict[str, int] = {}
    for observation_type in ("manifest", "full_data_loaded"):
        obs = run.observations.latest(observation_type)
        if not obs:
            continue
        for key, value in obs.data.items():
            if isinstance(value, int):
                facts[key] = value
    return facts


def _batch_count(run: Any) -> int:
    obs = run.observations.latest("batches")
    if not obs:
        return 0
    value = obs.data.get("batch_count", 0)
    return int(value) if isinstance(value, int) else 0


def _raw_leak_failures(answer: str) -> list[str]:
    failures: list[str] = []
    for pattern in RAW_LEAK_PATTERNS:
        match = pattern.search(answer)
        if match:
            failures.append(match.group(0))
    return failures


def _is_subsequence(expected: list[str], actual: list[str]) -> bool:
    cursor = 0
    for item in actual:
        if cursor < len(expected) and item == expected[cursor]:
            cursor += 1
    return cursor == len(expected)


def _preview(answer: str) -> str:
    return " ".join(answer.split())[:240]


def render_markdown(result: FullContextEvalRunResult) -> str:
    lines = [
        "# Full Context Eval Results",
        "",
        f"- passed: {result.passed}",
        f"- failed: {result.failed}",
        f"- skipped: {result.skipped}",
        "",
        "## Metrics",
        "- task_decomposition_correctness",
        "- required_information_identified",
        "- source_coverage",
        "- unsupported_claims_count",
        "- date_range_violations",
        "- source_ref_validity",
        "- safety_violation_count",
        "- answer_completeness",
        "",
    ]
    for item in result.results:
        marker = {"pass": "PASS", "fail": "FAIL", "skip": "SKIP"}[item.status]
        lines.append(f"## {marker} {item.case_id}")
        lines.append(f"- category: {item.category}")
        lines.append(f"- backend: {item.backend}")
        lines.append(f"- analysis_mode: {item.analysis_mode}")
        lines.append(f"- primary_intent: {item.metrics.get('primary_intent', '')}")
        lines.append(f"- source_coverage: {item.metrics.get('source_coverage', 0)}")
        lines.append(f"- answer_completeness: {item.metrics.get('answer_completeness', 0)}")
        if item.failures:
            lines.append("- failures:")
            lines.extend(f"  - {failure}" for failure in item.failures)
        lines.append(f"- answer_preview: {item.answer_preview}")
        lines.append("")
    return "\n".join(lines)


def render_json(result: FullContextEvalRunResult) -> str:
    return json.dumps(
        {
            "ok": result.ok,
            "passed": result.passed,
            "failed": result.failed,
            "skipped": result.skipped,
            "results": [asdict(item) for item in result.results],
        },
        ensure_ascii=False,
        indent=2,
    )


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run private full-context evals.")
    parser.add_argument("--cases", default=str(Path(__file__).resolve().parent / "full_context_eval_cases.yaml"))
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--mock", action="store_true", help="Run with mock adapters and no local LLM calls.")
    mode.add_argument("--real-data", action="store_true", help="Explicitly allow upstream_readonly eval cases.")
    parser.add_argument("--config", default=None, help="Optional local config for real-data mode.")
    parser.add_argument("--format", choices=("markdown", "json"), default="markdown")
    parser.add_argument("--output", default=None)
    parser.add_argument("--limit", type=int, default=None)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    mock = bool(args.mock or not args.real_data)
    result = run_eval(
        case_path=Path(args.cases),
        config_path=Path(args.config) if args.config else None,
        mock=mock,
        real_data=bool(args.real_data),
        limit=args.limit,
    )
    rendered = render_json(result) if args.format == "json" else render_markdown(result)
    if args.output:
        output_path = Path(args.output).expanduser()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(rendered, encoding="utf-8")
    print(rendered)
    return 0 if result.ok else 1


if __name__ == "__main__":
    sys.exit(main())
