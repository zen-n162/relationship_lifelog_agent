from __future__ import annotations

from pathlib import Path
import sys

import yaml

from relationship_lifelog_agent.agent.executor import answer_question


def run_case_file(path: Path) -> int:
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    failures = 0
    for case in data.get("cases", []):
        answer = answer_question(case["question"], mode=case.get("mode", "private"))
        if not answer.strip():
            failures += 1
            print(f"FAIL {case['id']}: empty answer")
        else:
            print(f"PASS {case['id']}")
    return failures


def main() -> int:
    base = Path(__file__).resolve().parent
    failures = run_case_file(base / "private_eval_cases.yaml")
    failures += run_case_file(base / "safety_eval_cases.yaml")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
