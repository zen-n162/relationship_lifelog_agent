from relationship_lifelog_agent.agent.executor import answer_question
from relationship_lifelog_agent.privacy.guard import (
    assert_answer_safe,
    detect_answer_safety_violations,
    detect_forbidden_phrases,
    sanitize_answer,
)
from relationship_lifelog_agent.privacy.policies import FORBIDDEN_PHRASES


def test_forbidden_phrases_are_detected() -> None:
    unsafe = " / ".join(FORBIDDEN_PHRASES)
    detected = detect_forbidden_phrases(unsafe)

    for phrase in FORBIDDEN_PHRASES:
        assert phrase in detected


def test_forbidden_phrases_are_rewritten() -> None:
    unsafe = "確実に喧嘩していた。相手は怒っていた。あなたが悪い。"
    safe = sanitize_answer(unsafe)
    for phrase in ("確実に喧嘩していた", "相手は怒っていた", "あなたが悪い"):
        assert phrase not in safe
    assert "喧嘩候補" in safe
    assert "断定できません" in safe
    assert detect_answer_safety_violations(safe) == []


def test_inner_feeling_assertion_is_rewritten_even_in_private_mode() -> None:
    unsafe = "相手は悲しんでいたので、関係は終わっていた。"
    safe = sanitize_answer(unsafe, mode="private")

    assert "相手は悲しんでいた" not in safe
    assert "関係は終わっていた" not in safe
    assert "相手の内心は記録から断定できません" in safe
    assert_answer_safe(safe)


def test_generated_answer_uses_candidate_language() -> None:
    answer = answer_question("喧嘩はどのくらいしている？")
    assert "喧嘩候補" in answer
    assert "注意:" in answer
    for phrase in FORBIDDEN_PHRASES:
        assert phrase not in answer
    assert_answer_safe(answer)


def test_public_generated_answer_has_no_public_violations() -> None:
    answer = answer_question("喧嘩の後どこへ行っている？", mode="public")
    assert "喧嘩候補" in answer
    assert "相手の気持ちは" in answer
    assert detect_answer_safety_violations(answer, mode="public") == []


def test_weak_evidence_produces_caution() -> None:
    answer = answer_question("喧嘩はどのくらいしている？")
    assert "弱い信号だけでは喧嘩とは扱いません" in answer
