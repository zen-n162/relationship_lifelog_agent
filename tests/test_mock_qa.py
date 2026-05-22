from relationship_lifelog_agent.agent.executor import answer_question


def test_mock_conflict_frequency_answer_contains_sections() -> None:
    answer = answer_question("喧嘩はどのくらいしている？")
    for section in ("要約:", "集計:", "具体例:", "根拠:", "信頼度:", "注意:"):
        assert section in answer
    assert "total_candidates" in answer


def test_mock_post_conflict_activity_answer() -> None:
    answer = answer_question("喧嘩の後にどこへ行っている？")
    assert "外出候補" in answer
    assert "仲直りだったとは断定しません" in answer
    assert "根拠:" in answer


def test_mock_emotional_note_lookup_answer() -> None:
    answer = answer_question("その時、自分は何を考えていた？")
    assert "自分側" in answer
    assert "メモ" in answer
    assert "相手の内面" in answer


def test_mock_monthly_review_answer() -> None:
    answer = answer_question("2025年1月の関係を振り返って")
    assert "2025-01" in answer
    assert "月次" in answer
    assert "採点" in answer


def test_public_mode_omits_excerpts() -> None:
    answer = answer_question("喧嘩はどのくらいしている？", mode="public")
    assert "excerpt:" not in answer
    assert "要約:" in answer
