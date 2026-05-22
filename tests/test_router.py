from relationship_lifelog_agent.agent.router import route_question


def test_japanese_conflict_frequency_routes() -> None:
    route = route_question("喧嘩はどのくらいしている？")
    assert route.intents == ("conflict_frequency",)


def test_post_conflict_activity_routes() -> None:
    route = route_question("喧嘩の後にどこへ行っている？")
    assert "post_conflict_activity" in route.intents


def test_monthly_review_routes() -> None:
    route = route_question("2025年1月の関係を振り返って")
    assert "monthly_relationship_review" in route.intents


def test_multiple_intents_are_allowed() -> None:
    route = route_question("喧嘩の後、LINEを返せていなかった時期は？")
    assert "post_conflict_activity" in route.intents
    assert "reply_delay_analysis" in route.intents


def test_unknown_question_falls_back_to_general_qa() -> None:
    route = route_question("今日は何を確認できますか？")
    assert route.intents == ("general_relationship_qa",)
