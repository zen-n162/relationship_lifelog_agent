from relationship_lifelog_agent.agent.executor import answer_chat, answer_question
from relationship_lifelog_agent.config import AdapterSettings, PathSettings, RelationshipSettings, Settings
from relationship_lifelog_agent.db.repository import RelationshipRepository
from relationship_lifelog_agent.privacy.guard import detect_answer_safety_violations
from relationship_lifelog_agent.review.actions import apply_review_action


def test_apply_review_action_saves_history_and_updates_status(tmp_path) -> None:
    repo, event_id = _repo_with_event(tmp_path)

    result = apply_review_action(repo, event_id=event_id, action="verify", note="checked by user")

    assert result.review_status == "verified"
    assert result.status == "candidate"
    assert repo.get_event(event_id)["review_status"] == "verified"
    actions = repo.list_review_actions(event_id=event_id)
    assert len(actions) == 1
    assert actions[0]["action"] == "verify"
    assert actions[0]["note"] == "checked by user"


def test_mark_as_misunderstanding_updates_event_type(tmp_path) -> None:
    repo, event_id = _repo_with_event(tmp_path)

    apply_review_action(repo, event_id=event_id, action="mark_as_misunderstanding")

    event = repo.get_event(event_id)
    assert event["event_type"] == "minor_misunderstanding"
    assert event["review_status"] == "corrected"
    assert event["severity"] == 1


def test_mark_as_misunderstanding_is_counted_as_minor_candidate(tmp_path) -> None:
    repo, event_id = _repo_with_event(tmp_path)
    settings = _settings(tmp_path)

    apply_review_action(repo, event_id=event_id, action="mark_as_misunderstanding")
    answer = answer_question(
        "喧嘩はどのくらいしている？",
        settings=settings,
        profile_id=1,
        date_from="2025-04-01",
        date_to="2025-04-30",
    )

    assert "conflict_candidates: 0" in answer
    assert "minor_misunderstanding_candidates: 1" in answer
    assert "人間確認済み件数: 1" in answer
    assert "軽いすれ違い候補" in answer


def test_rejected_event_is_excluded_from_normal_answer(tmp_path) -> None:
    repo, event_id = _repo_with_event(tmp_path)
    settings = _settings(tmp_path)

    before = answer_question(
        "喧嘩はどのくらいしている？",
        settings=settings,
        profile_id=1,
        date_from="2025-04-01",
        date_to="2025-04-30",
    )
    apply_review_action(repo, event_id=event_id, action="reject")
    after = answer_question(
        "喧嘩はどのくらいしている？",
        settings=settings,
        profile_id=1,
        date_from="2025-04-01",
        date_to="2025-04-30",
    )

    assert "保存済みの喧嘩候補" in before
    assert "保存済みの喧嘩候補" not in after
    assert "total_candidates: 0" in after
    assert "除外済み件数: 1" in after


def test_mark_as_joke_is_excluded_from_conflict_frequency(tmp_path) -> None:
    repo, event_id = _repo_with_event(tmp_path)
    settings = _settings(tmp_path)

    apply_review_action(repo, event_id=event_id, action="mark_as_joke")
    answer = answer_question(
        "喧嘩はどのくらいしている？",
        settings=settings,
        profile_id=1,
        date_from="2025-04-01",
        date_to="2025-04-30",
    )

    assert "保存済みの喧嘩候補" not in answer
    assert "total_candidates: 0" in answer
    assert "除外済み件数: 1" in answer


def test_verified_event_is_labeled_as_human_verified(tmp_path) -> None:
    repo, event_id = _repo_with_event(tmp_path)
    settings = _settings(tmp_path)

    apply_review_action(repo, event_id=event_id, action="verify")
    answer = answer_question(
        "喧嘩はどのくらいしている？",
        settings=settings,
        profile_id=1,
        date_from="2025-04-01",
        date_to="2025-04-30",
    )

    assert "人間確認済み" in answer
    assert "人間確認済み件数: 1" in answer
    assert "AI推定のみ件数: 0" in answer


def test_needs_reanalysis_event_is_counted_and_cautioned(tmp_path) -> None:
    repo, event_id = _repo_with_event(tmp_path)
    settings = _settings(tmp_path)

    result = apply_review_action(repo, event_id=event_id, action="needs_reanalysis")
    answer = answer_question(
        "喧嘩はどのくらいしている？",
        settings=settings,
        profile_id=1,
        date_from="2025-04-01",
        date_to="2025-04-30",
    )

    assert result.review_status == "needs_reanalysis"
    assert "要再分析" in answer
    assert "要再分析件数: 1" in answer
    assert "再抽出" in answer


def test_chat_answer_includes_public_safe_review_targets(tmp_path) -> None:
    repo, event_id = _repo_with_event(tmp_path, profile_name="山田太郎", summary="山田太郎との喧嘩候補。")
    settings = _settings(tmp_path)
    apply_review_action(repo, event_id=event_id, action="needs_reanalysis")

    chat_answer = answer_chat(
        "喧嘩はどのくらいしている？",
        mode="public",
        settings=settings,
        profile_id=1,
        date_from="2025-04-01",
        date_to="2025-04-30",
    )

    assert chat_answer.review_targets
    assert "山田太郎" not in chat_answer.text
    assert all("山田太郎" not in target.label for target in chat_answer.review_targets)
    assert detect_answer_safety_violations(chat_answer.text, mode="public", person_names=["山田太郎"]) == []


def _repo_with_event(tmp_path, *, profile_name: str = "Aさん", summary: str = "保存済みの喧嘩候補。"):
    repo = RelationshipRepository(tmp_path / "relationship.sqlite")
    profile_id = repo.create_profile(profile_name, relationship_label="partner")
    event_id = repo.create_event(
        profile_id=profile_id,
        event_type="conflict",
        event_date="2025-04-10",
        summary=summary,
        status="candidate",
        review_status="unreviewed",
        confidence=0.62,
        evidence_strength=0.58,
        severity=2,
    )
    repo.create_evidence(
        event_id=event_id,
        source_type="manual",
        source_id="review-test",
        source_pointer="manual:review-test",
        summary="保存済みcandidateのsource pointer。",
        confidence=0.6,
        evidence_strength=0.58,
    )
    return repo, event_id


def _settings(tmp_path) -> Settings:
    return Settings(
        adapter=AdapterSettings(backend="upstream_readonly"),
        paths=PathSettings(relationship_db=str(tmp_path / "relationship.sqlite")),
        relationship=RelationshipSettings(default_profile_id=1),
    )
