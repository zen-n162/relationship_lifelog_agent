import sqlite3

from relationship_lifelog_agent.adapters.mock import MockRelationshipMemory
from relationship_lifelog_agent.db import RelationshipRepository, initialize_database


def test_schema_can_be_initialized(tmp_path) -> None:
    db_path = initialize_database(tmp_path / "relationship.sqlite")
    assert db_path.exists()
    with sqlite3.connect(db_path) as conn:
        table_names = {
            row[0]
            for row in conn.execute("SELECT name FROM sqlite_master WHERE type = 'table'")
        }
        expected = {
            "relationship_profiles",
            "relationship_events",
            "relationship_event_evidence",
            "interaction_metrics",
            "post_conflict_activities",
            "relationship_review_actions",
        }
        assert expected <= table_names
        for table in expected:
            columns = {row[1] for row in conn.execute(f"PRAGMA table_info({table})")}
            assert "created_at" in columns
            assert "updated_at" in columns
        profile_columns = {row[1] for row in conn.execute("PRAGMA table_info(relationship_profiles)")}
        assert {
            "line_speaker_group_source_id",
            "self_person_source_id",
            "self_line_speaker_source_id",
            "self_line_speaker_group_source_id",
        } <= profile_columns


def test_profile_migration_adds_self_and_group_columns_without_deleting_rows(tmp_path) -> None:
    db_path = tmp_path / "relationship.sqlite"
    with sqlite3.connect(db_path) as conn:
        conn.execute(
            """
            CREATE TABLE relationship_profiles (
              id INTEGER PRIMARY KEY AUTOINCREMENT,
              profile_name TEXT NOT NULL,
              person_source_id TEXT,
              line_speaker_source_id TEXT,
              relationship_label TEXT,
              label_source TEXT NOT NULL DEFAULT 'user_manual',
              valid_from TEXT,
              valid_to TEXT,
              visibility TEXT NOT NULL DEFAULT 'private',
              notes TEXT,
              created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
              updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
              CHECK (relationship_label IS NULL OR relationship_label IN ('partner', 'ex_partner', 'close_person', 'other_private')),
              CHECK (label_source = 'user_manual'),
              CHECK (visibility IN ('private', 'hidden'))
            )
            """
        )
        conn.execute(
            """
            INSERT INTO relationship_profiles (
              profile_name,
              person_source_id,
              line_speaker_source_id,
              relationship_label
            ) VALUES (?, ?, ?, ?)
            """,
            ("manual", "plr:person:target", "plr:line_speaker:target", "partner"),
        )

    initialize_database(db_path)

    with sqlite3.connect(db_path) as conn:
        columns = {row[1] for row in conn.execute("PRAGMA table_info(relationship_profiles)")}
        assert {
            "line_speaker_group_source_id",
            "self_person_source_id",
            "self_line_speaker_source_id",
            "self_line_speaker_group_source_id",
        } <= columns
        rows = conn.execute("SELECT profile_name, person_source_id FROM relationship_profiles").fetchall()
        assert rows == [("manual", "plr:person:target")]


def test_repository_can_save_relationship_event(tmp_path) -> None:
    repo = RelationshipRepository(tmp_path / "relationship.sqlite")
    profile_id = repo.create_profile("manual private profile", relationship_label="partner")
    event = MockRelationshipMemory().search_events(event_type="conflict")[0]
    event_id = repo.save_event(event, profile_id=profile_id)

    assert event_id > 0
    saved = repo.get_event(event_id)
    assert saved is not None
    assert saved["event_type"] == "conflict"
    assert saved["status"] == "candidate"
    assert saved["review_status"] == "unreviewed"
    assert saved["confidence"] == event.confidence
    assert saved["evidence_strength"] == event.evidence_strength

    assert repo.update_event(event_id, review_status="verified", confidence=0.8) == 1
    updated = repo.get_event(event_id)
    assert updated is not None
    assert updated["review_status"] == "verified"
    assert updated["confidence"] == 0.8


def test_repository_can_save_evidence(tmp_path) -> None:
    repo = RelationshipRepository(tmp_path / "relationship.sqlite")
    event_id = repo.create_event(
        event_type="conflict",
        event_date="2025-01-10",
        summary="予定調整のすれ違い候補",
        review_status="candidate",
        confidence=0.55,
        evidence_strength=0.6,
    )
    evidence_id = repo.create_evidence(
        event_id=event_id,
        source_type="note",
        source_id="mock-note",
        source_date="2025-01-10",
        role="primary",
        summary="自分側の反省メモ候補",
        source_pointer="note:mock-note",
        excerpt="短い private excerpt",
        confidence=0.52,
        evidence_strength=0.57,
    )

    saved = repo.get_evidence(evidence_id)
    assert saved is not None
    assert saved["event_id"] == event_id
    assert saved["source_type"] == "note"
    assert saved["source_pointer"] == "note:mock-note"
    assert saved["confidence"] == 0.52
    assert saved["evidence_strength"] == 0.57
    assert repo.list_evidence(event_id=event_id)[0]["id"] == evidence_id


def test_repository_can_save_review_action(tmp_path) -> None:
    repo = RelationshipRepository(tmp_path / "relationship.sqlite")
    event_id = repo.create_event(
        event_type="minor_misunderstanding",
        event_date="2025-01-24",
        summary="軽いすれ違い候補",
    )
    action_id = repo.save_review_action(event_id, "verify", note="manual review in test")

    saved = repo.get_review_action(action_id)
    assert saved is not None
    assert saved["event_id"] == event_id
    assert saved["action"] == "verify"
    assert repo.update_review_action(action_id, action="correct", note="corrected in test") == 1
    assert repo.get_review_action(action_id)["action"] == "correct"


def test_repository_can_save_metrics_and_post_conflict_activity(tmp_path) -> None:
    repo = RelationshipRepository(tmp_path / "relationship.sqlite")
    profile_id = repo.create_profile("manual private profile")
    conflict_id = repo.create_event(
        profile_id=profile_id,
        event_type="conflict",
        event_date="2025-02-14",
        summary="謝罪表現を含む喧嘩候補",
    )
    outing_id = repo.create_event(
        profile_id=profile_id,
        event_type="date_or_outing",
        event_date="2025-02-17",
        summary="外出候補",
    )

    metric_id = repo.create_interaction_metric(
        profile_id=profile_id,
        metric_date="2025-02-14",
        metric_type="reply_delay",
        metric_value=12.5,
        source_pointer="mock-line-metric",
    )
    activity_id = repo.create_post_conflict_activity(
        conflict_event_id=conflict_id,
        activity_event_id=outing_id,
        activity_date="2025-02-17",
        days_after_conflict=3,
        place_label="食事候補",
        activity_type="outing_candidate",
        confidence=0.58,
        evidence_strength=0.62,
    )

    assert repo.get_interaction_metric(metric_id)["metric_type"] == "reply_delay"
    assert repo.get_post_conflict_activity(activity_id)["days_after_conflict"] == 3
    assert repo.update_post_conflict_activity(activity_id, confidence=0.6) == 1
    assert repo.get_post_conflict_activity(activity_id)["confidence"] == 0.6
