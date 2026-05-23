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
            "full_analysis_runs",
            "full_analysis_batches",
            "full_analysis_observations",
            "llm_analysis_cache",
        }
        assert expected <= table_names
        for table in expected - {"full_analysis_observations"}:
            columns = {row[1] for row in conn.execute(f"PRAGMA table_info({table})")}
            assert "created_at" in columns
            assert "updated_at" in columns
        observation_columns = {row[1] for row in conn.execute("PRAGMA table_info(full_analysis_observations)")}
        assert "created_at" in observation_columns
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


def test_repository_can_save_full_analysis_run_batch_and_observation(tmp_path) -> None:
    repo = RelationshipRepository(tmp_path / "relationship.sqlite")
    profile_id = repo.create_profile("manual private profile")
    run_id = repo.create_full_analysis_run(
        question="この期間に何があった？",
        analysis_mode="private_full_range",
        profile_id=profile_id,
        date_from="2025-01-01",
        date_to="2025-01-31",
        status="running",
        model_name="qwen3.6:27b",
        manifest={"line_count": 2, "source_refs": ["line:1", "line:2"]},
    )
    batch_id = repo.create_full_analysis_batch(
        run_id=run_id,
        batch_index=0,
        source_types=("line", "note"),
        date_from="2025-01-01",
        date_to="2025-01-07",
        item_count=2,
        source_refs=("line:1", "note:1"),
        input_hash="hash-1",
        output={"key_findings": ["finding"], "source_refs": ["line:1"]},
        status="succeeded",
    )
    observation_id = repo.create_full_analysis_observation(
        run_id=run_id,
        batch_id=batch_id,
        observation_type="batch_finding",
        summary="structured finding only",
        source_refs=("line:1",),
        result={"summary": "finding", "source_refs": ["line:1"]},
        confidence=0.8,
    )

    saved_run = repo.get_full_analysis_run(run_id)
    assert saved_run is not None
    assert saved_run["analysis_mode"] == "private_full_range"
    assert saved_run["manifest"]["line_count"] == 2
    saved_batch = repo.get_full_analysis_batch(batch_id)
    assert saved_batch is not None
    assert saved_batch["source_types"] == ["line", "note"]
    assert saved_batch["source_refs"] == ["line:1", "note:1"]
    assert saved_batch["output"]["key_findings"] == ["finding"]
    saved_observation = repo.get_full_analysis_observation(observation_id)
    assert saved_observation is not None
    assert saved_observation["source_refs"] == ["line:1"]
    assert saved_observation["result"]["summary"] == "finding"
    assert repo.list_full_analysis_runs(profile_id=profile_id)[0]["id"] == run_id
    assert repo.list_full_analysis_batches(run_id=run_id)[0]["id"] == batch_id
    assert repo.list_full_analysis_observations(run_id=run_id)[0]["id"] == observation_id
    assert repo.update_full_analysis_run(run_id, status="succeeded", final_synthesis={"answer": "ok"}) == 1
    assert repo.get_full_analysis_run(run_id)["final_synthesis"]["answer"] == "ok"


def test_repository_can_upsert_llm_analysis_cache_without_raw_payload(tmp_path) -> None:
    repo = RelationshipRepository(tmp_path / "relationship.sqlite")
    cache_id = repo.upsert_llm_analysis_cache(
        analysis_type="full_batch",
        source_window_hash="hash-1",
        model_name="qwen3.6:27b",
        prompt_version="full-v1",
        result={"summary": "structured result", "source_refs": ["line:1"]},
        confidence=0.7,
    )
    same_id = repo.upsert_llm_analysis_cache(
        analysis_type="full_batch",
        source_window_hash="hash-1",
        model_name="qwen3.6:27b",
        prompt_version="full-v1",
        result={"summary": "updated structured result", "source_refs": ["line:1"]},
        confidence=0.8,
    )

    assert same_id == cache_id
    cached = repo.get_llm_analysis_cache(
        analysis_type="full_batch",
        source_window_hash="hash-1",
        model_name="qwen3.6:27b",
        prompt_version="full-v1",
    )
    assert cached is not None
    assert cached["result"]["summary"] == "updated structured result"
    assert cached["confidence"] == 0.8


def test_full_analysis_repository_rejects_raw_payload_fields(tmp_path) -> None:
    repo = RelationshipRepository(tmp_path / "relationship.sqlite")
    run_id = repo.create_full_analysis_run(
        question="safe structured save",
        analysis_mode="private_full_range",
        manifest={"line_count": 1},
    )

    for raw_result in (
        {"raw_prompt": "do not store"},
        {"raw_payload": {"x": "do not store"}},
        {"raw_text": "raw LINE"},
        {"raw_body": "raw note"},
        {"file_path": "/private/photo.jpg"},
        {"exact_gps": {"lat": 35.0, "lon": 139.0}},
        {"face_embeddings": [[0.1, 0.2]]},
    ):
        try:
            repo.create_full_analysis_observation(
                run_id=run_id,
                observation_type="unsafe",
                summary="unsafe",
                result=raw_result,
            )
        except ValueError as exc:
            assert "raw payload field is not allowed" in str(exc)
        else:
            raise AssertionError(f"raw payload was accepted: {raw_result}")


def test_existing_db_migration_adds_full_analysis_tables_without_deleting_profiles(tmp_path) -> None:
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
        conn.execute("INSERT INTO relationship_profiles (profile_name) VALUES ('existing')")

    initialize_database(db_path)

    with sqlite3.connect(db_path) as conn:
        table_names = {row[0] for row in conn.execute("SELECT name FROM sqlite_master WHERE type = 'table'")}
        assert {
            "full_analysis_runs",
            "full_analysis_batches",
            "full_analysis_observations",
            "llm_analysis_cache",
        } <= table_names
        rows = conn.execute("SELECT profile_name FROM relationship_profiles").fetchall()
        assert rows == [("existing",)]
