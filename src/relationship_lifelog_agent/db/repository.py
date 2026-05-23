from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
import json
import sqlite3
from typing import Any

from relationship_lifelog_agent.adapters.types import (
    EvidenceItem,
    PostConflictActivity,
    RelationshipEvent,
)
from relationship_lifelog_agent.db.migrate import initialize_database


RowDict = dict[str, Any]
ALLOWED_RELATIONSHIP_LABELS = frozenset({"partner", "ex_partner", "close_person", "other_private"})
MANUAL_LABEL_SOURCE = "user_manual"
ALLOWED_EVENT_STATUSES = frozenset({"candidate", "hidden", "archived"})
ALLOWED_REVIEW_STATUSES = frozenset({"unreviewed", "verified", "corrected", "needs_reanalysis", "rejected"})
ALLOWED_FULL_ANALYSIS_MODES = frozenset({"private_full_range", "private_full_corpus"})
ALLOWED_FULL_RUN_STATUSES = frozenset({"pending", "running", "succeeded", "failed", "cancelled"})
ALLOWED_FULL_BATCH_STATUSES = frozenset({"pending", "running", "succeeded", "failed", "skipped"})
FORBIDDEN_RAW_JSON_KEYS = frozenset(
    {
        "raw_prompt",
        "prompt_text",
        "raw_payload",
        "raw_line_text",
        "raw_note_text",
        "raw_text",
        "raw_body",
        "line_full_text",
        "note_full_text",
        "exact_gps",
        "photo_path",
        "photo_paths",
        "file_path",
        "thumbnail_path",
        "source_path",
        "private_path",
        "private_file_path",
        "face_crop",
        "face_crop_path",
        "face_crop_paths",
        "embedding",
        "face_embedding",
        "face_embeddings",
        "raw_face_embedding_values",
    }
)


class RelationshipRepository:
    """CRUD access for the relationship-local SQLite database only."""

    def __init__(self, db_path: str | Path) -> None:
        self.db_path = initialize_database(db_path)

    def connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys = ON")
        return conn

    # relationship_profiles

    def create_profile(
        self,
        profile_name: str,
        relationship_label: str | None = None,
        *,
        person_source_id: str | None = None,
        line_speaker_source_id: str | None = None,
        line_speaker_group_source_id: str | None = None,
        self_person_source_id: str | None = None,
        self_line_speaker_source_id: str | None = None,
        self_line_speaker_group_source_id: str | None = None,
        label_source: str = MANUAL_LABEL_SOURCE,
        valid_from: str | None = None,
        valid_to: str | None = None,
        visibility: str = "private",
        notes: str | None = None,
    ) -> int:
        _validate_profile_name(profile_name)
        _validate_relationship_label(relationship_label)
        _validate_label_source(label_source)
        values = {
            "profile_name": profile_name,
            "person_source_id": person_source_id,
            "line_speaker_source_id": line_speaker_source_id,
            "line_speaker_group_source_id": line_speaker_group_source_id,
            "self_person_source_id": self_person_source_id,
            "self_line_speaker_source_id": self_line_speaker_source_id,
            "self_line_speaker_group_source_id": self_line_speaker_group_source_id,
            "relationship_label": relationship_label,
            "label_source": label_source,
            "valid_from": valid_from,
            "valid_to": valid_to,
            "visibility": visibility,
            "notes": notes,
        }
        with self.connect() as conn:
            return self._insert(conn, "relationship_profiles", values)

    def get_profile(self, profile_id: int) -> RowDict | None:
        return self._get_by_id("relationship_profiles", profile_id)

    def get_profile_by_name(self, profile_name: str) -> RowDict | None:
        rows = self._select_many(
            "relationship_profiles",
            ["profile_name = ?"],
            [profile_name],
            order_by="id",
            limit=1,
        )
        return rows[0] if rows else None

    def find_active_duplicate_profiles(
        self,
        *,
        profile_name: str,
        relationship_label: str | None,
        visibility: str = "private",
        limit: int = 20,
    ) -> list[RowDict]:
        clauses = ["profile_name = ?", "visibility = ?"]
        params: list[Any] = [profile_name, visibility]
        if relationship_label is None:
            clauses.append("relationship_label IS NULL")
        else:
            clauses.append("relationship_label = ?")
            params.append(relationship_label)
        return self._select_many("relationship_profiles", clauses, params, order_by="id", limit=limit)

    def list_profiles(self, *, visibility: str | None = None, limit: int = 100) -> list[RowDict]:
        clauses: list[str] = []
        params: list[Any] = []
        if visibility:
            clauses.append("visibility = ?")
            params.append(visibility)
        return self._select_many("relationship_profiles", clauses, params, order_by="id", limit=limit)

    def update_profile(self, profile_id: int, **fields: Any) -> int:
        allowed = {
            "profile_name",
            "person_source_id",
            "line_speaker_source_id",
            "line_speaker_group_source_id",
            "self_person_source_id",
            "self_line_speaker_source_id",
            "self_line_speaker_group_source_id",
            "relationship_label",
            "label_source",
            "valid_from",
            "valid_to",
            "visibility",
            "notes",
        }
        if "profile_name" in fields:
            _validate_profile_name(fields["profile_name"])
        if "relationship_label" in fields:
            _validate_relationship_label(fields["relationship_label"])
        if "label_source" in fields:
            _validate_label_source(fields["label_source"])
        return self._update("relationship_profiles", profile_id, allowed, fields)

    def delete_profile(self, profile_id: int) -> int:
        return self._delete("relationship_profiles", profile_id)

    # relationship_events

    def create_event(
        self,
        *,
        event_type: str,
        event_date: str,
        summary: str,
        profile_id: int | None = None,
        status: str = "candidate",
        review_status: str = "unreviewed",
        confidence: float = 0.5,
        evidence_strength: float = 0.5,
        severity: int = 0,
        generated_by_model: str | None = None,
        prompt_version: str | None = None,
    ) -> int:
        values = {
            "profile_id": profile_id,
            "event_type": event_type,
            "event_date": event_date,
            "summary": summary,
            "status": _normalize_event_status(status),
            "review_status": _normalize_review_status(review_status),
            "confidence": confidence,
            "evidence_strength": evidence_strength,
            "severity": severity,
            "generated_by_model": generated_by_model,
            "prompt_version": prompt_version,
        }
        with self.connect() as conn:
            return self._insert(conn, "relationship_events", values)

    def save_event(self, event: RelationshipEvent, profile_id: int | None = None) -> int:
        values = {
            "profile_id": profile_id,
            "event_type": event.event_type,
            "event_date": event.date,
            "summary": event.summary,
            "status": _normalize_event_status(event.metadata.get("status", "candidate")),
            "review_status": _normalize_review_status(event.review_status),
            "confidence": event.confidence,
            "evidence_strength": event.evidence_strength,
            "severity": event.severity,
        }
        with self.connect() as conn:
            event_id = self._insert(conn, "relationship_events", values)
            for item in event.evidence:
                self.save_evidence(item, event_id=event_id, conn=conn)
            return event_id

    def get_event(self, event_id: int) -> RowDict | None:
        return self._get_by_id("relationship_events", event_id)

    def list_events(
        self,
        *,
        profile_id: int | None = None,
        event_type: str | None = None,
        status: str | None = None,
        review_status: str | None = None,
        date_from: str | None = None,
        date_to: str | None = None,
        limit: int = 100,
    ) -> list[RowDict]:
        clauses: list[str] = []
        params: list[Any] = []
        if profile_id is not None:
            clauses.append("profile_id = ?")
            params.append(profile_id)
        if event_type:
            clauses.append("event_type = ?")
            params.append(event_type)
        if status:
            clauses.append("status = ?")
            params.append(status)
        if review_status:
            clauses.append("review_status = ?")
            params.append(review_status)
        if date_from:
            clauses.append("event_date >= ?")
            params.append(date_from)
        if date_to:
            clauses.append("event_date <= ?")
            params.append(date_to)
        return self._select_many("relationship_events", clauses, params, order_by="event_date, id", limit=limit)

    def update_event(self, event_id: int, **fields: Any) -> int:
        allowed = {
            "profile_id",
            "event_type",
            "event_date",
            "summary",
            "status",
            "review_status",
            "confidence",
            "evidence_strength",
            "severity",
            "generated_by_model",
            "prompt_version",
        }
        if "status" in fields:
            fields = {**fields, "status": _normalize_event_status(fields["status"])}
        if "review_status" in fields:
            fields = {**fields, "review_status": _normalize_review_status(fields["review_status"])}
        return self._update("relationship_events", event_id, allowed, fields)

    def delete_event(self, event_id: int) -> int:
        return self._delete("relationship_events", event_id)

    # relationship_event_evidence

    def create_evidence(
        self,
        *,
        event_id: int,
        source_type: str,
        source_id: str,
        summary: str,
        source_pointer: str | None = None,
        source_date: str | None = None,
        role: str = "supporting",
        excerpt: str | None = None,
        confidence: float = 0.5,
        evidence_strength: float = 0.5,
    ) -> int:
        values = {
            "event_id": event_id,
            "source_type": source_type,
            "source_id": source_id,
            "source_pointer": source_pointer or f"{source_type}:{source_id}",
            "source_date": source_date,
            "role": role,
            "summary": summary,
            "excerpt": excerpt,
            "confidence": confidence,
            "evidence_strength": evidence_strength,
        }
        with self.connect() as conn:
            return self._insert(conn, "relationship_event_evidence", values)

    def save_evidence(
        self,
        evidence: EvidenceItem,
        event_id: int,
        conn: sqlite3.Connection | None = None,
    ) -> int:
        values = {
            "event_id": event_id,
            "source_type": evidence.source_type,
            "source_id": evidence.source_id,
            "source_pointer": evidence.source_pointer or f"{evidence.source_type}:{evidence.source_id}",
            "source_date": evidence.date,
            "role": evidence.role,
            "summary": evidence.summary,
            "excerpt": evidence.excerpt,
            "confidence": evidence.confidence,
            "evidence_strength": evidence.evidence_strength,
        }
        if conn is not None:
            return self._insert(conn, "relationship_event_evidence", values)
        with self.connect() as owned_conn:
            return self._insert(owned_conn, "relationship_event_evidence", values)

    def get_evidence(self, evidence_id: int) -> RowDict | None:
        return self._get_by_id("relationship_event_evidence", evidence_id)

    def list_evidence(self, *, event_id: int, limit: int = 100) -> list[RowDict]:
        return self._select_many(
            "relationship_event_evidence",
            ["event_id = ?"],
            [event_id],
            order_by="id",
            limit=limit,
        )

    def update_evidence(self, evidence_id: int, **fields: Any) -> int:
        allowed = {
            "event_id",
            "source_type",
            "source_id",
            "source_pointer",
            "source_date",
            "role",
            "summary",
            "excerpt",
            "confidence",
            "evidence_strength",
        }
        return self._update("relationship_event_evidence", evidence_id, allowed, fields)

    def delete_evidence(self, evidence_id: int) -> int:
        return self._delete("relationship_event_evidence", evidence_id)

    # interaction_metrics

    def create_interaction_metric(
        self,
        *,
        metric_date: str,
        metric_type: str,
        metric_value: float,
        profile_id: int | None = None,
        source_pointer: str | None = None,
    ) -> int:
        values = {
            "profile_id": profile_id,
            "metric_date": metric_date,
            "metric_type": metric_type,
            "metric_value": metric_value,
            "source_pointer": source_pointer,
        }
        with self.connect() as conn:
            return self._insert(conn, "interaction_metrics", values)

    def get_interaction_metric(self, metric_id: int) -> RowDict | None:
        return self._get_by_id("interaction_metrics", metric_id)

    def list_interaction_metrics(
        self,
        *,
        profile_id: int | None = None,
        metric_type: str | None = None,
        limit: int = 100,
    ) -> list[RowDict]:
        clauses: list[str] = []
        params: list[Any] = []
        if profile_id is not None:
            clauses.append("profile_id = ?")
            params.append(profile_id)
        if metric_type:
            clauses.append("metric_type = ?")
            params.append(metric_type)
        return self._select_many("interaction_metrics", clauses, params, order_by="metric_date, id", limit=limit)

    def update_interaction_metric(self, metric_id: int, **fields: Any) -> int:
        allowed = {"profile_id", "metric_date", "metric_type", "metric_value", "source_pointer"}
        return self._update("interaction_metrics", metric_id, allowed, fields)

    def delete_interaction_metric(self, metric_id: int) -> int:
        return self._delete("interaction_metrics", metric_id)

    # post_conflict_activities

    def create_post_conflict_activity(
        self,
        *,
        activity_date: str,
        days_after_conflict: int,
        activity_type: str,
        conflict_event_id: int | None = None,
        activity_event_id: int | None = None,
        place_label: str | None = None,
        confidence: float = 0.5,
        evidence_strength: float = 0.5,
    ) -> int:
        values = {
            "conflict_event_id": conflict_event_id,
            "activity_event_id": activity_event_id,
            "activity_date": activity_date,
            "days_after_conflict": days_after_conflict,
            "place_label": place_label,
            "activity_type": activity_type,
            "confidence": confidence,
            "evidence_strength": evidence_strength,
        }
        with self.connect() as conn:
            return self._insert(conn, "post_conflict_activities", values)

    def save_post_conflict_activity(self, activity: PostConflictActivity) -> int:
        return self.create_post_conflict_activity(
            conflict_event_id=_int_or_none(activity.conflict_event_id),
            activity_event_id=_int_or_none(activity.activity_event_id),
            activity_date=activity.date,
            days_after_conflict=activity.days_after_conflict,
            place_label=activity.place_label,
            activity_type=activity.activity_type,
            confidence=activity.confidence,
            evidence_strength=activity.evidence_strength,
        )

    def get_post_conflict_activity(self, activity_id: int) -> RowDict | None:
        return self._get_by_id("post_conflict_activities", activity_id)

    def list_post_conflict_activities(
        self,
        *,
        conflict_event_id: int | None = None,
        limit: int = 100,
    ) -> list[RowDict]:
        clauses: list[str] = []
        params: list[Any] = []
        if conflict_event_id is not None:
            clauses.append("conflict_event_id = ?")
            params.append(conflict_event_id)
        return self._select_many("post_conflict_activities", clauses, params, order_by="activity_date, id", limit=limit)

    def update_post_conflict_activity(self, activity_id: int, **fields: Any) -> int:
        allowed = {
            "conflict_event_id",
            "activity_event_id",
            "activity_date",
            "days_after_conflict",
            "place_label",
            "activity_type",
            "confidence",
            "evidence_strength",
        }
        return self._update("post_conflict_activities", activity_id, allowed, fields)

    def delete_post_conflict_activity(self, activity_id: int) -> int:
        return self._delete("post_conflict_activities", activity_id)

    # relationship_review_actions

    def create_review_action(self, *, event_id: int | None, action: str, note: str | None = None) -> int:
        values = {"event_id": event_id, "action": action, "note": note}
        with self.connect() as conn:
            return self._insert(conn, "relationship_review_actions", values)

    def save_review_action(self, event_id: int, action: str, note: str | None = None) -> int:
        return self.create_review_action(event_id=event_id, action=action, note=note)

    def get_review_action(self, action_id: int) -> RowDict | None:
        return self._get_by_id("relationship_review_actions", action_id)

    def list_review_actions(self, *, event_id: int | None = None, limit: int = 100) -> list[RowDict]:
        clauses: list[str] = []
        params: list[Any] = []
        if event_id is not None:
            clauses.append("event_id = ?")
            params.append(event_id)
        return self._select_many("relationship_review_actions", clauses, params, order_by="id", limit=limit)

    def update_review_action(self, action_id: int, **fields: Any) -> int:
        allowed = {"event_id", "action", "note"}
        return self._update("relationship_review_actions", action_id, allowed, fields)

    def delete_review_action(self, action_id: int) -> int:
        return self._delete("relationship_review_actions", action_id)

    # full_analysis_runs / full_analysis_batches / full_analysis_observations

    def create_full_analysis_run(
        self,
        *,
        question: str,
        analysis_mode: str,
        profile_id: int | None = None,
        date_from: str | None = None,
        date_to: str | None = None,
        status: str = "pending",
        model_name: str | None = None,
        manifest: Mapping[str, Any] | None = None,
        final_synthesis: Mapping[str, Any] | None = None,
    ) -> int:
        values = {
            "question": question,
            "analysis_mode": _normalize_full_analysis_mode(analysis_mode),
            "profile_id": profile_id,
            "date_from": date_from,
            "date_to": date_to,
            "status": _normalize_full_run_status(status),
            "model_name": model_name,
            "manifest_json": _json_dumps_structured(manifest or {}),
            "final_synthesis_json": (
                _json_dumps_structured(final_synthesis) if final_synthesis is not None else None
            ),
        }
        with self.connect() as conn:
            return self._insert(conn, "full_analysis_runs", values)

    def get_full_analysis_run(self, run_id: int) -> RowDict | None:
        row = self._get_by_id("full_analysis_runs", run_id)
        if row:
            _decode_json_fields(row, ("manifest_json", "final_synthesis_json"))
        return row

    def list_full_analysis_runs(
        self,
        *,
        profile_id: int | None = None,
        status: str | None = None,
        limit: int = 100,
    ) -> list[RowDict]:
        clauses: list[str] = []
        params: list[Any] = []
        if profile_id is not None:
            clauses.append("profile_id = ?")
            params.append(profile_id)
        if status is not None:
            clauses.append("status = ?")
            params.append(_normalize_full_run_status(status))
        rows = self._select_many("full_analysis_runs", clauses, params, order_by="id DESC", limit=limit)
        for row in rows:
            _decode_json_fields(row, ("manifest_json", "final_synthesis_json"))
        return rows

    def update_full_analysis_run(self, run_id: int, **fields: Any) -> int:
        allowed = {
            "question",
            "analysis_mode",
            "profile_id",
            "date_from",
            "date_to",
            "status",
            "model_name",
            "manifest",
            "final_synthesis",
        }
        unknown = set(fields) - allowed
        if unknown:
            names = ", ".join(sorted(unknown))
            raise ValueError(f"Unsupported fields for full_analysis_runs: {names}")
        db_fields: dict[str, Any] = {}
        for key, value in fields.items():
            if key == "analysis_mode":
                db_fields["analysis_mode"] = _normalize_full_analysis_mode(value)
            elif key == "status":
                db_fields["status"] = _normalize_full_run_status(value)
            elif key == "manifest":
                db_fields["manifest_json"] = _json_dumps_structured(value or {})
            elif key == "final_synthesis":
                db_fields["final_synthesis_json"] = _json_dumps_structured(value or {})
            else:
                db_fields[key] = value
        return self._update(
            "full_analysis_runs",
            run_id,
            {
                "question",
                "analysis_mode",
                "profile_id",
                "date_from",
                "date_to",
                "status",
                "model_name",
                "manifest_json",
                "final_synthesis_json",
            },
            db_fields,
        )

    def create_full_analysis_batch(
        self,
        *,
        run_id: int,
        batch_index: int,
        input_hash: str,
        source_types: list[str] | tuple[str, ...] = (),
        date_from: str | None = None,
        date_to: str | None = None,
        item_count: int = 0,
        source_refs: list[str] | tuple[str, ...] = (),
        output: Mapping[str, Any] | None = None,
        status: str = "pending",
    ) -> int:
        values = {
            "run_id": run_id,
            "batch_index": batch_index,
            "source_types": _json_dumps_structured(list(source_types), allow_arrays=True),
            "date_from": date_from,
            "date_to": date_to,
            "item_count": item_count,
            "source_refs_json": _json_dumps_structured(list(source_refs), allow_arrays=True),
            "input_hash": input_hash,
            "output_json": _json_dumps_structured(output) if output is not None else None,
            "status": _normalize_full_batch_status(status),
        }
        with self.connect() as conn:
            return self._insert(conn, "full_analysis_batches", values)

    def get_full_analysis_batch(self, batch_id: int) -> RowDict | None:
        row = self._get_by_id("full_analysis_batches", batch_id)
        if row:
            _decode_json_fields(row, ("source_types", "source_refs_json", "output_json"))
        return row

    def list_full_analysis_batches(self, *, run_id: int, limit: int = 500) -> list[RowDict]:
        rows = self._select_many(
            "full_analysis_batches",
            ["run_id = ?"],
            [run_id],
            order_by="batch_index, id",
            limit=limit,
        )
        for row in rows:
            _decode_json_fields(row, ("source_types", "source_refs_json", "output_json"))
        return rows

    def update_full_analysis_batch(self, batch_id: int, **fields: Any) -> int:
        allowed = {
            "source_types",
            "date_from",
            "date_to",
            "item_count",
            "source_refs",
            "input_hash",
            "output",
            "status",
        }
        unknown = set(fields) - allowed
        if unknown:
            names = ", ".join(sorted(unknown))
            raise ValueError(f"Unsupported fields for full_analysis_batches: {names}")
        db_fields: dict[str, Any] = {}
        for key, value in fields.items():
            if key == "source_types":
                db_fields["source_types"] = _json_dumps_structured(list(value or ()), allow_arrays=True)
            elif key == "source_refs":
                db_fields["source_refs_json"] = _json_dumps_structured(list(value or ()), allow_arrays=True)
            elif key == "output":
                db_fields["output_json"] = _json_dumps_structured(value or {})
            elif key == "status":
                db_fields["status"] = _normalize_full_batch_status(value)
            else:
                db_fields[key] = value
        return self._update(
            "full_analysis_batches",
            batch_id,
            {
                "source_types",
                "date_from",
                "date_to",
                "item_count",
                "source_refs_json",
                "input_hash",
                "output_json",
                "status",
            },
            db_fields,
        )

    def create_full_analysis_observation(
        self,
        *,
        run_id: int,
        observation_type: str,
        summary: str,
        batch_id: int | None = None,
        source_refs: list[str] | tuple[str, ...] = (),
        result: Mapping[str, Any] | None = None,
        confidence: float = 0.5,
    ) -> int:
        values = {
            "run_id": run_id,
            "batch_id": batch_id,
            "observation_type": observation_type,
            "summary": summary,
            "source_refs_json": _json_dumps_structured(list(source_refs), allow_arrays=True),
            "result_json": _json_dumps_structured(result or {}),
            "confidence": confidence,
        }
        with self.connect() as conn:
            return self._insert(conn, "full_analysis_observations", values)

    def get_full_analysis_observation(self, observation_id: int) -> RowDict | None:
        row = self._get_by_id("full_analysis_observations", observation_id)
        if row:
            _decode_json_fields(row, ("source_refs_json", "result_json"))
        return row

    def list_full_analysis_observations(
        self,
        *,
        run_id: int,
        batch_id: int | None = None,
        limit: int = 500,
    ) -> list[RowDict]:
        clauses = ["run_id = ?"]
        params: list[Any] = [run_id]
        if batch_id is not None:
            clauses.append("batch_id = ?")
            params.append(batch_id)
        rows = self._select_many(
            "full_analysis_observations",
            clauses,
            params,
            order_by="id",
            limit=limit,
        )
        for row in rows:
            _decode_json_fields(row, ("source_refs_json", "result_json"))
        return rows

    # llm_analysis_cache

    def get_llm_analysis_cache(
        self,
        *,
        analysis_type: str,
        source_window_hash: str,
        model_name: str,
        prompt_version: str,
    ) -> RowDict | None:
        rows = self._select_many(
            "llm_analysis_cache",
            [
                "analysis_type = ?",
                "source_window_hash = ?",
                "model_name = ?",
                "prompt_version = ?",
            ],
            [analysis_type, source_window_hash, model_name, prompt_version],
            order_by="id",
            limit=1,
        )
        if not rows:
            return None
        row = rows[0]
        try:
            row["result"] = json.loads(str(row.get("result_json") or "{}"))
        except json.JSONDecodeError:
            row["result"] = {}
        return row

    def upsert_llm_analysis_cache(
        self,
        *,
        analysis_type: str,
        source_window_hash: str,
        model_name: str,
        prompt_version: str,
        result: Mapping[str, Any],
        confidence: float = 0.5,
    ) -> int:
        values = {
            "analysis_type": analysis_type,
            "source_window_hash": source_window_hash,
            "model_name": model_name,
            "prompt_version": prompt_version,
            "result_json": _json_dumps_structured(result),
            "confidence": confidence,
        }
        with self.connect() as conn:
            row = conn.execute(
                """
                SELECT id FROM llm_analysis_cache
                WHERE analysis_type = ?
                  AND source_window_hash = ?
                  AND model_name = ?
                  AND prompt_version = ?
                """,
                (analysis_type, source_window_hash, model_name, prompt_version),
            ).fetchone()
            if row is None:
                return self._insert(conn, "llm_analysis_cache", values)
            cache_id = int(row["id"])
            assignments = ", ".join(f"{field} = ?" for field in ("result_json", "confidence"))
            conn.execute(
                f"UPDATE llm_analysis_cache SET {assignments} WHERE id = ?",
                (values["result_json"], confidence, cache_id),
            )
            return cache_id

    # shared helpers

    def _get_by_id(self, table: str, row_id: int) -> RowDict | None:
        with self.connect() as conn:
            row = conn.execute(f"SELECT * FROM {table} WHERE id = ?", (row_id,)).fetchone()
        return _row_to_dict(row)

    def _select_many(
        self,
        table: str,
        clauses: list[str],
        params: list[Any],
        *,
        order_by: str,
        limit: int,
    ) -> list[RowDict]:
        where = f" WHERE {' AND '.join(clauses)}" if clauses else ""
        sql = f"SELECT * FROM {table}{where} ORDER BY {order_by} LIMIT ?"
        with self.connect() as conn:
            rows = conn.execute(sql, (*params, limit)).fetchall()
        return [_row_to_dict(row) for row in rows if row is not None]

    @staticmethod
    def _insert(conn: sqlite3.Connection, table: str, values: Mapping[str, Any]) -> int:
        columns = ", ".join(values.keys())
        placeholders = ", ".join("?" for _ in values)
        cursor = conn.execute(
            f"INSERT INTO {table} ({columns}) VALUES ({placeholders})",
            tuple(values.values()),
        )
        return int(cursor.lastrowid)

    def _update(self, table: str, row_id: int, allowed: set[str], fields: Mapping[str, Any]) -> int:
        unknown = set(fields) - allowed
        if unknown:
            names = ", ".join(sorted(unknown))
            raise ValueError(f"Unsupported fields for {table}: {names}")
        if not fields:
            return 0
        assignments = ", ".join(f"{field} = ?" for field in fields)
        sql = f"UPDATE {table} SET {assignments}, updated_at = CURRENT_TIMESTAMP WHERE id = ?"
        with self.connect() as conn:
            cursor = conn.execute(sql, (*fields.values(), row_id))
            return int(cursor.rowcount)

    def _delete(self, table: str, row_id: int) -> int:
        with self.connect() as conn:
            cursor = conn.execute(f"DELETE FROM {table} WHERE id = ?", (row_id,))
            return int(cursor.rowcount)


def _row_to_dict(row: sqlite3.Row | None) -> RowDict | None:
    if row is None:
        return None
    return dict(row)


def _int_or_none(value: str) -> int | None:
    try:
        return int(value)
    except ValueError:
        return None


def _validate_profile_name(value: object) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ValueError("profile_name is required")


def _validate_relationship_label(value: object) -> None:
    if value is None:
        return
    if value not in ALLOWED_RELATIONSHIP_LABELS:
        allowed = ", ".join(sorted(ALLOWED_RELATIONSHIP_LABELS))
        raise ValueError(f"relationship_label must be one of: {allowed}")


def _validate_label_source(value: object) -> None:
    if value != MANUAL_LABEL_SOURCE:
        raise ValueError("label_source must be user_manual")


def _normalize_event_status(value: object) -> str:
    if value not in ALLOWED_EVENT_STATUSES:
        allowed = ", ".join(sorted(ALLOWED_EVENT_STATUSES))
        raise ValueError(f"status must be one of: {allowed}")
    return str(value)


def _normalize_review_status(value: object) -> str:
    if value == "candidate":
        return "unreviewed"
    if value not in ALLOWED_REVIEW_STATUSES:
        allowed = ", ".join(sorted(ALLOWED_REVIEW_STATUSES))
        raise ValueError(f"review_status must be one of: {allowed}")
    return str(value)


def _normalize_full_analysis_mode(value: object) -> str:
    if value not in ALLOWED_FULL_ANALYSIS_MODES:
        allowed = ", ".join(sorted(ALLOWED_FULL_ANALYSIS_MODES))
        raise ValueError(f"analysis_mode must be one of: {allowed}")
    return str(value)


def _normalize_full_run_status(value: object) -> str:
    if value not in ALLOWED_FULL_RUN_STATUSES:
        allowed = ", ".join(sorted(ALLOWED_FULL_RUN_STATUSES))
        raise ValueError(f"full analysis run status must be one of: {allowed}")
    return str(value)


def _normalize_full_batch_status(value: object) -> str:
    if value not in ALLOWED_FULL_BATCH_STATUSES:
        allowed = ", ".join(sorted(ALLOWED_FULL_BATCH_STATUSES))
        raise ValueError(f"full analysis batch status must be one of: {allowed}")
    return str(value)


def _json_dumps_structured(value: Mapping[str, Any] | list[Any] | tuple[Any, ...], *, allow_arrays: bool = False) -> str:
    if not isinstance(value, Mapping) and not allow_arrays:
        raise ValueError("structured JSON payload must be an object")
    _assert_no_raw_payload(value)
    return json.dumps(value, ensure_ascii=False, sort_keys=True)


def _assert_no_raw_payload(value: Any, *, path: str = "") -> None:
    if isinstance(value, Mapping):
        for key, item in value.items():
            key_text = str(key)
            next_path = f"{path}.{key_text}" if path else key_text
            if key_text in FORBIDDEN_RAW_JSON_KEYS:
                raise ValueError(f"raw payload field is not allowed in relationship DB: {next_path}")
            _assert_no_raw_payload(item, path=next_path)
        return
    if isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            _assert_no_raw_payload(item, path=f"{path}[{index}]")


def _decode_json_fields(row: RowDict, fields: tuple[str, ...]) -> None:
    for field in fields:
        value = row.get(field)
        if value is None:
            row[field.removesuffix("_json")] = None
            continue
        try:
            decoded = json.loads(str(value))
        except json.JSONDecodeError:
            decoded = {} if not str(value).strip().startswith("[") else []
        if field.endswith("_json"):
            row[field.removesuffix("_json")] = decoded
        else:
            row[field] = decoded
