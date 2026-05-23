from __future__ import annotations

import json

from relationship_lifelog_agent.adapters.types import EventEvidence, LineEvidence, MediaEvidence, NoteEvidence
from relationship_lifelog_agent.agent.answerability import check_answerability
from relationship_lifelog_agent.agent.executor import answer_chat
from relationship_lifelog_agent.agent.profile_resolver import resolve_profile
from relationship_lifelog_agent.agent.query_planner import build_conversation_query_plan
from relationship_lifelog_agent.agent.question_understanding import understand_question
from relationship_lifelog_agent.agent.router import route_question
from relationship_lifelog_agent.analytics.llm_line_window import analyze_line_window_with_llm
from relationship_lifelog_agent.analytics.llm_media_summary import summarize_media_day_with_llm
from relationship_lifelog_agent.analytics.llm_note_window import analyze_note_window_with_llm
from relationship_lifelog_agent.config import LlmSettings, PathSettings, RelationshipSettings, Settings
from relationship_lifelog_agent.db.repository import RelationshipRepository
from relationship_lifelog_agent.llm.local_client import LocalLlmClient


QUESTION = "いおりと喧嘩した日はいつ？その日の前後の日は写真から何をしていた？"


def test_compound_question_routes_to_conflict_dates_with_surrounding_media() -> None:
    route = route_question(QUESTION)

    assert route.intents[0] == "conflict_dates_with_surrounding_media"
    assert "conflict_date_lookup" in route.intents
    assert "surrounding_media_summary" in route.intents


def test_compound_query_plan_uses_exact_media_window_not_post_conflict(tmp_path) -> None:
    settings = _settings_with_profile(tmp_path)
    frame = understand_question(QUESTION, settings=settings)
    resolution = resolve_profile(settings, frame, selected_profile_id=1)
    answerability = check_answerability(frame, resolution, settings)
    plan = build_conversation_query_plan(frame, answerability, settings=settings)
    tools = [step.tool_name for step in plan.plan_steps]

    assert "find_conflict_candidate_dates" in tools
    assert "get_media_by_exact_date_range" in tools
    assert "summarize_media_day_with_llm" in tools
    assert "search_post_conflict_activities" not in tools


def test_compound_answer_limits_media_to_before_same_after_days(tmp_path) -> None:
    settings = _settings_with_profile(tmp_path)

    answer = answer_chat(
        QUESTION,
        settings=settings,
        profile_id=1,
        memory=_MemoryWithLeakyMedia(),
        mode="private",
    ).text

    assert "2024-12-16" in answer
    assert "2024-12-15" in answer
    assert "2024-12-17" in answer
    assert "2025-01-05" not in answer
    assert "2026-02-09" not in answer
    assert "activity_candidates" not in answer
    assert "- media_items_in_exact_window: 2" in answer


def test_public_compound_answer_hides_media_excerpt(tmp_path) -> None:
    settings = _settings_with_profile(tmp_path)

    answer = answer_chat(
        QUESTION,
        settings=settings,
        profile_id=1,
        memory=_MemoryWithLeakyMedia(),
        mode="public",
    ).text

    assert "secret caption" not in answer
    assert "/home/" not in answer


def test_line_window_analyzer_sends_only_small_window_to_llm() -> None:
    captured: dict[str, object] = {}

    def post(_url, payload, _timeout):
        captured["payload"] = payload
        return {
            "message": {
                "content": json.dumps(
                    {
                        "is_conflict_candidate": False,
                        "is_minor_misunderstanding": True,
                        "severity": 1,
                        "confidence": 0.6,
                        "estimated_start_date": "2024-12-16",
                        "estimated_end_date": "2024-12-16",
                        "summary": "軽いすれ違い候補です。",
                        "evidence_message_refs": ["line_message:0"],
                        "evidence_excerpt_short": ["短いexcerpt"],
                        "is_joke_or_general_statement_possible": False,
                        "cautions": ["断定しません。"],
                    },
                    ensure_ascii=False,
                )
            }
        }

    client = LocalLlmClient(_llm_settings(), http_post=post)
    messages = [
        LineEvidence(
            source_id=str(index),
            date="2024-12-16",
            summary=f"summary {index}",
            excerpt=f"excerpt {index} fixture/private/raw.txt",
        )
        for index in range(40)
    ]

    analysis = analyze_line_window_with_llm(None, messages, ("2024-12-16", "2024-12-16"), client)
    prompt = str(captured["payload"])

    assert analysis.backend == "llm"
    assert "summary 29" in prompt
    assert "summary 30" not in prompt
    assert "fixture/private/raw.txt" not in prompt


def test_note_window_analyzer_excludes_1970_and_unrelated_notes() -> None:
    notes = [
        NoteEvidence(source_id="sentinel", date="1970-01-01", summary="反省", excerpt="反省"),
        NoteEvidence(source_id="school", date="2024-12-16", summary="履修登録のメモ", excerpt="履修"),
    ]

    analyses = analyze_note_window_with_llm(notes, ("2024-12-15", "2024-12-17"), None, None)

    assert analyses == []


def test_media_day_summary_does_not_send_path_gps_or_embedding_to_llm() -> None:
    captured: dict[str, object] = {}

    def post(_url, payload, _timeout):
        captured["payload"] = payload
        return {
            "message": {
                "content": json.dumps(
                    {
                        "date": "2024-12-17",
                        "has_media": True,
                        "likely_activities": ["カフェ"],
                        "place_summaries": [],
                        "people_roles_present": [],
                        "summary": "カフェのような場所にいた可能性があります。",
                        "confidence": 0.6,
                        "evidence_media_refs": ["media_caption:m1"],
                        "cautions": ["写真だけでは断定しません。"],
                    },
                    ensure_ascii=False,
                )
            }
        }

    client = LocalLlmClient(_llm_settings(), http_post=post)
    summarize_media_day_with_llm(
        "2024-12-17",
        [
            MediaEvidence(
                source_id="m1",
                date="2024-12-17",
                summary="カフェ fixture/private/photo.jpg 99.9999,99.9999 FACE_VECTOR_TOKEN",
                excerpt="fixture/private/photo.jpg exact gps 99.9999,99.9999",
            )
        ],
        None,
        client,
    )
    prompt = str(captured["payload"])

    assert "fixture/private/photo.jpg" not in prompt
    assert "99.9999" not in prompt
    assert "FACE_VECTOR_TOKEN" not in prompt


def _settings_with_profile(tmp_path) -> Settings:
    db_path = tmp_path / "relationship.sqlite"
    repo = RelationshipRepository(db_path)
    repo.create_profile(
        "いおり",
        person_source_id="plr:person:target",
        line_speaker_source_id="plr:line_speaker:target",
        self_line_speaker_source_id="plr:line_speaker:self",
        relationship_label="partner",
    )
    return Settings(
        paths=PathSettings(relationship_db=str(db_path)),
        relationship=RelationshipSettings(default_profile_id=1, surrounding_media_days_before=1, surrounding_media_days_after=1),
    )


def _llm_settings() -> LlmSettings:
    return LlmSettings(
        provider="ollama",
        model="fake-local",
        enabled=True,
        use_for_query_planning=True,
        use_for_answer_composition=True,
    )


class _MemoryWithLeakyMedia:
    backend = "test"

    def __init__(self) -> None:
        self.personal = self
        self.notes = self
        self.events = [
            EventEvidence(
                source_id="conflict-2024-12-16",
                date="2024-12-16",
                event_type="minor_misunderstanding",
                summary="短いLINE上の軽いすれ違い候補。",
                severity=1,
                confidence=0.6,
                evidence_strength=0.6,
            )
        ]
        self.media = [
            MediaEvidence(source_id="m-before", date="2024-12-15", summary="家で過ごした写真候補。", excerpt="secret caption before"),
            MediaEvidence(source_id="m-after", date="2024-12-17", summary="カフェのような場所の写真候補。", excerpt="secret caption after"),
            MediaEvidence(source_id="m-outside-2025", date="2025-01-05", summary="日付外の外出写真。", excerpt="leak"),
            MediaEvidence(source_id="m-outside-2026", date="2026-02-09", summary="日付外の外出写真。", excerpt="leak"),
        ]

    def search_events(self, date_from=None, date_to=None, event_type=None, **_kwargs):
        return [
            item
            for item in self.events
            if (event_type is None or item.event_type == event_type)
            and (not date_from or item.date >= date_from)
            and (not date_to or item.date <= date_to)
        ]

    def search_line(self, _query, date_from=None, date_to=None, **_kwargs):
        item = LineEvidence(
            source_id="line-2024-12-16",
            date="2024-12-16",
            summary="短い謝罪と予定確認がある軽いすれ違い候補。",
            excerpt="短い謝罪と予定確認。",
        )
        return [item] if (not date_from or item.date >= date_from) and (not date_to or item.date <= date_to) else []

    def search_media(self, _query, date_from=None, date_to=None, **_kwargs):
        return [
            item
            for item in self.media
            if (not date_from or item.date >= date_from) and (not date_to or item.date <= date_to)
        ]

    def search_notes(self, *_args, **_kwargs):
        return []
