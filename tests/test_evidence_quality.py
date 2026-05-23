from __future__ import annotations

from relationship_lifelog_agent.adapters.types import EventEvidence, LineEvidence, NoteEvidence
from relationship_lifelog_agent.agent.executor import answer_question
from relationship_lifelog_agent.config import PathSettings, RelationshipSettings, Settings
from relationship_lifelog_agent.db.repository import RelationshipRepository


def test_conflict_answer_filters_low_quality_notes_and_limits_excerpts(tmp_path) -> None:
    settings = _settings_with_profile(tmp_path)

    answer = answer_question(
        "いおりとの喧嘩はどれくらいしている？",
        settings=settings,
        memory=_EvidenceQualityMemory(),
    )

    assert "profile_id: 1" in answer
    assert "1970-01-01" not in answer
    assert "GEN ROCK" not in answer
    assert "履修" not in answer
    assert "遠い日の関係メモ" not in answer
    assert "中程度以上の喧嘩候補は0件、軽いすれ違い候補は1件あります" in answer
    assert "喧嘩候補が 1 件" not in answer
    assert answer.count(" / excerpt: ") <= 5
    for line in answer.splitlines():
        if "[note:supporting]" in line and " / excerpt: " in line:
            excerpt = line.split(" / excerpt: ", 1)[1]
            assert len(excerpt) <= 123


def test_public_mode_omits_filtered_note_excerpts(tmp_path) -> None:
    settings = _settings_with_profile(tmp_path)

    answer = answer_question(
        "いおりとの喧嘩はどれくらいしている？",
        mode="public",
        settings=settings,
        memory=_EvidenceQualityMemory(),
    )

    assert "excerpt:" not in answer
    assert "partner" not in answer


def test_same_date_same_summary_evidence_is_deduplicated(tmp_path) -> None:
    settings = _settings_with_profile(tmp_path)

    answer = answer_question(
        "いおりとの喧嘩はどれくらいしている？",
        settings=settings,
        memory=_DuplicateEvidenceMemory(),
    )

    evidence_block = answer.split("<summary>根拠を表示</summary>", 1)[1].split("</details>", 1)[0]
    assert evidence_block.count("同一説明の軽いすれ違い候補。") == 1


class _EvidenceQualityMemory:
    backend = "upstream_readonly"

    def __init__(self) -> None:
        self.personal = _EvidenceQualityPersonal()
        self.notes = _EvidenceQualityNotes()

    def warnings(self) -> list[str]:
        return []


class _DuplicateEvidenceMemory(_EvidenceQualityMemory):
    def __init__(self) -> None:
        self.personal = _DuplicateEvidencePersonal()
        self.notes = _EmptyNotes()


class _EvidenceQualityPersonal:
    def search_events(self, **kwargs):
        del kwargs
        return [
            EventEvidence(
                source_id="event-minor-2024-12-16",
                date="2024-12-16",
                role="primary",
                summary="2024-12-16の軽いすれ違い候補。",
                confidence=0.55,
                evidence_strength=0.58,
                event_type="minor_misunderstanding",
                severity=1,
            )
        ]

    def search_line(self, query, **kwargs):
        del query, kwargs
        return [
            LineEvidence(
                source_id="line-minor-2024-12-16",
                date="2024-12-16",
                role="primary",
                summary="LINE上に軽いすれ違い候補があります。",
                excerpt="短いLINE候補です。",
                confidence=0.55,
                evidence_strength=0.58,
            )
        ]

    def search_media(self, *args, **kwargs):
        del args, kwargs
        return []

    def get_month_summary(self, *args, **kwargs):
        del args, kwargs
        return {"summary": ""}


class _DuplicateEvidencePersonal(_EvidenceQualityPersonal):
    def search_events(self, **kwargs):
        del kwargs
        return [
            EventEvidence(
                source_id="event-dup",
                date="2024-12-16",
                role="primary",
                summary="同一説明の軽いすれ違い候補。",
                event_type="minor_misunderstanding",
                confidence=0.55,
                evidence_strength=0.58,
                severity=1,
            )
        ]

    def search_line(self, query, **kwargs):
        del query, kwargs
        return [
            LineEvidence(
                source_id="line-dup",
                date="2024-12-16",
                role="primary",
                summary="同一説明の軽いすれ違い候補。",
                excerpt="短いLINE候補です。",
                confidence=0.55,
                evidence_strength=0.58,
            )
        ]


class _EvidenceQualityNotes:
    def search_notes(self, query, **kwargs):
        del query, kwargs
        return [
            NoteEvidence(
                source_id="note-undated",
                date="1970-01-01",
                role="supporting",
                summary="GEN ROCK さんWSのメモ。",
                excerpt="GEN ROCK さんWSの長いメモ本文候補。",
            ),
            NoteEvidence(
                source_id="note-far",
                date="2024-08-01",
                role="supporting",
                summary="遠い日の関係メモ。",
                excerpt="いおりについて書いているが、イベント日から大きく離れています。",
            ),
            NoteEvidence(
                source_id="note-unrelated-near",
                date="2024-12-16",
                role="supporting",
                summary="履修についてのメモ。",
                excerpt="履修登録と学業についてのメモです。",
            ),
            NoteEvidence(
                source_id="note-related-near",
                date="2024-12-16",
                role="supporting",
                summary="いおりとの関係について自分側の反省メモ候補。",
                excerpt="いおりとの関係について反省している短い候補。" + "長い本文" * 80,
                category="relationship_reflection",
                metadata={"tags": ("関係", "反省")},
            ),
        ]

    def search_thoughts(self, *args, **kwargs):
        del args, kwargs
        return []

    def get_monthly_reflection(self, *args, **kwargs):
        del args, kwargs
        raise AssertionError("not used")


class _EmptyNotes(_EvidenceQualityNotes):
    def search_notes(self, query, **kwargs):
        del query, kwargs
        return []


def _settings_with_profile(tmp_path) -> Settings:
    db_path = tmp_path / "relationship.sqlite"
    repo = RelationshipRepository(db_path)
    repo.create_profile(
        "いおり",
        relationship_label="partner",
        person_source_id="plr:person:target",
        line_speaker_source_id="plr:line_speaker:target",
        self_person_source_id="plr:person:self",
        self_line_speaker_source_id="plr:line_speaker:self",
    )
    return Settings(
        paths=PathSettings(relationship_db=str(db_path)),
        relationship=RelationshipSettings(default_profile_id=1),
    )
