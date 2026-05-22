from pathlib import Path
import sqlite3

from relationship_lifelog_agent.adapters.notes_lifelog import NotesSqliteReadOnlyAdapter
from relationship_lifelog_agent.adapters.personal_lifelog import PersonalSqliteReadOnlyAdapter
from relationship_lifelog_agent.adapters import upstream_sqlite
from relationship_lifelog_agent.adapters.mock import (
    MockNotesLifelogAdapter,
    MockPersonalLifelogAdapter,
    MockRelationshipMemory,
)
from relationship_lifelog_agent.adapters.types import (
    AdapterEvidence,
    EventEvidence,
    EvidenceBundle,
    EvidenceItem,
    LineEvidence,
    MediaEvidence,
    MonthlyReflection,
    NoteEvidence,
    ThoughtEvidence,
)
from relationship_lifelog_agent.agent.executor import answer_question
from relationship_lifelog_agent.config import AdapterSettings, PathSettings, RelationshipSettings, Settings, load_config
from relationship_lifelog_agent.db.repository import RelationshipRepository


def test_mock_personal_adapter_returns_conflict_and_reconciliation_line_evidence() -> None:
    adapter = MockPersonalLifelogAdapter()

    conflict = adapter.search_line("喧嘩")
    reconciliation = adapter.search_line("仲直り")

    assert conflict
    assert reconciliation
    assert all(isinstance(item, LineEvidence) for item in conflict)
    assert any("喧嘩候補" in item.summary for item in conflict)
    assert any("仲直り候補" in item.summary for item in reconciliation)


def test_mock_personal_adapter_returns_post_conflict_outing_evidence() -> None:
    adapter = MockPersonalLifelogAdapter()

    media = adapter.search_media("喧嘩の後どこへ行っている？")
    events = adapter.search_events(event_type="date_or_outing")

    assert media
    assert events
    assert all(isinstance(item, MediaEvidence) for item in media)
    assert all(isinstance(item, EventEvidence) for item in events)
    assert any("外出候補" in item.summary for item in media)
    assert any(item.event_type == "date_or_outing" for item in events)


def test_mock_notes_adapter_returns_reflection_and_anxiety_evidence() -> None:
    adapter = MockNotesLifelogAdapter()

    notes = adapter.search_notes("反省 不安")
    thoughts = adapter.search_thoughts("不安")
    reflection = adapter.get_monthly_reflection("2025-01")

    assert notes
    assert thoughts
    assert isinstance(reflection, MonthlyReflection)
    assert all(isinstance(item, NoteEvidence) for item in notes)
    assert all(isinstance(item, ThoughtEvidence) for item in thoughts)
    assert any("反省" in item.summary for item in notes)
    assert any("不安" in item.summary for item in thoughts)


def test_mock_relationship_memory_bundle_supports_required_questions() -> None:
    memory = MockRelationshipMemory()

    frequency_bundle = memory.evidence_bundle_for_question("喧嘩はどのくらいしている？")
    outing_bundle = memory.evidence_bundle_for_question("喧嘩の後どこへ行っている？")

    assert isinstance(frequency_bundle, EvidenceBundle)
    assert frequency_bundle.line
    assert frequency_bundle.notes
    assert frequency_bundle.to_evidence_items()
    assert any("喧嘩候補" in item.summary for item in frequency_bundle.line)

    assert outing_bundle.media
    assert outing_bundle.events
    assert outing_bundle.places
    assert any("外出候補" in item.summary for item in outing_bundle.media)
    assert any(item.event_type == "date_or_outing" for item in outing_bundle.events)


def test_mock_adapter_evidence_satisfies_source_contract() -> None:
    personal = MockPersonalLifelogAdapter()
    notes = MockNotesLifelogAdapter()
    evidence: list[AdapterEvidence] = [
        *personal.search_line("喧嘩"),
        *personal.search_media("喧嘩後"),
        *personal.search_events(event_type="conflict"),
        *personal.search_places("喧嘩後"),
        *notes.search_notes("反省"),
        *notes.search_thoughts("不安"),
        notes.get_monthly_reflection("2025-01"),
    ]

    assert evidence
    for item in evidence:
        _assert_adapter_evidence_contract(item)
        converted = item.as_evidence_item()
        _assert_evidence_item_contract(converted)
        assert converted.source_pointer == item.source_pointer
        assert converted.evidence_strength == item.evidence_strength


def test_public_mode_evidence_items_omit_private_excerpts() -> None:
    adapter = MockPersonalLifelogAdapter()
    private_item = adapter.search_line("喧嘩")[0]

    assert private_item.sensitivity == "private"
    assert private_item.excerpt
    assert private_item.as_evidence_item(mode="public").excerpt is None

    bundle = EvidenceBundle(line=tuple(adapter.search_line("喧嘩")))
    assert bundle.to_evidence_items(mode="private")[0].excerpt
    assert bundle.to_evidence_items(mode="public")[0].excerpt is None


def test_evidence_scores_and_sensitivity_are_validated() -> None:
    valid = EvidenceItem(
        source_type="manual",
        source_id="fixture",
        date="2025-01-01",
        role="supporting",
        summary="fixture",
        confidence=1.0,
        evidence_strength=0.0,
        sensitivity="public",
    )

    assert valid.source_pointer == "manual:fixture"


def test_config_example_defines_readonly_adapter_backend() -> None:
    settings = load_config(Path("config.example.yaml"))

    assert settings.adapter.backend == "mock"
    assert settings.adapter.upstream_access_mode == "readonly"
    assert settings.adapter.allow_sqlite_readonly is True
    assert settings.adapter.copy_raw_upstream_data is False
    assert settings.paths.personal_lifelog_db is None
    assert settings.paths.notes_lifelog_db is None


def test_upstream_readonly_backend_without_paths_fails_safely(tmp_path) -> None:
    db_path = tmp_path / "relationship.sqlite"
    RelationshipRepository(db_path).create_profile("manual", relationship_label="partner")
    settings = Settings(
        adapter=AdapterSettings(backend="upstream_readonly"),
        paths=PathSettings(relationship_db=str(db_path)),
        relationship=RelationshipSettings(default_profile_id=1),
    )

    answer = answer_question("喧嘩はどのくらいしている？", settings=settings)

    assert "上流adapter未設定" in answer
    assert "personal_lifelog_db" in answer
    assert "notes_lifelog_db" in answer


def test_open_readonly_sqlite_uses_mode_ro(tmp_path, monkeypatch) -> None:
    db_path = tmp_path / "upstream.sqlite"
    conn = sqlite3.connect(db_path)
    conn.execute("CREATE TABLE fixture (id INTEGER PRIMARY KEY)")
    conn.commit()
    conn.close()
    calls: list[tuple[str, dict[str, object]]] = []
    real_connect = sqlite3.connect

    def recording_connect(database, *args, **kwargs):
        calls.append((str(database), dict(kwargs)))
        return real_connect(database, *args, **kwargs)

    monkeypatch.setattr(upstream_sqlite.sqlite3, "connect", recording_connect)
    ro_conn = upstream_sqlite.open_readonly_sqlite(db_path)
    ro_conn.close()

    assert calls
    assert "mode=ro" in calls[0][0]
    assert calls[0][1]["uri"] is True


def test_personal_upstream_search_line_reads_synthetic_db_readonly(tmp_path) -> None:
    db_path = tmp_path / "personal.sqlite"
    conn = sqlite3.connect(db_path)
    conn.execute(
        "CREATE TABLE line_messages (id TEXT PRIMARY KEY, chat_id TEXT, sent_at TEXT, sender TEXT, text TEXT, message_type TEXT)"
    )
    conn.execute(
        "INSERT INTO line_messages VALUES (?, ?, ?, ?, ?, ?)",
        ("line-1", "chat-1", "2025-01-10T10:00:00", "speaker-a", "synthetic 謝罪 and schedule mismatch", "text"),
    )
    conn.commit()
    conn.close()
    adapter = PersonalSqliteReadOnlyAdapter(db_path)

    private_results = adapter.search_line("謝罪", mode="private")
    public_results = adapter.search_line("謝罪", mode="public")

    assert len(private_results) == 1
    assert private_results[0].source_pointer == "personal_lifelog_rag:line_messages:line-1"
    assert private_results[0].excerpt
    assert public_results[0].excerpt is None


def test_notes_upstream_search_notes_public_mode_omits_excerpt(tmp_path) -> None:
    db_path = tmp_path / "notes.sqlite"
    conn = sqlite3.connect(db_path)
    conn.execute("CREATE TABLE notes (id TEXT PRIMARY KEY, note_date TEXT, title TEXT, body TEXT)")
    conn.execute(
        "INSERT INTO notes VALUES (?, ?, ?, ?)",
        ("note-1", "2025-01-11", "synthetic reflection", "synthetic 反省 note body"),
    )
    conn.commit()
    conn.close()
    adapter = NotesSqliteReadOnlyAdapter(db_path)

    private_results = adapter.search_notes("反省", mode="private")
    public_results = adapter.search_notes("反省", mode="public")

    assert len(private_results) == 1
    assert private_results[0].source_pointer == "notes_lifelog_rag:notes:note-1"
    assert private_results[0].excerpt
    assert public_results[0].excerpt is None


def _assert_adapter_evidence_contract(item: AdapterEvidence) -> None:
    assert item.source_id
    assert item.source_type
    assert item.source_pointer == f"{item.source_type}:{item.source_id}"
    assert item.sensitivity in {"public", "private", "sensitive"}
    assert 0.0 <= item.confidence <= 1.0
    assert 0.0 <= item.evidence_strength <= 1.0


def _assert_evidence_item_contract(item: EvidenceItem) -> None:
    assert item.source_id
    assert item.source_type
    assert item.source_pointer == f"{item.source_type}:{item.source_id}"
    assert item.sensitivity in {"public", "private", "sensitive"}
    assert 0.0 <= item.confidence <= 1.0
    assert 0.0 <= item.evidence_strength <= 1.0
