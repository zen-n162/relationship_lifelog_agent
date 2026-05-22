from relationship_lifelog_agent.adapters.mock import (
    MockNotesLifelogAdapter,
    MockPersonalLifelogAdapter,
    MockRelationshipMemory,
)
from relationship_lifelog_agent.adapters.types import (
    EventEvidence,
    EvidenceBundle,
    LineEvidence,
    MediaEvidence,
    MonthlyReflection,
    NoteEvidence,
    ThoughtEvidence,
)


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
