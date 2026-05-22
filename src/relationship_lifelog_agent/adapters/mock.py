from __future__ import annotations

from datetime import date

from relationship_lifelog_agent.adapters.types import (
    AdapterEvidence,
    EventEvidence,
    EvidenceBundle,
    LineEvidence,
    MediaEvidence,
    MonthlyReflection,
    NoteEvidence,
    PlaceEvidence,
    PostConflictActivity,
    RelationshipEvent,
    ThoughtEvidence,
)


class MockPersonalLifelogAdapter:
    """Synthetic personal_lifelog_rag adapter.

    It never reads upstream files or DBs. The data is deliberately small and
    privacy-safe, but shaped like the real adapter contract.
    """

    def __init__(self) -> None:
        self._line = (
            LineEvidence(
                source_id="mock-line-jan10-conflict",
                date="2025-01-10",
                role="primary",
                summary="予定変更への不満と短い謝罪表現が同じ日にある喧嘩候補。",
                excerpt="予定の話が合わず、謝罪に近い短文がある。",
                confidence=0.65,
                speaker_id="mock-speaker-user",
                conversation_id="mock-line-thread",
                metadata={"tags": ("喧嘩", "予定調整", "謝罪")},
            ),
            LineEvidence(
                source_id="mock-line-jan13-reconcile",
                date="2025-01-13",
                role="after_event",
                summary="通常の予定確認に戻っている仲直り候補。",
                excerpt="通常の会話に戻り、会う予定を確認している。",
                confidence=0.6,
                speaker_id="mock-speaker-user",
                conversation_id="mock-line-thread",
                metadata={"tags": ("仲直り", "通常会話", "喧嘩後")},
            ),
            LineEvidence(
                source_id="mock-line-jan24-minor",
                date="2025-01-24",
                role="context",
                summary="返信間隔の変化のみで、単独では弱い軽いすれ違い候補。",
                excerpt="返信が少し空いた後、通常の予定確認に戻っている。",
                confidence=0.42,
                speaker_id="mock-speaker-user",
                conversation_id="mock-line-thread",
                metadata={"tags": ("軽いすれ違い", "返信遅延", "弱い信号")},
            ),
            LineEvidence(
                source_id="mock-line-feb14-conflict",
                date="2025-02-14",
                role="primary",
                summary="謝罪表現と予定の再調整がある中程度の喧嘩候補。",
                excerpt="謝る表現と、次の予定を決め直す流れがある。",
                confidence=0.68,
                speaker_id="mock-speaker-user",
                conversation_id="mock-line-thread",
                metadata={"tags": ("喧嘩", "謝罪", "再調整")},
            ),
        )
        self._media = (
            MediaEvidence(
                source_id="mock-media-jan13-cafe",
                date="2025-01-13",
                role="after_event",
                summary="喧嘩候補の3日後にカフェ付近の外出候補がある。",
                excerpt="カフェ付近のイベント候補。",
                confidence=0.62,
                media_id="mock-photo-jan13",
                media_type="photo",
                place_id="mock-place-cafe",
                metadata={"tags": ("喧嘩後", "外出", "カフェ")},
            ),
            MediaEvidence(
                source_id="mock-media-feb17-meal",
                date="2025-02-17",
                role="after_event",
                summary="喧嘩候補の3日後に食事に関連する外出候補がある。",
                excerpt="食事の写真に近いキャプション候補。",
                confidence=0.59,
                media_id="mock-photo-feb17",
                media_type="photo",
                place_id="mock-place-meal",
                metadata={"tags": ("喧嘩後", "外出", "食事")},
            ),
        )
        self._places = (
            PlaceEvidence(
                source_id="mock-place-jan13-cafe",
                date="2025-01-13",
                role="after_event",
                summary="カフェ候補の場所記録。",
                excerpt="カフェ付近の場所候補。",
                confidence=0.62,
                place_id="mock-place-cafe",
                place_label="カフェ候補",
                area_label="mock area",
                metadata={"tags": ("喧嘩後", "外出", "場所")},
            ),
            PlaceEvidence(
                source_id="mock-place-feb17-meal",
                date="2025-02-17",
                role="after_event",
                summary="食事候補の場所記録。",
                excerpt="食事場所に近い場所候補。",
                confidence=0.58,
                place_id="mock-place-meal",
                place_label="食事候補",
                area_label="mock area",
                metadata={"tags": ("喧嘩後", "外出", "場所")},
            ),
        )
        self._events = (
            EventEvidence(
                source_id="mock-event-jan10-conflict",
                date="2025-01-10",
                role="primary",
                summary="予定調整をめぐる中程度の喧嘩候補。",
                confidence=0.64,
                event_type="conflict",
                review_status="candidate",
                evidence_strength=0.72,
                severity=2,
                metadata={"tags": ("喧嘩", "予定調整")},
            ),
            EventEvidence(
                source_id="mock-event-jan13-reconcile",
                date="2025-01-13",
                role="after_event",
                summary="通常会話と外出候補がある仲直り候補。",
                confidence=0.58,
                event_type="reconciliation",
                review_status="candidate",
                evidence_strength=0.64,
                severity=0,
                metadata={"tags": ("仲直り", "喧嘩後", "外出")},
            ),
            EventEvidence(
                source_id="mock-event-jan24-minor",
                date="2025-01-24",
                role="context",
                summary="返信間隔の変化のみを含む軽いすれ違い候補。",
                confidence=0.48,
                event_type="minor_misunderstanding",
                review_status="candidate",
                evidence_strength=0.46,
                severity=1,
                metadata={"tags": ("軽いすれ違い", "返信遅延", "弱い信号")},
            ),
            EventEvidence(
                source_id="mock-event-feb14-conflict",
                date="2025-02-14",
                role="primary",
                summary="謝罪表現と予定の再調整がある中程度の喧嘩候補。",
                confidence=0.67,
                event_type="conflict",
                review_status="candidate",
                evidence_strength=0.75,
                severity=2,
                metadata={"tags": ("喧嘩", "謝罪", "再調整")},
            ),
            EventEvidence(
                source_id="mock-event-jan13-outing",
                date="2025-01-13",
                role="after_event",
                summary="喧嘩候補の後のカフェ外出候補。",
                confidence=0.62,
                event_type="date_or_outing",
                review_status="candidate",
                evidence_strength=0.68,
                severity=0,
                metadata={"tags": ("喧嘩後", "外出", "カフェ")},
            ),
            EventEvidence(
                source_id="mock-event-feb17-outing",
                date="2025-02-17",
                role="after_event",
                summary="喧嘩候補の後の食事外出候補。",
                confidence=0.59,
                event_type="date_or_outing",
                review_status="candidate",
                evidence_strength=0.64,
                severity=0,
                metadata={"tags": ("喧嘩後", "外出", "食事")},
            ),
        )

    def search_line(
        self,
        query: str,
        date_from: str | None = None,
        date_to: str | None = None,
        speaker_id: str | None = None,
        limit: int = 50,
    ) -> list[LineEvidence]:
        matches = [
            item
            for item in self._line
            if _in_range(item.date, date_from, date_to)
            and (speaker_id is None or item.speaker_id == speaker_id)
            and _matches_query(item, query)
        ]
        return matches[:limit]

    def search_media(
        self,
        query: str,
        date_from: str | None = None,
        date_to: str | None = None,
        person_id: str | None = None,
        place_id: str | None = None,
        limit: int = 50,
    ) -> list[MediaEvidence]:
        del person_id
        matches = [
            item
            for item in self._media
            if _in_range(item.date, date_from, date_to)
            and (place_id is None or item.place_id == place_id)
            and _matches_query(item, query)
        ]
        return matches[:limit]

    def search_events(
        self,
        date_from: str | None = None,
        date_to: str | None = None,
        event_type: str | None = None,
        person_id: str | None = None,
        place_id: str | None = None,
        limit: int = 50,
    ) -> list[EventEvidence]:
        del person_id, place_id
        matches = [
            item
            for item in self._events
            if _in_range(item.date, date_from, date_to)
            and (event_type is None or item.event_type == event_type)
        ]
        return matches[:limit]

    def search_places(
        self,
        query: str,
        date_from: str | None = None,
        date_to: str | None = None,
        limit: int = 50,
    ) -> list[PlaceEvidence]:
        matches = [
            item
            for item in self._places
            if _in_range(item.date, date_from, date_to) and _matches_query(item, query)
        ]
        return matches[:limit]

    def get_day_summary(self, day: str, mode: str = "private") -> dict[str, str]:
        del mode
        return {"date": day, "summary": "mock day summary with no private upstream reads"}

    def get_month_summary(self, month: str, mode: str = "private") -> dict[str, str]:
        del mode
        return {
            "month": month,
            "summary": "2025年1月は、予定調整のすれ違い候補と、その後の通常化候補がある月。",
        }


class MockNotesLifelogAdapter:
    """Synthetic notes_lifelog_rag adapter."""

    def __init__(self) -> None:
        self._notes = (
            NoteEvidence(
                source_id="mock-note-jan10-reflection",
                date="2025-01-10",
                role="supporting",
                summary="予定をうまく伝えられなかったかもしれない、という自分側の反省メモ候補。",
                excerpt="予定をうまく伝えられなかったかもしれない、という反省。",
                confidence=0.61,
                note_id="mock-note-jan10",
                category="regret",
                metadata={"tags": ("反省", "不安", "喧嘩")},
            ),
            NoteEvidence(
                source_id="mock-note-jan28-anxiety",
                date="2025-01-28",
                role="primary",
                summary="関係について自分が不安と感謝を両方書いたメモ候補。",
                excerpt="不安もあるが、話せてよかったという感謝もある。",
                confidence=0.58,
                note_id="mock-note-jan28",
                category="anxiety_gratitude",
                metadata={"tags": ("不安", "感謝", "メモ")},
            ),
        )
        self._thoughts = (
            ThoughtEvidence(
                source_id="mock-thought-jan10-reflection",
                date="2025-01-10",
                role="supporting",
                summary="自分側の反省として扱える思考メモ候補。",
                excerpt="伝え方をもう少し考えたい、という自分側のメモ候補。",
                confidence=0.57,
                thought_id="mock-thought-jan10",
                thought_type="own_thought",
                metadata={"tags": ("反省", "自分側")},
            ),
            ThoughtEvidence(
                source_id="mock-thought-jan28-anxiety",
                date="2025-01-28",
                role="primary",
                summary="不安と感謝が混在する自分側の思考メモ候補。",
                excerpt="不安と感謝が混在している。",
                confidence=0.56,
                thought_id="mock-thought-jan28",
                thought_type="relationship_reflection",
                metadata={"tags": ("不安", "感謝", "自分側")},
            ),
        )
        self._monthly_reflections = {
            "2025-01": MonthlyReflection(
                source_id="mock-reflection-2025-01",
                date="2025-01-31",
                role="context",
                summary="月次振り返り候補。すれ違い候補、外出候補、自分側の反省が混在。",
                excerpt="関係を落ち着いて振り返る必要がある、という自分側の月次メモ候補。",
                confidence=0.56,
                month="2025-01",
                metadata={"tags": ("月次", "振り返り", "反省")},
            )
        }

    def search_notes(
        self,
        query: str,
        date_from: str | None = None,
        date_to: str | None = None,
        limit: int = 50,
    ) -> list[NoteEvidence]:
        matches = [
            item
            for item in self._notes
            if _in_range(item.date, date_from, date_to) and _matches_query(item, query)
        ]
        return matches[:limit]

    def search_thoughts(
        self,
        query: str,
        date_from: str | None = None,
        date_to: str | None = None,
        limit: int = 50,
    ) -> list[ThoughtEvidence]:
        matches = [
            item
            for item in self._thoughts
            if _in_range(item.date, date_from, date_to) and _matches_query(item, query)
        ]
        return matches[:limit]

    def get_monthly_reflection(self, month: str) -> MonthlyReflection:
        return self._monthly_reflections.get(
            month,
            MonthlyReflection(
                source_id=f"mock-reflection-{month}",
                date=f"{month}-01",
                role="context",
                summary="mock 月次振り返り候補はありますが、この月の詳細データは薄いです。",
                excerpt=None,
                confidence=0.3,
                month=month,
                metadata={"tags": ("月次", "低信頼")},
            ),
        )


class MockRelationshipMemory:
    """Synthetic aggregate memory used by analytics and QA tests."""

    def __init__(
        self,
        personal: MockPersonalLifelogAdapter | None = None,
        notes: MockNotesLifelogAdapter | None = None,
    ) -> None:
        self.personal = personal or MockPersonalLifelogAdapter()
        self.notes = notes or MockNotesLifelogAdapter()
        jan10_line = self.personal.search_line("喧嘩", date_from="2025-01-10", date_to="2025-01-10")[0]
        jan10_note = self.notes.search_notes("反省", date_from="2025-01-10", date_to="2025-01-10")[0]
        jan24_line = self.personal.search_line("軽いすれ違い", date_from="2025-01-24", date_to="2025-01-24")[0]
        jan28_note = self.notes.search_notes("不安", date_from="2025-01-28", date_to="2025-01-28")[0]
        feb14_line = self.personal.search_line("喧嘩", date_from="2025-02-14", date_to="2025-02-14")[0]
        jan13_line = self.personal.search_line("仲直り", date_from="2025-01-13", date_to="2025-01-13")[0]
        self._events = (
            RelationshipEvent(
                event_id="mock-conflict-2025-01-10",
                event_type="conflict",
                date="2025-01-10",
                summary="予定調整の不一致と謝罪に近い表現が同じ日にある喧嘩候補。",
                review_status="candidate",
                confidence=0.64,
                evidence_strength=0.72,
                severity=2,
                evidence=(jan10_line.as_evidence_item(), jan10_note.as_evidence_item()),
            ),
            RelationshipEvent(
                event_id="mock-reconciliation-2025-01-13",
                event_type="reconciliation",
                date="2025-01-13",
                summary="通常会話に戻っている仲直り候補。",
                review_status="candidate",
                confidence=0.58,
                evidence_strength=0.64,
                severity=0,
                evidence=(jan13_line.as_evidence_item(),),
            ),
            RelationshipEvent(
                event_id="mock-minor-2025-01-24",
                event_type="minor_misunderstanding",
                date="2025-01-24",
                summary="返信間隔が少し長く、後で通常のやりとりに戻る軽いすれ違い候補。",
                review_status="candidate",
                confidence=0.48,
                evidence_strength=0.46,
                severity=1,
                evidence=(jan24_line.as_evidence_item(),),
            ),
            RelationshipEvent(
                event_id="mock-emotional-note-2025-01-28",
                event_type="emotional_note",
                date="2025-01-28",
                summary="関係について自分が不安と感謝を両方書いたメモ候補。",
                review_status="candidate",
                confidence=0.58,
                evidence_strength=0.55,
                severity=0,
                evidence=(jan28_note.as_evidence_item(),),
            ),
            RelationshipEvent(
                event_id="mock-conflict-2025-02-14",
                event_type="conflict",
                date="2025-02-14",
                summary="謝罪表現と後日の通常会話がある中程度の喧嘩候補。",
                review_status="candidate",
                confidence=0.67,
                evidence_strength=0.75,
                severity=2,
                evidence=(feb14_line.as_evidence_item(),),
            ),
        )
        jan13_media = self.personal.search_media("喧嘩後", date_from="2025-01-13", date_to="2025-01-13")[0]
        feb17_media = self.personal.search_media("喧嘩後", date_from="2025-02-17", date_to="2025-02-17")[0]
        self._activities = (
            PostConflictActivity(
                conflict_event_id="mock-conflict-2025-01-10",
                activity_event_id="mock-outing-2025-01-13",
                date="2025-01-13",
                days_after_conflict=3,
                place_label="カフェ候補",
                activity_type="outing_candidate",
                confidence=0.62,
                evidence_strength=0.68,
                evidence=(jan13_media.as_evidence_item(),),
            ),
            PostConflictActivity(
                conflict_event_id="mock-conflict-2025-02-14",
                activity_event_id="mock-outing-2025-02-17",
                date="2025-02-17",
                days_after_conflict=3,
                place_label="食事候補",
                activity_type="outing_candidate",
                confidence=0.59,
                evidence_strength=0.64,
                evidence=(feb17_media.as_evidence_item(),),
            ),
        )

    def search_line(
        self,
        query: str,
        date_from: str | None = None,
        date_to: str | None = None,
        speaker_id: str | None = None,
        limit: int = 50,
    ) -> list[LineEvidence]:
        return self.personal.search_line(
            query=query,
            date_from=date_from,
            date_to=date_to,
            speaker_id=speaker_id,
            limit=limit,
        )

    def search_media(
        self,
        query: str,
        date_from: str | None = None,
        date_to: str | None = None,
        person_id: str | None = None,
        place_id: str | None = None,
        limit: int = 50,
    ) -> list[MediaEvidence]:
        return self.personal.search_media(
            query=query,
            date_from=date_from,
            date_to=date_to,
            person_id=person_id,
            place_id=place_id,
            limit=limit,
        )

    def search_events(
        self,
        date_from: str | None = None,
        date_to: str | None = None,
        event_type: str | None = None,
        person_id: str | None = None,
        place_id: str | None = None,
        limit: int = 50,
    ) -> list[RelationshipEvent]:
        del person_id, place_id
        matches = [
            event
            for event in self._events
            if (event_type is None or event.event_type == event_type)
            and _in_range(event.date, date_from, date_to)
        ]
        return matches[:limit]

    def search_post_conflict_activities(
        self,
        date_from: str | None = None,
        date_to: str | None = None,
        limit: int = 50,
    ) -> list[PostConflictActivity]:
        matches = [
            activity
            for activity in self._activities
            if _in_range(activity.date, date_from, date_to)
        ]
        return matches[:limit]

    def get_day_summary(self, day: str, mode: str = "private") -> dict[str, str]:
        return self.personal.get_day_summary(day, mode=mode)

    def get_month_summary(self, month: str, mode: str = "private") -> dict[str, str]:
        return self.personal.get_month_summary(month, mode=mode)

    def search_notes(
        self,
        query: str,
        date_from: str | None = None,
        date_to: str | None = None,
        limit: int = 50,
    ) -> list[NoteEvidence]:
        return self.notes.search_notes(query=query, date_from=date_from, date_to=date_to, limit=limit)

    def search_thoughts(
        self,
        query: str,
        date_from: str | None = None,
        date_to: str | None = None,
        limit: int = 50,
    ) -> list[ThoughtEvidence]:
        return self.notes.search_thoughts(query=query, date_from=date_from, date_to=date_to, limit=limit)

    def get_monthly_reflection(self, month: str) -> MonthlyReflection:
        return self.notes.get_monthly_reflection(month)

    def evidence_bundle_for_question(self, question: str) -> EvidenceBundle:
        if "後" in question or "どこ" in question or "行" in question:
            return EvidenceBundle(
                line=tuple(self.personal.search_line("喧嘩 仲直り")),
                media=tuple(self.personal.search_media("喧嘩後 外出")),
                events=tuple(self.personal.search_events(event_type="date_or_outing")),
                places=tuple(self.personal.search_places("喧嘩後 外出")),
                notes=tuple(self.notes.search_notes("反省")),
            )
        return EvidenceBundle(
            line=tuple(self.personal.search_line("喧嘩")),
            events=tuple(self.personal.search_events(event_type="conflict")),
            notes=tuple(self.notes.search_notes("反省 不安")),
            thoughts=tuple(self.notes.search_thoughts("反省 不安")),
        )


def _in_range(value: str, date_from: str | None, date_to: str | None) -> bool:
    current = date.fromisoformat(value)
    if date_from and current < date.fromisoformat(date_from):
        return False
    if date_to and current > date.fromisoformat(date_to):
        return False
    return True


def _matches_query(item: AdapterEvidence, query: str) -> bool:
    terms = _query_terms(query)
    if not terms:
        return True
    tags = item.metadata.get("tags", ())
    haystack = " ".join(
        [
            item.summary,
            item.excerpt or "",
            " ".join(str(tag) for tag in tags),
        ]
    )
    return any(term in haystack for term in terms)


def _query_terms(query: str) -> list[str]:
    raw = query.strip()
    if not raw:
        return []
    terms = [part for part in raw.replace("　", " ").split(" ") if part]
    for token in ("喧嘩", "けんか", "すれ違い", "仲直り", "外出", "場所", "どこ", "行", "不安", "反省", "謝罪", "予定", "メモ"):
        if token in raw:
            terms.append(token)
    if "喧嘩" in raw or "けんか" in raw:
        terms.extend(["喧嘩", "すれ違い"])
    if "後" in raw or "あと" in raw or "どこ" in raw or "行" in raw:
        terms.extend(["喧嘩後", "外出", "場所", "カフェ", "食事"])
    if "仲直り" in raw or "通常" in raw:
        terms.extend(["仲直り", "通常会話"])
    return list(dict.fromkeys(terms))
