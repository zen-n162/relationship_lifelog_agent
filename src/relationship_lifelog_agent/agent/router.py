from __future__ import annotations

from dataclasses import dataclass
import re


@dataclass(frozen=True)
class RouteResult:
    question: str
    intents: tuple[str, ...]


def route_question(question: str) -> RouteResult:
    text = question.strip().lower()
    intents: list[str] = []

    if _has_any(text, "喧嘩", "けんか", "すれ違い", "もめ") and _has_any(text, "いつ", "日") and _has_any(text, "写真", "メディア", "前後"):
        intents.extend(("conflict_dates_with_surrounding_media", "conflict_date_lookup", "surrounding_media_summary"))
    elif _has_any(text, "喧嘩", "けんか", "すれ違い", "もめ") and _has_any(text, "いつ", "日"):
        intents.append("conflict_date_lookup")
    if _has_any(text, "写真", "メディア") and _has_any(text, "前後", "その日", "当日", "翌日", "前日"):
        if "surrounding_media_summary" not in intents:
            intents.append("surrounding_media_summary")
    if _has_any(text, "喧嘩", "けんか", "すれ違い", "もめ") and _has_any(text, "後", "あと", "どこ", "行っ"):
        intents.append("post_conflict_activity")
    if _has_any(text, "喧嘩", "けんか", "すれ違い", "もめ") and _has_any(text, "いつ", "時期", "タイムライン", "多かった"):
        intents.append("conflict_timeline")
    if _has_any(text, "喧嘩", "けんか", "すれ違い", "もめ") and _has_any(text, "どのくらい", "頻度", "何回", "どれくらい"):
        intents.append("conflict_frequency")
    if _has_any(text, "仲直り", "通常", "戻っ"):
        intents.append("reconciliation_duration")
    if _has_any(text, "line", "返信", "返せ", "返事", "沈黙"):
        intents.append("reply_delay_analysis")
    if _has_any(text, "約束", "予定", "行かなかった"):
        intents.append("promise_tracking")
    if _has_any(text, "会った", "外出", "遊び", "行って", "場所", "デート"):
        intents.append("outing_history")
    if _has_any(text, "考えて", "思って", "メモ", "ノート", "気持ち"):
        intents.append("emotional_note_lookup")
    if re.search(r"20\d{2}年\s*\d{1,2}月", question) or _has_any(text, "月", "振り返"):
        if _has_any(text, "関係", "振り返", "レビュー"):
            intents.append("monthly_relationship_review")
    if _has_any(text, "根拠", "証拠", "evidence"):
        intents.append("evidence_lookup")

    if not intents:
        intents.append("general_relationship_qa")
    return RouteResult(question=question, intents=tuple(dict.fromkeys(intents)))


def _has_any(text: str, *needles: str) -> bool:
    return any(needle in text for needle in needles)
