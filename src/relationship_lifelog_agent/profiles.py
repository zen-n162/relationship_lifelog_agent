from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from relationship_lifelog_agent.config import Settings
from relationship_lifelog_agent.db.repository import RowDict, RelationshipRepository


PROFILE_NONE_VALUE = "__none__"


@dataclass(frozen=True)
class ProfileContext:
    id: int
    profile_name: str
    person_source_id: str | None
    line_speaker_source_id: str | None
    relationship_label: str | None
    label_source: str


def get_repository(settings: Settings) -> RelationshipRepository:
    return RelationshipRepository(settings.paths.relationship_db)


def load_profile_context(settings: Settings, profile_id: int | None) -> ProfileContext | None:
    if profile_id is None:
        return None
    db_path = Path(settings.paths.relationship_db).expanduser()
    if not db_path.exists():
        return None
    row = RelationshipRepository(db_path).get_profile(profile_id)
    if row is None:
        return None
    return ProfileContext(
        id=int(row["id"]),
        profile_name=str(row["profile_name"]),
        person_source_id=row.get("person_source_id"),
        line_speaker_source_id=row.get("line_speaker_source_id"),
        relationship_label=row.get("relationship_label"),
        label_source=str(row["label_source"]),
    )


def profile_choices(settings: Settings) -> list[tuple[str, str]]:
    db_path = Path(settings.paths.relationship_db).expanduser()
    if not db_path.exists():
        return [("未設定", PROFILE_NONE_VALUE)]
    profiles = RelationshipRepository(db_path).list_profiles()
    choices = [(_profile_choice_label(row), str(row["id"])) for row in profiles]
    return choices or [("未設定", PROFILE_NONE_VALUE)]


def default_profile_choice(settings: Settings, choices: list[tuple[str, str]]) -> str:
    default_value = str(settings.relationship.default_profile_id)
    values = {value for _, value in choices}
    if default_value in values:
        return default_value
    return PROFILE_NONE_VALUE


def parse_profile_choice(value: str | int | None) -> int | None:
    if value is None or value == PROFILE_NONE_VALUE:
        return None
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return None
    return parsed if parsed > 0 else None


def is_relationship_profile_question(question: str) -> bool:
    return any(
        token in question
        for token in (
            "関係",
            "恋人",
            "パートナー",
            "喧嘩",
            "仲直り",
            "LINE",
            "返せ",
            "外出",
            "デート",
            "会った",
            "約束",
            "メモ",
            "振り返",
        )
    )


def profile_missing_guidance() -> str:
    return (
        "relationship profile が手動設定されていないため、この質問には具体的な関係対象を前提にして回答しません。"
        "人物ID、LINE speaker ID、relationship_label はユーザーの手動設定だけを使います。"
        "AIが恋人ラベルや person と LINE speaker の対応を推定することはありません。"
    )


def _profile_choice_label(row: RowDict) -> str:
    return f"{row['id']}: {row['profile_name']}"
