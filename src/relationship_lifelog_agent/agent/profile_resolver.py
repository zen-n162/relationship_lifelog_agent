from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from relationship_lifelog_agent.agent.question_understanding import QuestionFrame
from relationship_lifelog_agent.config import Settings
from relationship_lifelog_agent.db.repository import RowDict, RelationshipRepository
from relationship_lifelog_agent.profiles import ProfileContext, load_profile_context


@dataclass(frozen=True)
class ProfileResolution:
    status: str
    profile: ProfileContext | None
    candidates: tuple[RowDict, ...] = ()
    warnings: tuple[str, ...] = ()

    @property
    def profile_id(self) -> int | None:
        return self.profile.id if self.profile else None


def resolve_profile(
    settings: Settings,
    frame: QuestionFrame,
    *,
    selected_profile_id: int | None = None,
) -> ProfileResolution:
    explicit_id = selected_profile_id or frame.target_profile_id
    if explicit_id is not None:
        profile = load_profile_context(settings, explicit_id)
        if profile is None:
            return ProfileResolution("missing_profile", None, warnings=(f"profile id={explicit_id} は見つかりません。",))
        return ProfileResolution("resolved", profile, warnings=_profile_warnings(profile))

    default_profile = load_profile_context(settings, settings.relationship.default_profile_id)
    if default_profile is not None:
        return ProfileResolution("resolved", default_profile, warnings=_profile_warnings(default_profile))

    if not frame.target_profile_name:
        return ProfileResolution("missing_profile", None)

    db_path = Path(settings.paths.relationship_db).expanduser()
    if not db_path.exists():
        return ProfileResolution("missing_profile", None)
    repo = RelationshipRepository(db_path)
    candidates = [row for row in repo.list_profiles(limit=200) if row.get("profile_name") == frame.target_profile_name]
    if not candidates:
        return ProfileResolution("missing_profile", None)
    visible = [row for row in candidates if row.get("visibility") == "private"] or candidates
    ranked = sorted(visible, key=_profile_completeness_score, reverse=True)
    if len(ranked) > 1 and _profile_completeness_score(ranked[0]) == _profile_completeness_score(ranked[1]):
        return ProfileResolution("ambiguous_profile", None, candidates=tuple(ranked))
    profile = load_profile_context(settings, int(ranked[0]["id"]))
    return ProfileResolution("resolved", profile, candidates=tuple(ranked), warnings=_profile_warnings(profile))


def _profile_completeness_score(row: RowDict) -> int:
    score = 0
    if row.get("person_source_id"):
        score += 1
    if row.get("line_speaker_source_id") or row.get("line_speaker_group_source_id"):
        score += 1
    if row.get("self_line_speaker_source_id") or row.get("self_line_speaker_group_source_id"):
        score += 1
    return score


def _profile_warnings(profile: ProfileContext | None) -> tuple[str, ...]:
    if profile is None:
        return ()
    warnings: list[str] = []
    if not profile.target_line_speaker_source_id:
        warnings.append("target側のLINE speaker source IDが未設定です。")
    if not profile.self_line_speaker_filter_source_id:
        warnings.append("self側のLINE speaker source IDが未設定です。")
    return tuple(warnings)
