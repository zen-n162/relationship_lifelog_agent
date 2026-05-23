from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

from relationship_lifelog_agent.agent.information_needs import InformationNeed, split_needs
from relationship_lifelog_agent.agent.profile_resolver import ProfileResolution
from relationship_lifelog_agent.agent.question_understanding import QuestionFrame
from relationship_lifelog_agent.config import Settings


@dataclass(frozen=True)
class AnswerabilityReport:
    status: str
    needs: tuple[InformationNeed, ...]
    warnings: tuple[str, ...] = ()

    @property
    def available_needs(self) -> tuple[InformationNeed, ...]:
        return split_needs(self.needs)[0]

    @property
    def missing_needs(self) -> tuple[InformationNeed, ...]:
        return split_needs(self.needs)[1]

    @property
    def can_execute(self) -> bool:
        return self.status in {"answerable", "answerable_with_caution", "partially_answerable"}

    def to_dict(self) -> dict[str, object]:
        return {
            "status": self.status,
            "available_needs": [asdict(item) for item in self.available_needs],
            "missing_needs": [asdict(item) for item in self.missing_needs],
            "warnings": list(self.warnings),
        }


def check_answerability(
    frame: QuestionFrame,
    profile_resolution: ProfileResolution,
    settings: Settings,
) -> AnswerabilityReport:
    needs: list[InformationNeed] = []
    warnings: list[str] = list(profile_resolution.warnings)
    if profile_resolution.status == "ambiguous_profile":
        needs.append(
            InformationNeed(
                "manual_profile",
                True,
                "ambiguous",
                "同じ名前のprofileが複数あります。",
                "Chat UIまたはCLIでprofile idを明示してください。",
            )
        )
        return AnswerabilityReport("ambiguous_profile", tuple(needs), tuple(warnings))
    if profile_resolution.profile is None:
        target = frame.target_profile_name or "対象人物"
        needs.append(
            InformationNeed(
                "manual_profile",
                True,
                "missing",
                f"{target} の手動profileが必要です。",
                "person_source_id, line_speaker_source_id, self_line_speaker_source_id を手動設定してください。",
            )
        )
        return AnswerabilityReport("missing_profile", tuple(needs), tuple(warnings))

    profile = profile_resolution.profile
    needs.append(InformationNeed("manual_profile", True, "available", f"profile id={profile.id} を使用します。"))
    target_speaker_status = "available" if profile.target_line_speaker_source_id else "missing"
    needs.append(
        InformationNeed(
            "target_line_speaker",
            True,
            target_speaker_status,
            "対象側LINE speakerはLINE検索の絞り込みに使います。",
            "line_speaker_source_id または line_speaker_group_source_id を手動設定してください。",
        )
    )
    self_required = frame.primary_intent == "reply_delay_analysis"
    self_speaker_status = "available" if profile.self_line_speaker_filter_source_id else "missing"
    needs.append(
        InformationNeed(
            "self_line_speaker",
            self_required,
            self_speaker_status,
            "自分側LINE speakerは返信方向や返信間隔の分析に使います。",
            "self_line_speaker_source_id または self_line_speaker_group_source_id を手動設定してください。",
        )
    )
    upstream_status = _upstream_status(settings)
    needs.append(
        InformationNeed(
            "upstream_adapter",
            settings.adapter.backend == "upstream_readonly",
            upstream_status,
            "upstream_readonlyでは上流DBのread-only接続が必要です。",
            "config.local.yaml の personal_lifelog_db / notes_lifelog_db を確認してください。",
        )
    )
    if target_speaker_status == "missing":
        return AnswerabilityReport("missing_target_speaker", tuple(needs), tuple(warnings))
    if self_required and self_speaker_status == "missing":
        return AnswerabilityReport("missing_self_speaker", tuple(needs), tuple(warnings))
    if (
        settings.adapter.backend == "upstream_readonly"
        and upstream_status == "missing"
        and frame.requested_output == "line_detail_summary"
    ):
        return AnswerabilityReport("missing_upstream_adapter", tuple(needs), tuple(warnings))
    if settings.adapter.backend == "upstream_readonly" and upstream_status == "missing":
        return AnswerabilityReport("partially_answerable", tuple(needs), tuple(warnings))
    if self_speaker_status == "missing":
        return AnswerabilityReport("partially_answerable", tuple(needs), tuple(warnings))
    return AnswerabilityReport("answerable", tuple(needs), tuple(warnings))


def _upstream_status(settings: Settings) -> str:
    if settings.adapter.backend != "upstream_readonly":
        return "optional"
    personal = Path(settings.paths.personal_lifelog_db).expanduser() if settings.paths.personal_lifelog_db else None
    notes = Path(settings.paths.notes_lifelog_db).expanduser() if settings.paths.notes_lifelog_db else None
    return "available" if personal and notes and personal.exists() and notes.exists() else "missing"
