from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ToolSpec:
    name: str
    description: str
    required_inputs: tuple[str, ...]
    output_schema: str
    privacy_level: str
    can_use_in_public_mode: bool


TOOLS: tuple[ToolSpec, ...] = (
    ToolSpec("resolve_profile", "Resolve a manually configured relationship profile.", ("question",), "ProfileResolution", "private", True),
    ToolSpec("get_profile", "Load a relationship profile from the relationship DB.", ("profile_id",), "ProfileContext", "private", True),
    ToolSpec("search_line_conflict_signals", "Search read-only LINE evidence for conflict or misunderstanding signals.", ("profile", "date_range"), "LineEvidence[]", "private", False),
    ToolSpec("conflict_audit", "Aggregate conflict candidates by date using deterministic Python.", ("line_evidence", "events"), "AnalysisResult", "private", True),
    ToolSpec("search_relationship_events", "Read saved relationship event candidates.", ("profile_id",), "EventEvidence[]", "private", True),
    ToolSpec("search_date_near_notes", "Search date-near relationship notes and score relevance.", ("date_range",), "NoteEvidence[]", "private", False),
    ToolSpec("search_post_conflict_activities", "Find outings after conflict candidates.", ("conflict_dates",), "PostConflictActivity[]", "private", True),
    ToolSpec("get_monthly_reflection", "Load monthly reflection from notes adapter.", ("month",), "MonthlyReflection", "private", False),
    ToolSpec("search_media_events", "Search media/place event candidates through read-only personal adapter.", ("date_range",), "MediaEvidence[]", "sensitive", False),
    ToolSpec("calculate_reply_delay", "Calculate reply delay metrics with Python/SQL.", ("self_speaker", "target_speaker"), "InteractionMetric[]", "private", True),
    ToolSpec("compose_answer", "Compose cautious Japanese answer after deterministic analytics.", ("analysis_result",), "Markdown", "private", True),
)


def get_tool(name: str) -> ToolSpec | None:
    for tool in TOOLS:
        if tool.name == name:
            return tool
    return None


def list_tools() -> tuple[ToolSpec, ...]:
    return TOOLS
