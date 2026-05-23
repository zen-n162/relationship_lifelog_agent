from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class LlmSummary:
    text: str
    confidence: float


QUESTION_FRAME_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "intents": {"type": "array", "items": {"type": "string"}},
        "target_profile_name": {"type": ["string", "null"]},
        "date_from": {"type": ["string", "null"]},
        "date_to": {"type": ["string", "null"]},
        "referenced_previous_answer": {"type": "boolean"},
        "requested_output": {"type": "string"},
        "needs_evidence": {"type": "boolean"},
        "risk_level": {"type": "string"},
    },
    "required": [
        "intents",
        "target_profile_name",
        "date_from",
        "date_to",
        "referenced_previous_answer",
        "requested_output",
        "needs_evidence",
        "risk_level",
    ],
}


INFORMATION_NEEDS_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "needs": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "name": {"type": "string"},
                    "required": {"type": "boolean"},
                    "reason": {"type": "string"},
                    "how_to_get": {"type": ["string", "null"]},
                },
                "required": ["name", "required", "reason", "how_to_get"],
            },
        }
    },
    "required": ["needs"],
}


QUERY_PLAN_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "plan_steps": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "tool_name": {"type": "string"},
                    "purpose": {"type": "string"},
                    "required": {"type": "boolean"},
                    "params": {"type": "object"},
                    "output_key": {"type": "string"},
                },
                "required": ["tool_name", "purpose", "required", "params", "output_key"],
            },
        },
        "compound_intent": {"type": ["string", "null"]},
        "sub_plan": {"type": ["array", "null"], "items": {"type": "string"}},
        "expected_answer_shape": {"type": "string"},
    },
    "required": ["plan_steps", "expected_answer_shape"],
}


ANSWER_COMPOSITION_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "answer_markdown": {"type": "string"},
    },
    "required": ["answer_markdown"],
}


LINE_WINDOW_ANALYSIS_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "is_conflict_candidate": {"type": "boolean"},
        "is_minor_misunderstanding": {"type": "boolean"},
        "severity": {"type": "integer", "minimum": 0, "maximum": 3},
        "confidence": {"type": "number", "minimum": 0.0, "maximum": 1.0},
        "estimated_start_date": {"type": ["string", "null"]},
        "estimated_end_date": {"type": ["string", "null"]},
        "summary": {"type": "string"},
        "evidence_message_refs": {"type": "array", "items": {"type": "string"}},
        "evidence_excerpt_short": {"type": "array", "items": {"type": "string"}},
        "is_joke_or_general_statement_possible": {"type": "boolean"},
        "cautions": {"type": "array", "items": {"type": "string"}},
    },
    "required": [
        "is_conflict_candidate",
        "is_minor_misunderstanding",
        "severity",
        "confidence",
        "estimated_start_date",
        "estimated_end_date",
        "summary",
        "evidence_message_refs",
        "evidence_excerpt_short",
        "is_joke_or_general_statement_possible",
        "cautions",
    ],
}


NOTE_WINDOW_ANALYSIS_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "relationship_relevance": {"type": "number", "minimum": 0.0, "maximum": 1.0},
        "date_relevance": {"type": "number", "minimum": 0.0, "maximum": 1.0},
        "own_thought_likelihood": {"type": "number", "minimum": 0.0, "maximum": 1.0},
        "lyrics_or_quote_likelihood": {"type": "number", "minimum": 0.0, "maximum": 1.0},
        "conflict_related": {"type": "boolean"},
        "summary": {"type": "string"},
        "confidence": {"type": "number", "minimum": 0.0, "maximum": 1.0},
        "include_as_supporting_evidence": {"type": "boolean"},
        "cautions": {"type": "array", "items": {"type": "string"}},
    },
    "required": [
        "relationship_relevance",
        "date_relevance",
        "own_thought_likelihood",
        "lyrics_or_quote_likelihood",
        "conflict_related",
        "summary",
        "confidence",
        "include_as_supporting_evidence",
        "cautions",
    ],
}


MEDIA_DAY_SUMMARY_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "date": {"type": "string"},
        "has_media": {"type": "boolean"},
        "likely_activities": {"type": "array", "items": {"type": "string"}},
        "place_summaries": {"type": "array", "items": {"type": "string"}},
        "people_roles_present": {"type": "array", "items": {"type": "string"}},
        "summary": {"type": "string"},
        "confidence": {"type": "number", "minimum": 0.0, "maximum": 1.0},
        "evidence_media_refs": {"type": "array", "items": {"type": "string"}},
        "cautions": {"type": "array", "items": {"type": "string"}},
    },
    "required": [
        "date",
        "has_media",
        "likely_activities",
        "place_summaries",
        "people_roles_present",
        "summary",
        "confidence",
        "evidence_media_refs",
        "cautions",
    ],
}
