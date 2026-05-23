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
