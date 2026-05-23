from __future__ import annotations

from dataclasses import asdict, is_dataclass
from math import ceil
from typing import Any, Iterable


DEFAULT_NUM_CTX = 32768
SINGLE_CONTEXT_RATIO = 0.75


def estimate_tokens_for_text(text: str | None) -> int:
    if not text:
        return 0
    return max(1, ceil(len(text) / 4))


def estimate_chars_for_item(item: Any) -> int:
    return len(_budget_text_for_item(item))


def estimate_tokens_for_item(item: Any) -> int:
    return estimate_tokens_for_text(_budget_text_for_item(item))


def estimate_bundle_chars(
    *,
    line_items: Iterable[Any] = (),
    note_items: Iterable[Any] = (),
    media_items: Iterable[Any] = (),
    face_items: Iterable[Any] = (),
    location_items: Iterable[Any] = (),
) -> int:
    return sum(
        estimate_chars_for_item(item)
        for collection in (line_items, note_items, media_items, face_items, location_items)
        for item in collection
    )


def estimate_bundle_tokens(
    *,
    line_items: Iterable[Any] = (),
    note_items: Iterable[Any] = (),
    media_items: Iterable[Any] = (),
    face_items: Iterable[Any] = (),
    location_items: Iterable[Any] = (),
) -> int:
    return sum(
        estimate_tokens_for_item(item)
        for collection in (line_items, note_items, media_items, face_items, location_items)
        for item in collection
    )


def decide_context_mode(manifest: Any, llm_config: Any) -> str:
    num_ctx = _num_ctx_from_config(llm_config)
    threshold = int(num_ctx * SINGLE_CONTEXT_RATIO)
    if int(getattr(manifest, "estimated_tokens", 0)) <= threshold:
        return "single_context"
    return "iterative_full_scan"


def can_fit_single_context(estimated_tokens: int, llm_config: Any | None = None) -> bool:
    return estimated_tokens <= int(_num_ctx_from_config(llm_config) * SINGLE_CONTEXT_RATIO)


def _num_ctx_from_config(llm_config: Any | None) -> int:
    if llm_config is None:
        return DEFAULT_NUM_CTX
    if isinstance(llm_config, dict):
        return int(llm_config.get("num_ctx") or DEFAULT_NUM_CTX)
    return int(getattr(llm_config, "num_ctx", DEFAULT_NUM_CTX) or DEFAULT_NUM_CTX)


def _budget_text_for_item(item: Any) -> str:
    data = _safe_item_dict(item)
    item_type = type(item).__name__
    parts = [item_type]
    for key, value in data.items():
        if value is None or value == "" or value == () or value == {}:
            continue
        if key == "face_embeddings":
            parts.append(f"{key}=embedding_count:{len(value)}")
            continue
        if key == "embedding":
            parts.append(f"{key}=embedding_dim:{len(value) if value else 0}")
            continue
        parts.append(f"{key}={_stringify_budget_value(value)}")
    return "\n".join(parts)


def _safe_item_dict(item: Any) -> dict[str, Any]:
    if is_dataclass(item):
        return asdict(item)
    if isinstance(item, dict):
        return dict(item)
    return {
        key: getattr(item, key)
        for key in dir(item)
        if not key.startswith("_") and not callable(getattr(item, key))
    }


def _stringify_budget_value(value: Any) -> str:
    if isinstance(value, float):
        return f"{value:.6g}"
    if isinstance(value, (list, tuple)):
        return "[" + ", ".join(_stringify_budget_value(item) for item in value) + "]"
    if isinstance(value, dict):
        return "{" + ", ".join(f"{key}:{_stringify_budget_value(val)}" for key, val in value.items()) + "}"
    return str(value)
