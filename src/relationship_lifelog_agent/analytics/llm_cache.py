from __future__ import annotations

import hashlib
import json
from typing import Any


def source_window_hash(payload: dict[str, Any]) -> str:
    """Hash source refs and short summaries without storing raw private text."""

    encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()
