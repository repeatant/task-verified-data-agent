"""Shared JSON reporting helpers for offline and live evaluators.

This module owns serialization and usage summaries only. It has no model,
database, task fixture or credential dependency, so evaluators can reuse these
helpers without importing each other's execution pipelines.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


def write_json(path: Path, document: Any) -> None:
    """Replace a report atomically after serializing one complete document."""
    partial = path.with_suffix(".part.json")
    partial.write_text(
        json.dumps(document, ensure_ascii=False, indent=2, allow_nan=False),
        encoding="utf-8",
    )
    partial.replace(path)


def digest_json(value: Any) -> str:
    """Hash a stable JSON representation using the existing report contract."""
    encoded = json.dumps(value, ensure_ascii=False, sort_keys=True, allow_nan=False)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def summarize_usage(events: list[dict[str, Any]]) -> dict[str, Any]:
    """Count known usage while keeping missing/inconsistent events visible.

    A missing usage record is unknown, rather than a zero-cost model call.
    Boolean values are deliberately excluded from integer Token counts.
    """
    fields = ("prompt_tokens", "completion_tokens", "total_tokens")
    totals = {name + "_known": 0 for name in fields}
    unknown_events = 0
    for event in events:
        usage = event.get("usage") or {}
        complete = all(type(usage.get(name)) is int and usage[name] >= 0 for name in fields)
        consistent = (
            complete
            and usage["prompt_tokens"] + usage["completion_tokens"] == usage["total_tokens"]
        )
        if not complete or not consistent:
            unknown_events += 1
        for name in fields:
            if type(usage.get(name)) is int and usage[name] >= 0:
                totals[name + "_known"] += usage[name]
    return {
        **totals,
        "event_count": len(events),
        "unknown_usage_events": unknown_events,
        "usage_complete": unknown_events == 0,
    }
