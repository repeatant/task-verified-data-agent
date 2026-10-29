"""Report helpers preserve unknown usage and complete JSON writes."""

import json

import pytest

from dataverify.reporting import digest_json, summarize_usage, write_json


def test_unknown_usage_is_visible_and_known_tokens_are_retained():
    events = [
        {"usage": {"prompt_tokens": 10, "completion_tokens": 2, "total_tokens": 12}},
        {"usage": {"prompt_tokens": 3, "completion_tokens": None, "total_tokens": None}},
        {"usage": {"prompt_tokens": True, "completion_tokens": 1, "total_tokens": 2}},
        {"usage": None},
    ]
    assert summarize_usage(events) == {
        "prompt_tokens_known": 13,
        "completion_tokens_known": 3,
        "total_tokens_known": 14,
        "event_count": 4,
        "unknown_usage_events": 3,
        "usage_complete": False,
    }


def test_inconsistent_usage_is_not_a_complete_cost_record():
    result = summarize_usage(
        [{"usage": {"prompt_tokens": 10, "completion_tokens": 2, "total_tokens": 99}}]
    )
    assert not result["usage_complete"]
    assert result["unknown_usage_events"] == 1
    assert result["total_tokens_known"] == 99


def test_report_serialization_failure_preserves_the_previous_document(tmp_path):
    path = tmp_path / "summary.json"
    write_json(path, {"status": "completed", "label": "反例"})
    with pytest.raises(ValueError):
        write_json(path, {"invalid": float("nan")})
    assert json.loads(path.read_text(encoding="utf-8")) == {"status": "completed", "label": "反例"}


def test_json_digest_ignores_mapping_order_but_preserves_values():
    assert digest_json({"a": 1, "b": 2}) == digest_json({"b": 2, "a": 1})
    assert digest_json({"amount": 1}) != digest_json({"amount": 2})
