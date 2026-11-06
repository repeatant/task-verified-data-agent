"""Recover accounted truncation without allowing unknown spending to disappear."""

from copy import deepcopy

import pytest

from dataverify.runtime.discovery_budget import reviewed_truncation
from dataverify.runtime.probe_cost_budget import MonetaryRequestBudget


def example():
    budget = MonetaryRequestBudget(limit=5)
    start = budget.snapshot()
    budget.reserve(100)
    usage = {"prompt_tokens": 20, "completion_tokens": 900, "total_tokens": 920}
    budget.record_tokens(usage)
    budget.stop("PROVIDER_ERROR")
    end = budget.snapshot()
    report = {
        "budget_start": start,
        "budget_end": end,
        "http_events": [
            {
                "attempt": 1,
                "http_status": 200,
                "usage_complete": True,
                "status": "failed",
                "error_code": "GRAPH_QWEN_RESPONSE_TRUNCATED",
                "usage": usage,
            }
        ],
    }
    return end, report


def test_recovery_preserves_all_paid_usage_and_original_record():
    end, report = example()
    recovered = reviewed_truncation(end, report)
    assert recovered == {**end, "stopped_reason": None}
    assert end["stopped_reason"] == "PROVIDER_ERROR"


@pytest.mark.parametrize("change", ["usage", "code", "journal", "inflight", "events"])
def test_recovery_rejects_uncertain_or_changed_evidence(change):
    end, report = example()
    report = deepcopy(report)
    if change == "usage":
        report["http_events"][0]["usage"]["total_tokens"] += 1
    elif change == "code":
        report["http_events"][0]["error_code"] = "TRANSPORT_ERROR"
    elif change == "journal":
        end = {**end, "http_attempts": 3}
    elif change == "inflight":
        end["in_flight_count"] = 1
        report["budget_end"] = end
    else:
        report["http_events"] = []
    with pytest.raises(ValueError):
        reviewed_truncation(end, report)
