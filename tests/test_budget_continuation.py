"""No-network checks for restoring a LangGraph request ledger exactly."""

import pytest

from dataverify.agent.graph_checkpoint import resume_budget
from dataverify.runtime.probe_cost_budget import MonetaryRequestBudget


def completed_snapshot():
    budget = MonetaryRequestBudget()
    budget.reserve(100)
    budget.record_tokens({"prompt_tokens": 100, "completion_tokens": 20, "total_tokens": 120})
    return budget.snapshot()


def test_continuation_reproduces_reservations_and_usage_exactly():
    snapshot = completed_snapshot()
    resumed = resume_budget(snapshot)
    assert resumed.snapshot() == snapshot
    assert resumed.reserve(200) == 2
    assert resumed.snapshot()["reserved_upper_cny"] > snapshot["reserved_upper_cny"]


@pytest.mark.parametrize(
    "field,value",
    [
        ("reported_cost_complete", False),
        ("stopped_reason", "USAGE_UNAVAILABLE"),
        ("unknown_count", 1),
        ("in_flight_count", 1),
        ("unmatched_usage_events", 1),
        ("http_attempts", 0),
        ("estimated_reported_cny", 0),
        ("reserved_upper_cny", 0),
    ],
)
def test_incomplete_or_inconsistent_ledger_cannot_restart_allowance(field, value):
    snapshot = completed_snapshot()
    snapshot[field] = value
    with pytest.raises(ValueError):
        resume_budget(snapshot)


def test_changed_price_or_missing_reservation_is_rejected():
    snapshot = completed_snapshot()
    snapshot["pricing"]["input_cny_per_million_tokens"] = 0
    with pytest.raises(ValueError):
        resume_budget(snapshot)
    snapshot = completed_snapshot()
    snapshot["reservation_ledger"] = []
    with pytest.raises(ValueError):
        resume_budget(snapshot)
