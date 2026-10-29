"""Account for crash windows and stale writers without any model API."""

import json

import pytest

from dataverify.graph_checkpoint import (
    GraphResumeError,
    JournaledBudget,
    exclusive_run,
    read_budget,
    resume_budget,
    validate_budget_continuation,
)
from dataverify.probe_cost_budget import MonetaryRequestBudget


def usage():
    return {"prompt_tokens": 10, "completion_tokens": 2, "total_tokens": 12}


def test_reservation_is_on_disk_before_http_could_start(tmp_path):
    path = tmp_path / "budget.json"
    with exclusive_run(path):
        budget = JournaledBudget(MonetaryRequestBudget(limit=3), path)
        assert budget.reserve(100) == 1
        saved = read_budget(path)
        assert saved["in_flight_count"] == 1
        assert saved["reserved_upper_cny"] > 0
        # A process restart cannot treat an unaccounted POST as free/unmade.
        with pytest.raises(ValueError, match="complete, unstopped"):
            resume_budget(saved)
        budget.record_tokens(usage())
        restored = resume_budget(read_budget(path))
        assert restored.snapshot() == budget.snapshot()
        assert restored.limit == 3


def test_stale_budget_cannot_overwrite_new_spending(tmp_path):
    path = tmp_path / "budget.json"
    current = JournaledBudget(MonetaryRequestBudget(limit=3), path)
    current.reserve(100)
    current.record_tokens(usage())
    with pytest.raises(GraphResumeError, match="STALE_BUDGET_REFUSED"):
        JournaledBudget(MonetaryRequestBudget(limit=3), path)
    assert read_budget(path)["http_attempts"] == 1


def test_local_run_lease_rejects_concurrent_writer_and_releases(tmp_path):
    path = tmp_path / "run.db"
    with exclusive_run(path):
        with pytest.raises(GraphResumeError, match="CONCURRENT_RUN_REFUSED"):
            with exclusive_run(path):
                pytest.fail("second writer acquired the lease")
    with exclusive_run(path):
        pass


def test_accounting_prefix_can_advance_but_never_reset():
    budget = MonetaryRequestBudget(limit=3)
    old = budget.snapshot()
    budget.reserve(100)
    budget.record_tokens(usage())
    new = budget.snapshot()
    validate_budget_continuation(old, new)
    with pytest.raises(GraphResumeError, match="MODEL_BUDGET_RESET_REFUSED"):
        validate_budget_continuation(new, old)
    new["http_attempt_limit"] = 6
    with pytest.raises(GraphResumeError, match="MODEL_BUDGET_CHANGED"):
        validate_budget_continuation(old, new)


def test_unknown_usage_remains_explicit_on_disk(tmp_path):
    path = tmp_path / "budget.json"
    budget = JournaledBudget(MonetaryRequestBudget(), path)
    budget.reserve(20)
    budget.stop("TRANSPORT_ERROR")
    saved = json.loads(path.read_text(encoding="utf-8"))["budget"]
    assert saved["unknown_count"] == 1
    assert saved["reported_cost_complete"] is False
    with pytest.raises(GraphResumeError, match="UNRESOLVED_MODEL_SPENDING"):
        validate_budget_continuation(saved, saved)
