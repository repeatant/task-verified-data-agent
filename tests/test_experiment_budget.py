"""Continuation validation retains paid usage without a framework dependency."""

import hashlib
import json
from copy import deepcopy

import pytest

from dataverify.runtime import experiment_budget as experiment
from dataverify.runtime.probe_cost_budget import MonetaryRequestBudget


@pytest.fixture
def report_root(tmp_path):
    reports = tmp_path / "project" / "reports"
    reports.mkdir(parents=True)
    return reports


def completed_budget():
    budget = MonetaryRequestBudget()
    budget.reserve(100)
    budget.record_tokens({"prompt_tokens": 120, "completion_tokens": 20, "total_tokens": 140})
    return budget


def test_continuation_preserves_every_spending_counter_and_only_changes_call_cap():
    snapshot = completed_budget().snapshot()
    restored = experiment.restore_budget(snapshot, max_extra_requests=24).snapshot()
    assert restored["http_attempt_limit"] == 25
    for key, value in snapshot.items():
        if key != "http_attempt_limit":
            assert restored[key] == value


@pytest.mark.parametrize(
    "field,value",
    [
        ("reported_cost_complete", False),
        ("stopped_reason", "ERROR"),
        ("unknown_count", 1),
        ("in_flight_count", 1),
        ("unmatched_usage_events", 1),
        ("reserved_upper_cny", 0),
        ("reported_total_tokens", 0),
        ("http_attempts", 0),
    ],
)
def test_bad_prior_ledgers_cannot_reset_the_allowance(field, value):
    snapshot = completed_budget().snapshot()
    snapshot[field] = value
    with pytest.raises(ValueError):
        experiment.restore_budget(snapshot)


def test_new_request_cap_is_enforced_without_erasing_complete_global_usage():
    budget = completed_budget()
    wrapper = experiment.IncrementalAllowance(budget, 1, 0.8)
    wrapper.reserve(100)
    wrapper.record_tokens({"prompt_tokens": 20, "completion_tokens": 10, "total_tokens": 30})
    with pytest.raises(Exception, match="ADDITIONAL_REQUEST_LIMIT"):
        wrapper.reserve(100)
    assert budget.snapshot()["http_attempts"] == 2
    assert budget.snapshot()["reported_cost_complete"]
    assert not budget.snapshot()["stopped_reason"]


def test_additional_money_reservation_blocks_before_the_post():
    budget = completed_budget()
    before = budget.snapshot()
    wrapper = experiment.IncrementalAllowance(budget, 24, 0.001)
    with pytest.raises(Exception, match="ADDITIONAL_RESERVATION_LIMIT"):
        wrapper.reserve(100)
    assert budget.snapshot() == before


@pytest.mark.parametrize("value", [0, 37, True, 1.5])
def test_invalid_extra_attempt_limit_rejected(value):
    with pytest.raises(ValueError):
        experiment.IncrementalAllowance(completed_budget(), value, 0.8)


def truncated_document():
    budget = completed_budget()
    start = deepcopy(budget.snapshot())
    budget.reserve(1000)
    usage = {"prompt_tokens": 1000, "completion_tokens": 900, "total_tokens": 1900}
    budget.record_tokens(usage)
    budget.stop("PROVIDER_ERROR")
    return {
        "real_model_used": True,
        "source_integrity_ok": True,
        "audit_ledger_complete": True,
        "audit_event_count": 1,
        "this_run_http_attempts": 1,
        "budget_start": start,
        "budget_end": budget.snapshot(),
        "rows": [
            {
                "case_id": "d03",
                "track": "natural",
                "arm": "neither",
                "status": "completed",
                "http_attempts": 1,
                "http_events": [
                    {
                        "attempt": 2,
                        "status": "failed",
                        "http_status": 200,
                        "error_code": "QWEN_RESPONSE_TRUNCATED",
                        "usage": usage,
                        "usage_complete": True,
                    }
                ],
            }
        ],
    }


def save_prior(report_root, document):
    path = report_root / "prior.json"
    path.write_text(json.dumps(document, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def test_explicit_truncation_review_preserves_source_and_all_paid_usage(report_root):
    original = truncated_document()
    path = save_prior(report_root, original)
    raw = path.read_bytes()
    with pytest.raises(ValueError, match="complete, unstopped"):
        experiment.load_prior_budget(path, project_root=report_root.parent)
    budget = experiment.load_prior_budget(
        path, project_root=report_root.parent, allow_truncated_resume=True
    )
    snapshot = budget.snapshot()
    for key in (
        "http_attempts",
        "reported_total_tokens",
        "reserved_upper_cny",
        "estimated_reported_cny",
        "reservation_ledger",
    ):
        assert snapshot[key] == original["budget_end"][key]
    assert snapshot["stopped_reason"] is None
    assert snapshot["max_output_tokens"] == 900
    assert path.read_bytes() == raw
    review = budget._continuation_review[0]
    assert review["reviewed_failed_attempt"] == 2
    assert review["original_stop_reason"] == "PROVIDER_ERROR"
    assert review["source_report"] == "reports/prior.json"
    assert not review["refund_applied"]
    assert review["source_report_sha256"] == hashlib.sha256(raw).hexdigest()


@pytest.mark.parametrize(
    "mutation",
    [
        "unknown_usage",
        "event_incomplete",
        "wrong_event_total",
        "wrong_error",
        "http_error",
        "wrong_stop",
        "event_count",
        "wrong_attempt",
        "duplicate_event",
        "ledger_usage",
        "audit_incomplete",
        "source_changed",
        "in_flight",
        "unmatched",
        "historical_ledger_changed",
    ],
)
def test_truncation_flag_cannot_bypass_unreviewed_failures(report_root, mutation):
    doc = truncated_document()
    event = doc["rows"][0]["http_events"][0]
    if mutation == "unknown_usage":
        doc["budget_end"]["unknown_count"] = 1
    elif mutation == "event_incomplete":
        event["usage_complete"] = False
    elif mutation == "wrong_event_total":
        event["usage"]["total_tokens"] += 1
    elif mutation == "wrong_error":
        event["error_code"] = "QWEN_INVALID_RESPONSE_ENVELOPE"
    elif mutation == "http_error":
        event["http_status"] = 500
    elif mutation == "wrong_stop":
        doc["budget_end"]["stopped_reason"] = "TRANSPORT_ERROR"
    elif mutation == "event_count":
        doc["this_run_http_attempts"] = 2
    elif mutation == "wrong_attempt":
        event["attempt"] = 1
    elif mutation == "duplicate_event":
        doc["rows"][0]["http_events"].append(deepcopy(event))
        doc["rows"][0]["http_attempts"] = 2
    elif mutation == "ledger_usage":
        doc["budget_end"]["reservation_ledger"][-1]["prompt_tokens"] += 1
    elif mutation == "audit_incomplete":
        doc["audit_ledger_complete"] = False
    elif mutation == "source_changed":
        doc["source_integrity_ok"] = False
    elif mutation == "in_flight":
        doc["budget_end"]["in_flight_count"] = 1
    elif mutation == "unmatched":
        doc["budget_end"]["unmatched_usage_events"] = 1
    elif mutation == "historical_ledger_changed":
        doc["budget_end"]["reservation_ledger"][0]["prompt_chars"] += 1
    with pytest.raises(ValueError):
        experiment.load_prior_budget(
            save_prior(report_root, doc),
            project_root=report_root.parent,
            allow_truncated_resume=True,
        )
