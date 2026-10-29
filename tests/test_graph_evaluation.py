"""Publication grading stays independent of the runtime's acceptance flag."""

from copy import deepcopy

import pytest

from dataverify.probe_cases import get_case, reference_result
from tools.evaluate_graph import evaluate_reports


def report(case_id="d01", thread="test"):
    case = get_case(case_id)
    return {
        "runtime": "langgraph",
        "thread_id": thread,
        "case_id": case_id,
        "binding": {"source": "one-confirmed-source"},
        "track": "controlled",
        "status": "completed",
        "model_attempts": 1,
        "actions": [],
        "http_events": [],
        "budget_start": None,
        "budget_end": None,
        "source_integrity_ok": True,
        "audit_ledger_complete": True,
        "publication": {
            "status": "verified",
            "task_id": case.task.task_id,
            "rows": reference_result(case),
            "verified_sql": case.reference_sql,
        },
    }


def test_resumed_process_is_not_an_additional_successful_task():
    paused, completed = report(), report()
    paused["status"] = "paused"
    paused["publication"] = {"status": "unverified", "rows": None}
    result = evaluate_reports([paused, completed])
    assert result["completed_threads"] == result["correct_threads"] == 1
    assert result["process_reports"] == 2
    assert result["resume_processes"] == 1


def test_published_rows_matching_source_do_not_hide_a_boundary_bug():
    record = report("d06")
    record["publication"]["verified_sql"] = get_case("d06").candidate_sql
    result = evaluate_reports([record])
    assert result["rows"][0]["grade"]["source_correct"] is True
    assert result["rows"][0]["grade"]["boundary_correct"] is False
    assert result["wrong_releases"] == 1
    assert result["correct_threads"] == 0


def test_resume_reports_cannot_merge_different_task_bindings():
    previous, changed = report(), report()
    changed["binding"] = {"source": "different-data"}
    with pytest.raises(ValueError, match="changed sources"):
        evaluate_reports([previous, changed])


def test_missing_source_or_accounting_integrity_remains_visible():
    record = report()
    record["audit_ledger_complete"] = False
    record["source_integrity_ok"] = False
    result = evaluate_reports([deepcopy(record)])
    assert result["audit_ledger_complete"] is False
    assert result["source_integrity_ok"] is False
