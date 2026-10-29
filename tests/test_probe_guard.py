import json

import pytest

pytest.importorskip("hypothesis", reason="Optional probe experiment")

from dataverify.probe_cases import get_case
from dataverify.probe_guard import ProbeGuard


@pytest.mark.parametrize("mode", ["code", "full", "compact"])
def test_valid_candidate_publish_and_stale_result_clear(mode):
    case = get_case("d01")
    guard = ProbeGuard(case.task, case.source, mode)
    assert guard.execute(case.candidate_sql, case.task.task_id)["status"] == "verified"
    assert guard.publish()["rows"] == [[35000]]
    guard.invalidate()
    assert guard.publish()["rows"] is None
    assert guard.publish()["verified_sql"] is None


def test_feedback_modes_change_only_selected_evidence():
    case = get_case("d06")
    full = ProbeGuard(case.task, case.source, "full")
    compact = ProbeGuard(case.task, case.source, "compact")
    left = full.execute(case.candidate_sql, case.task.task_id)
    right = compact.execute(case.candidate_sql, case.task.task_id)
    l_text, r_text = left.pop("diagnostic_evidence"), right.pop("diagnostic_evidence")
    assert left == right
    assert len(r_text) < len(l_text)
    assert full.last_result["probe_report"]["counterexample"] == compact.last_result["probe_report"]["counterexample"]
    assert "reference_sql" not in json.dumps(right)


def test_injected_fault_then_repair_rechecks_old_witness():
    case = get_case("d06")
    guard = ProbeGuard(case.task, case.source)
    assert guard.execute(case.candidate_sql, case.task.task_id)["status"] == "unverified"
    assert guard.execute(case.reference_sql, case.task.task_id)["status"] == "verified"
    assert guard.last_result["regression_checks"]
    assert all(row["status"] == "passed" for row in guard.last_result["regression_checks"])


def test_confirmed_task_cannot_be_changed_by_action():
    case = get_case("d01")
    guard = ProbeGuard(case.task, case.source)
    result = guard.execute(case.candidate_sql, "invented-task")
    assert result["error_code"] == "TASK_DRIFT"
    assert guard.publish()["rows"] is None


def test_private_audit_cannot_mutate_publication():
    case = get_case("d01")
    guard = ProbeGuard(case.task, case.source)
    guard.execute(case.candidate_sql, case.task.task_id)
    audit = guard.audit
    audit[0]["source_execution"]["rows"][0][0] = -999
    assert guard.publish()["rows"] == [[35000]]


def test_code_feedback_never_contains_full_or_compact_private_data():
    case = get_case("d06")
    guard = ProbeGuard(case.task, case.source, "code")
    public = guard.execute(case.candidate_sql, case.task.task_id)
    assert "diagnostic_evidence" not in public
    assert "counterexample" not in public
    assert guard.last_result["probe_report"]["counterexample"]
