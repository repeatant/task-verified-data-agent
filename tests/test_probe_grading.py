"""Independent SQL grading must not depend on an Agent execution framework."""

import pytest

from dataverify.evaluation import probe_grading as bench
from dataverify.evaluation.probe_cases import get_case, reference_result


def record_for(case, *, sql=None, rows=None):
    return {
        "publication": {
            "status": "verified",
            "task_id": case.task.task_id,
            "verified_sql": sql or case.reference_sql,
            "rows": reference_result(case) if rows is None else rows,
        },
        "source_unchanged": True,
        "http_events": [],
        "actions": [],
        "llm_requests": [],
        "llm_calls": 1,
    }


@pytest.mark.parametrize("case_id", ["d01", "d03", "d04", "d16"])
def test_reference_passes_independent_source_and_boundary_grading(case_id):
    case = get_case(case_id)
    grade = bench.grade_probe_record(case, record_for(case))
    assert grade["correct_task"]
    assert grade["source_correct"] and grade["boundary_correct"]
    assert not grade["unscorable"]


@pytest.mark.parametrize("case_id", ["d05", "d06", "d07", "d10", "d11", "d12", "d13", "d14"])
def test_source_coincidence_is_not_counted_as_full_correctness(case_id):
    case = get_case(case_id)
    observed = bench.independent_query(case.candidate_sql, case.source)
    assert observed["status"] == "ok"
    grade = bench.grade_probe_record(
        case, record_for(case, sql=case.candidate_sql, rows=observed["rows"])
    )
    assert grade["source_correct"]
    assert not grade["boundary_correct"]
    assert not grade["correct_task"]
    assert grade["wrong_release"]


def test_false_published_rows_cannot_pass_by_hiding_behind_good_sql():
    case = get_case("d01")
    grade = bench.grade_probe_record(case, record_for(case, rows=[[1]]))
    assert grade["boundary_correct"]
    assert not grade["publication_matches_sql"]
    assert not grade["correct_task"]


@pytest.mark.parametrize(
    "sql",
    [
        "DELETE FROM orders",
        "ATTACH DATABASE 'x' AS y",
        "SELECT * FROM sqlite_master",
        "SELECT random()",
        "SELECT SUM(amount) FROM order_lines; DELETE FROM orders",
    ],
)
def test_independent_evaluator_has_own_read_boundary(sql):
    assert bench.independent_query(sql, get_case("d01").source)["status"] == "error"


def test_independent_grader_does_not_call_runtime_checker(monkeypatch):
    import dataverify.verification.probe_runtime as runtime

    monkeypatch.setattr(
        runtime,
        "static_check",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("must not grade itself")),
    )
    case = get_case("d01")
    assert bench.grade_probe_record(case, record_for(case))["correct_task"]


def test_operational_error_and_task_drift_are_not_correct_answers():
    case = get_case("d01")
    record = record_for(case)
    record["operational_error"] = "TIMEOUT"
    assert not bench.grade_probe_record(case, record)["correct_task"]
    record = record_for(case)
    record["publication"]["task_id"] = "other"
    assert bench.grade_probe_record(case, record)["wrong_release"]
